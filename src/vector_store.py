import chromadb
import pandas as pd
import os
import json

# --- Configuration ---
# The CSV from Person 2 (for risk scores)
SCORE_CSV_PATH = '../data/processed/predicted_interactions.csv'
# The new JSON file you just got (for names and explanations)
EXPLANATION_JSON_PATH = 'explanation_cache_with_names.json' 
DB_PATH = "./chroma_db"
COLLECTION_NAME = "drug_interactions"

def load_scores(score_path):
    """Loads the drug pairs and risk scores from the GNN output."""
    try:
        df_scores = pd.read_csv(score_path)
        # Convert IDs to string for merging
        df_scores['drug_a'] = df_scores['drug_a'].astype(str)
        df_scores['drug_b'] = df_scores['drug_b'].astype(str)
        print(f"Loaded {len(df_scores)} scores from {score_path}")
        return df_scores
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load score file: {e}")
        return None

def load_explanations(explanation_path):
    """Loads the new JSON explanation file."""
    try:
        with open(explanation_path, 'r', encoding='utf-8') as f:
            explanations_list = json.load(f)
        df_explanations = pd.DataFrame(explanations_list)
        # Convert IDs to string for merging
        df_explanations['drug_a_id'] = df_explanations['drug_a_id'].astype(str)
        df_explanations['drug_b_id'] = df_explanations['drug_b_id'].astype(str)
        print(f"Loaded {len(df_explanations)} explanations from {explanation_path}")
        return df_explanations
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load explanation JSON: {e}")
        return None

def main():
    print("Starting vector store creation...")
    
    df_scores = load_scores(SCORE_CSV_PATH)
    df_explanations = load_explanations(EXPLANATION_JSON_PATH)

    if df_scores is None or df_explanations is None:
        print("Missing critical files. Exiting.")
        return

    # --- 1. Merge the two files ---
    # We merge the scores and explanations together using the drug IDs
    try:
        df_final = pd.merge(
            df_explanations,
            df_scores,
            left_on=['drug_a_id', 'drug_b_id'],
            right_on=['drug_a', 'drug_b']
        )
        # Clean up extra columns from the merge
        df_final = df_final.drop(columns=['drug_a', 'drug_b'])
        
    except Exception as e:
        print(f"Error merging files: {e}")
        print("Please check that the column names in 'predicted_interactions.csv' are 'drug_a' and 'drug_b'")
        return

    if df_final.empty:
        print("No matching records found between files. Please check the drug IDs.")
        return

    print(f"Successfully merged {len(df_final)} records.")
    
    # --- 2. Initialize Chroma DB ---
    if os.path.exists(DB_PATH):
        print(f"Deleting old database at {DB_PATH}...")
        import shutil
        shutil.rmtree(DB_PATH)
        
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
        
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' created.")

    # --- 3. Prepare data for Chroma ---
    documents = df_final['explanation'].tolist()
    metadatas = df_final.to_dict('records') # Store everything as metadata
    ids = [f"{row['drug_a_id']}_{row['drug_b_id']}" for _, row in df_final.iterrows()]
    
    # --- 4. Add data to Chroma ---
    try:
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully upserted {len(documents)} entries to the vector store.")
    except Exception as e:
        print(f"Critical Error: Could not add or upsert data: {e}")
        return
            
    print("\n--- Vector store creation complete! ---")
    print(f"Total entries in collection: {collection.count()}")

if __name__ == "__main__":
    main()