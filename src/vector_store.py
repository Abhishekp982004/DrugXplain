import chromadb
import pandas as pd
import os

# --- Configuration ---
INPUT_CSV = 'explanations.csv'
DB_PATH = "./chroma_db"  # This will create a folder named 'chroma_db'
COLLECTION_NAME = "drug_interactions"

# --- Main Script Logic ---

def main():
    """
    Main function to read explanations and load them into Chroma DB.
    """
    
    print(f"Starting vector store creation...")
    
    # 1. Read the input CSV (from your llm_explainer.py script)
    try:
        df = pd.read_csv(INPUT_CSV)
        # Drop any rows where the explanation might be an error
        df = df[~df['explanation'].str.startswith("Error:")].dropna()
        if df.empty:
            print(f"No valid explanations found in '{INPUT_CSV}'. Stopping.")
            return

    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_CSV}' not found.")
        print("Please run 'llm_explainer.py' first to generate it.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: '{INPUT_CSV}' is empty. No data to process.")
        return

    # 2. Initialize Chroma DB client
    # We use PersistentClient to save the DB to disk in the DB_PATH
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
        
    client = chromadb.PersistentClient(path=DB_PATH)

    # 3. Create or get the collection
    # The default embedding model is great and will be downloaded automatically
    try:
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' loaded/created.")
    except Exception as e:
        print(f"Error initializing Chroma DB collection: {e}")
        return

    # 4. Prepare data for Chroma DB
    
    # We need to convert our DataFrame rows into lists
    documents = df['explanation'].tolist()
    
    # Create metadata for each document. This is CRITICAL for your app.
    # It allows you to search/filter by drug name or risk score.
    metadatas = df[['drug_a', 'drug_b', 'risk_score']].to_dict('records')
    
    # Create a unique ID for each entry. We can just use the index.
    ids = [str(i) for i in range(len(documents))]
    
    # 5. Add the data to the collection
    # ChromaDB handles embedding the 'documents' list automatically
    # We use 'upsert' which will add new items or update existing ones.
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
    print(f"Data is saved in the '{DB_PATH}' directory.")
    print(f"Total entries in collection: {collection.count()}")


# --- Run the Script ---

if __name__ == "__main__":
    main()
