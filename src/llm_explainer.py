import pandas as pd
import ollama  # <-- Changed from huggingface_hub
import os

# --- Configuration ---
INPUT_CSV = '../data/processed/predicted_interactions.csv'
OUTPUT_CSV = 'explanations.csv'
MODEL_NAME = 'gemma:2b'  # <-- Changed to your selected Ollama model

# --- LLM Function ---

def get_llm_explanation(drug_a, drug_b, score):
    """
    Generates a clinical explanation for a drug-drug interaction
    using a local Ollama model.
    """
    
    # 1. Craft a specific prompt
    prompt = f"""
    You are a clinical pharmacologist.
    A GNN model predicted a high-risk interaction (score: {score:.2f}) 
    between {drug_a} and {drug_b}.
    
    Please provide a concise, clinically interpretable explanation for 
    this adverse drug-drug interaction.
    
    Explanation:
    """

    print(f"Querying Ollama ({MODEL_NAME}) for: {drug_a} + {drug_b}...")

    try:
        # 3. Make the API Call to Ollama
        # This connects to 'http://localhost:11434' by default
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            options={
                'temperature': 0.5, # Lower temp for more factual answers
            }
        )
        
        # 4. Parse the response
        explanation = response['message']['content'].strip()
        return explanation

    except Exception as e:
        # Updated error message to be specific to Ollama
        print(f"\n--- CRITICAL ERROR ---")
        print(f"Error calling Ollama: {e}")
        print("Is the Ollama server running?")
        print(f"In a separate terminal, make sure you have run: 'ollama run {MODEL_NAME}'")
        print("----------------------\n")
        return "Error: Could not generate explanation."

# --- Main Script Logic (Same as before) ---

def main():
    """
    Main function to read, process, and write the CSV data.
    """
    
    print(f"Starting LLM explanation generation with Ollama...")
    
    # 1. Read the input CSV
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_CSV}' not found.")
        print("Please make sure the file exists (or create a mock version).")
        return

    # 2. Create a new column for explanations
    df['explanation'] = df.apply(
        lambda row: get_llm_explanation(row['drug_a'], row['drug_b'], row['risk_score']),
        axis=1
    )
    
    # 3. Save the new DataFrame to the output CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nProcessing complete!")
    print(f"Explanations saved to '{OUTPUT_CSV}'.")
    print("\n--- Output Data Sample ---")
    print(df.head())


# --- Run the Script ---

if __name__ == "__main__":
    main()
