"""
Preprocess STITCH chemical-chemical links (textmining-based drug-drug graph)
"""

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

RAW_FILE = "../data/raw/chemical_chemical.links.v5.0.tsv.gz"  # your file
OUT_FILE = "../data/processed/encoded_ddi.csv"

def load_stitch_links(path):
    df = pd.read_csv(path, sep='\t')
    print(f"Loaded file with {len(df)} rows and columns: {list(df.columns)}")

    # Check available columns
    if 'combined_score' in df.columns:
        df = df[df['combined_score'] > 700]
        print(f"Filtered to {len(df)} rows with combined_score > 700.")
    elif 'textmining' in df.columns:
        # Use textmining confidence instead
        df = df[df['textmining'] > 300]   # you can adjust threshold
        print(f"Filtered to {len(df)} rows with textmining > 300.")
    else:
        print("No score column found — no filtering applied.")

    # Rename and clean
    if 'chemical1' in df.columns and 'chemical2' in df.columns:
        df = df.rename(columns={'chemical1': 'drug_a', 'chemical2': 'drug_b'})
    else:
        raise KeyError("Expected columns 'chemical1' and 'chemical2' not found.")

    df = df[['drug_a', 'drug_b']].drop_duplicates()
    df['interaction'] = 1
    print(f"Final dataset: {len(df)} drug-drug pairs.")
    return df

def encode_drugs(df):
    le = LabelEncoder()
    le.fit(pd.concat([df['drug_a'], df['drug_b']]).unique())
    df['drug_a'] = le.transform(df['drug_a'])
    df['drug_b'] = le.transform(df['drug_b'])
    print(f"Encoded {len(le.classes_)} unique drugs.")
    return df, le

def preprocess():
    os.makedirs("../data/processed", exist_ok=True)
    df = load_stitch_links(RAW_FILE)
    df, _ = encode_drugs(df)
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved processed file to {OUT_FILE}")

if __name__ == "__main__":
    preprocess()
