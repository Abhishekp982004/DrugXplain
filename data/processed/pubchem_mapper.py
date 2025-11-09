"""
pubchem_mapper.py
--------------------------------------
Maps PubChem CIDs (numeric IDs from STITCH or DDI datasets)
to readable drug names using the PubChem REST API.
Caches all results locally for instant reuse.
Falls back to IUPACName or Synonyms if no Title is available.
Automatically retries failed lookups.
--------------------------------------
"""

import pandas as pd
import requests
import time
import json
import os

# ==== Configuration ====
INPUT_FILE = "predicted_interactions.csv"   # dataset
OUTPUT_FILE = "drug_map_pubchem.csv"        # final mapping
CACHE_FILE = "pubchem_name_cache.json"      # local cache
SLEEP_BETWEEN_CALLS = 0.3                   # delay between API calls (in seconds)


# ==== Cache management ====

def load_cache():
    """Load cached names to avoid repeated API calls."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    """Save updated cache to disk."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ==== PubChem lookup ====

def get_pubchem_name(cid, cache, retries=2):
    """Query PubChem for the compound name, with caching, retry, and fallbacks."""
    cid_str = str(cid)
    if cid_str in cache:
        return cache[cid_str]

    # Helper for logging
    def log(msg):
        print(f"{msg}")

    # Title and IUPACName fields
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Title,IUPACName/JSON"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            props = data["PropertyTable"]["Properties"][0]
            name = props.get("Title") or props.get("IUPACName")
            if name:
                cache[cid_str] = name
                save_cache(cache)
                log(f" {cid} → {name}")
                return name
        else:
            log(f" No 'Title/IUPACName' found for {cid}")
    except Exception as e:
        log(f" Error for {cid}: {e}")
        if retries > 0:
            log(f" Retrying {cid} ...")
            time.sleep(2)
            return get_pubchem_name(cid, cache, retries=retries-1)

    # Fallback: synonyms
    try:
        syn_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
        r = requests.get(syn_url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            syns = data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
            if syns:
                name = syns[0]
                cache[cid_str] = name
                save_cache(cache)
                log(f"🔄 {cid} → {name} (from synonym)")
                return name
    except Exception as e:
        log(f" Synonym lookup failed for {cid}: {e}")

    # If all fails
    cache[cid_str] = "Unknown"
    save_cache(cache)
    log(f" {cid} → Unknown")
    return "Unknown"


# ==== Main mapping logic ====

def main():
    print(f" Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)

    # Extract all unique CIDs
    ids = pd.unique(df[["drug_a", "drug_b"]].values.ravel())
    print(f" Found {len(ids)} unique IDs")

    cache = load_cache()
    mapping = []

    for i, cid in enumerate(ids, 1):
        name = get_pubchem_name(cid, cache)
        mapping.append((cid, name))
        time.sleep(SLEEP_BETWEEN_CALLS)

        if i % 20 == 0:
            print(f" Progress: {i}/{len(ids)} done")

    # Save to CSV
    mapping_df = pd.DataFrame(mapping, columns=["PubChem_CID", "Drug_Name"])
    mapping_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n Mapping saved to {OUTPUT_FILE}")
    print(f" Cache stored in {CACHE_FILE}")
    print("\nSample preview:")
    print(mapping_df.head())


if __name__ == "__main__":
    main()
