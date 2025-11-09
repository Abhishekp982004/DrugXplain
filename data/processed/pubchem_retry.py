"""
pubchem_retry_failed.py
--------------------------------------
Retries PubChem lookups for any IDs marked as 'Unknown'
in your existing cache file.
--------------------------------------
"""

import requests
import json
import time

CACHE_FILE = "pubchem_name_cache.json"
SLEEP_BETWEEN_CALLS = 0.3

def load_cache():
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def get_pubchem_name(cid):
    """Try multiple endpoints to get a name."""
    cid_str = str(cid)
    # Title + IUPAC
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Title,IUPACName/JSON"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            props = data["PropertyTable"]["Properties"][0]
            name = props.get("Title") or props.get("IUPACName")
            if name:
                print(f" {cid} → {name}")
                return name
    except Exception:
        pass

    # Synonyms
    try:
        syn_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
        r = requests.get(syn_url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            syns = data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
            if syns:
                print(f" {cid} → {syns[0]} (from synonym)")
                return syns[0]
    except Exception:
        pass

    print(f" {cid} still unresolved")
    return "Unknown"

def main():
    cache = load_cache()
    unknowns = [cid for cid, name in cache.items() if name == "Unknown"]
    print(f" Retrying {len(unknowns)} unresolved IDs...")

    for i, cid in enumerate(unknowns, 1):
        new_name = get_pubchem_name(cid)
        cache[cid] = new_name
        save_cache(cache)
        time.sleep(SLEEP_BETWEEN_CALLS)
        if i % 10 == 0:
            print(f" Progress: {i}/{len(unknowns)} done")

    print("\n Retry complete! Cache updated.")

if __name__ == "__main__":
    main()
