import pandas as pd
import ollama
import os
import subprocess
import json
from flask import Flask, render_template_string
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
INPUT_CSV = '../data/processed/predicted_interactions.csv'
OUTPUT_CSV = 'explanations.csv'
CACHE_FILE = 'explanation_cache.json'
PREFERRED_MODELS = ['mistral:latest', 'mistral:7b-instruct-q4', 'gemma:2b']
MAX_WORKERS = 6  # Number of parallel threads (tune based on GPU VRAM)


# ============================================================
# 🔍 Model Detection
# ============================================================

def get_available_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().splitlines()
        return [line.split()[0] for line in lines if line and not line.startswith('NAME')]
    except Exception:
        return []


def select_model():
    available = get_available_models()
    for model in PREFERRED_MODELS:
        model_name = model.split(':')[0]
        if any(model_name in a for a in available):
            print(f"✅ Using Ollama model: {model}")
            return model
    print("⚠️ No preferred model found locally. Defaulting to Mistral (will try to pull if needed).")
    return 'mistral:latest'


MODEL_NAME = select_model()


# ============================================================
# 💾 Cache Management
# ============================================================

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ============================================================
# 🧠 LLM Explanation Generator
# ============================================================

def get_llm_explanation(drug_a, drug_b, score, cache):
    drug_a, drug_b = str(drug_a).strip(), str(drug_b).strip()
    cache_key = f"{drug_a}|{drug_b}"

    if cache_key in cache:
        print(f"💾 Using cached explanation for: {drug_a} + {drug_b}")
        return cache[cache_key]

    prompt = f"""
    You are a clinical pharmacologist.
    A GNN model predicted a high-risk interaction (score: {score:.2f}) 
    between {drug_a} and {drug_b}.
    Please provide a concise, clinically interpretable explanation 
    for this adverse drug-drug interaction.
    """

    print(f"🧠 Querying Ollama ({MODEL_NAME}) for: {drug_a} + {drug_b}...")

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.5}
        )
        explanation = response['message']['content'].strip()
        cache[cache_key] = explanation
        save_cache(cache)
        return explanation
    except Exception as e:
        print(f"❌ Error generating {drug_a}+{drug_b}: {e}")
        return "Error: Could not generate explanation."


# ============================================================
# ⚡ Parallelized Processing
# ============================================================

def generate_explanations():
    print(f"🚀 Starting parallel LLM generation with Ollama ({MODEL_NAME}) using {MAX_WORKERS} threads...")

    if not os.path.exists(INPUT_CSV):
        print(f"❌ Input file '{INPUT_CSV}' not found.")
        return None

    df = pd.read_csv(INPUT_CSV)
    df['drug_a'], df['drug_b'] = df['drug_a'].astype(str), df['drug_b'].astype(str)

    cache = load_cache()
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_llm_explanation, row['drug_a'], row['drug_b'], row['risk_score'], cache): (row['drug_a'], row['drug_b'])
            for _, row in df.iterrows()
        }
        for future in as_completed(futures):
            drug_a, drug_b = futures[future]
            try:
                result = future.result()
                results.append((drug_a, drug_b, result))
            except Exception as e:
                print(f"⚠️ Error processing {drug_a}+{drug_b}: {e}")
                results.append((drug_a, drug_b, "Error"))

    df['explanation'] = [r[2] for r in results]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Explanations saved to '{OUTPUT_CSV}'. Cache updated ({len(cache)} entries).")

    return df


# ============================================================
# 🌐 Web Viewer
# ============================================================

def launch_web_viewer():
    app = Flask(__name__)

    @app.route('/')
    def home():
        if not os.path.exists(OUTPUT_CSV):
            return "<h3>No explanations.csv found. Run generation first.</h3>"
        df = pd.read_csv(OUTPUT_CSV)
        html = df.to_html(classes='table table-striped table-bordered', index=False, escape=False)
        return render_template_string("""
        <html>
        <head>
            <title>DrugXplain Dashboard</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        </head>
        <body class="p-4">
            <h2>💊 DrugXplain – Drug Interaction Explanations</h2>
            <p>Model used: <b>{{ model }}</b></p>
            {{ html | safe }}
        </body>
        </html>
        """, html=html, model=MODEL_NAME)

    print("\n🌐 Web viewer live at http://localhost:5000 ...")
    app.run(port=5000, debug=False)


# ============================================================
# 🚀 Entry Point
# ============================================================

if __name__ == "__main__":
    df = generate_explanations()
    if df is not None:
        web_thread = threading.Thread(target=launch_web_viewer)
        web_thread.start()
