import streamlit as st
import chromadb
import os
import pandas as pd
# from st_agraph import agraph, Node, Edge, Config # REMOVED THIS LINE

# --- Configuration ---
DB_PATH = "./chroma_db"  # Path to your persistent database
COLLECTION_NAME = "drug_interactions" # Must match vector_store.py
EXPLANATIONS_CSV = "explanations.csv" # Path to your LLM-generated CSV

# --- Page Setup ---
st.set_page_config(
    page_title="Drug Interaction Explainer",
    page_icon="💊",
    layout="wide"
)
st.title("Adverse Drug Interaction Explainer 🚀")
st.markdown("This dashboard provides a multi-tool analysis of the GNN's predicted drug-drug interactions.")

# --- Caching Functions (for speed) ---

@st.cache_resource
def get_db_collection():
    """
    Connects to the persistent Chroma DB and returns the collection.
    Caches the connection so it doesn't reconnect on every script rerun.
    """
    if not os.path.exists(DB_PATH):
        st.error(f"Database folder not found at {DB_PATH}. Have you run 'vector_store.py' yet?")
        return None
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection
    except Exception as e:
        st.error(f"Error connecting to Chroma DB: {e}")
        return None

@st.cache_data
def load_data(csv_path):
    """Loads the full CSV for browsing and graph building."""
    if not os.path.exists(csv_path):
        st.error(f"Explanation file not found at {csv_path}. Have you run 'llm_explainer.py' yet?")
        return None
    try:
        df = pd.read_csv(csv_path)
        # Convert drug IDs to strings for the graph
        df['drug_a'] = df['drug_a'].astype(str)
        df['drug_b'] = df['drug_b'].astype(str)
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

# --- Load Data ---
collection = get_db_collection()
df_explanations = load_data(EXPLANATIONS_CSV)

if not collection or df_explanations is None:
    st.error("A required data source is missing. Please run the data pipeline scripts first.")
else:
    # --- Create Tabs ---
    # UPDATED: Removed tab3
    tab1, tab2 = st.tabs(["Search Explanations (LLM)", "Browse All Interactions (Table)"])

    # --- Tab 1: Semantic Search (Your current app, but improved) ---
    with tab1:
        st.header("Search by Symptom or Drug")
        
        # --- Sidebar for Search Controls ---
        with st.sidebar:
            st.header("Search Filters")
            query_text = st.text_input(
                "Search by symptom or mechanism:",
                placeholder="e.g., 'kidney damage'"
            )
            drug_filter = st.text_input(
                "Filter by a specific drug ID (optional):",
                placeholder="e.g., '150' (case-sensitive)"
            )
            n_results = st.slider("Number of results:", min_value=1, max_value=20, value=5)

        # --- Search Logic ---
        if query_text:
            query_texts = [query_text]
        else:
            query_texts = ["A dangerous drug interaction"]

        query_args = {
            "query_texts": query_texts,
            "n_results": n_results,
            "include": ["documents", "metadatas"]
        }

        if drug_filter:
            where_filter = {
                "$or": [
                    {"drug_a": {"$contains": drug_filter.strip()}},
                    {"drug_b": {"$contains": drug_filter.strip()}}
                ]
            }
            query_args["where"] = where_filter
            
        try:
            results = collection.query(**query_args)
            
            st.divider()
            if not results['documents'][0]:
                st.warning("No results found. Try a broader search term or check your filters.")
            else:
                st.subheader(f"Found {len(results['documents'][0])} matching interactions:")
                
                for i in range(len(results['documents'][0])):
                    meta = results['metadatas'][0][i]
                    explanation = results['documents'][0][i]
                    
                    drug_a = meta.get('drug_a', 'N/A')
                    drug_b = meta.get('drug_b', 'N/A')
                    score = meta.get('risk_score', 0.0)

                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### {drug_a} + {drug_b}")
                        with col2:
                            # --- UPGRADE: Color-coded risk ---
                            if score >= 0.95:
                                st.metric(label="GNN Risk Score", value=f"{score:.2f}", delta="High Risk", delta_color="inverse")
                            elif score >= 0.8:
                                st.metric(label="GNN Risk Score", value=f"{score:.2f}", delta="Medium Risk", delta_color="normal")
                            else:
                                st.metric(label="GNN Risk Score", value=f"{score:.2f}", delta="Low Risk", delta_color="off")

                        st.info(f"**Clinical Explanation:** {explanation}")

        except Exception as e:
            st.error(f"An error occurred during the query: {e}")

    # --- Tab 2: Browse All Data ---
    with tab2:
        st.header("Browse All 200 Predicted Interactions")
        st.markdown("You can search, filter, and sort the entire dataset.")
        
        # Display the full dataframe in an interactive table
        st.dataframe(
            df_explanations,
            use_container_width=True,
            height=600,
            column_config={
                "risk_score": st.column_config.ProgressColumn(
                    "Risk Score",
                    format="%.3f",
                    min_val=0.0,
                    max_val=1.0,
                ),
            }
        )

    # --- Tab 3: Interaction Graph (REMOVED) ---
    # The entire graph tab has been deleted to fix the error.