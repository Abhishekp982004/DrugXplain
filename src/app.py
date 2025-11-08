import streamlit as st
import chromadb
import os
import pandas as pd
from streamlit_option_menu import option_menu
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import regex as re

# --- Configuration ---
DB_PATH = "./chroma_db"
COLLECTION_NAME = "drug_interactions"

# --- Page Setup ---
st.set_page_config(
    page_title="Drug Interaction Explainer",
    page_icon="",
    layout="wide"
)

# --- Background and Styling ---
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         [data-testid="stAppViewContainer"] {{
             background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
                             url("https://images.pexels.com/photos/159211/headache-pain-pills-medication-159211.jpeg?cs=srgb&dl=pexels-pixabay-159211.jpg&fm=jpg");
             background-size: cover;
             background-position: center center;
             background-repeat: no-repeat;
             background-attachment: fixed;
         }}
         
         [data-testid="stAppViewContainer"] > .main {{
             background-color: transparent;
         }}
         
         [data-testid="stSidebar"] > div:first-child {{
             background: rgba(20, 20, 20, 0.7);
             backdrop-filter: blur(10px);
             border-radius: 10px;
             margin: 1rem;
         }}
         
         div[data-testid*="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid*="stVerticalBlock"] {{
             background: rgba(40, 40, 40, 0.7);
             backdrop-filter: blur(10px);
             border-radius: 10px;
             padding: 20px;
             margin-bottom: 10px;
         }}

         div[data-testid="stExpander"] {{
             background: rgba(40, 40, 40, 0.7);
             backdrop-filter: blur(10px);
             border-radius: 10px;
         }}
         
         [data-testid="stHeader"] {{
             background-color: rgba(0,0,0,0);
         }}
         [data-testid="stToolbar"] {{
             right: 2rem;
         }}
         
         iframe {{
             background-color: transparent !important;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# --- Caching Functions ---
@st.cache_resource
def get_db_collection():
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
def load_all_data_from_db(_collection):
    try:
        data = _collection.get(include=["metadatas"])
        df = pd.DataFrame(data['metadatas'])
        df['risk_score'] = pd.to_numeric(df['risk_score'])
        df['drug_a_name'] = df['drug_a_name'].astype(str)
        df['drug_b_name'] = df['drug_b_name'].astype(str)
        return df
    except Exception as e:
        st.error(f"Error loading full dataset from Chroma: {e}")
        return pd.DataFrame()

# --- Load Data ---
collection = get_db_collection()
if not collection:
    st.error("A required data source is missing. Please run 'vector_store.py' first.")
    st.stop()
df_all_data = load_all_data_from_db(collection)

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("Core Workflow")
    st.markdown("""
    1.  *Graph Construction:* Building the drug-drug interaction network.
    2.  *GNN Risk Prediction:* Identifying novel, high-risk interactions.
    3.  *LLM Clinical Explanation:* Generating human-readable explanations.
    """)
    st.divider()

    selected_page = option_menu(
        menu_title="Main",
        options=["Home", "Search", "Browse All", "Graph"],
        icons=["house-fill", "search", "table", "share-fill"],
        menu_icon="cast",
        default_index=0,
    )

# --- Page Title ---
st.title("Adverse Drug Interaction Explainer")

# --- Page 1: Home ---
if selected_page == "Home":
    st.header("Project Overview")
    st.markdown("""
    This dashboard is the final step in an AI pipeline designed to predict and explain adverse Drug-Drug Interactions (DDIs).
    
    Adverse DDIs are a major cause of medical complications. Our project addresses this by:
    1.  *GNN Prediction* A Graph Neural Network (GNN) was trained on known drug data to predict novel, high-risk interactions.
    2.  *LLM Explanation* The top 200 high-risk pairs were fed into a local LLM to generate clinical, human-readable explanations.
    3.  *Vector Database* These 200 explanations are stored in a vector database, allowing you to search for them by symptom, mechanism, or drug name.
    
    Use the sidebar to navigate between the Search, Browse, and Graph views.
    """)

# --- Page 2: Search ---
elif selected_page == "Search":
    st.header("Search by Symptom or Drug Name")
    
    with st.sidebar:
        st.header("Search Filters")
        query_text = st.text_input(
            "Search by symptom or mechanism:",
            placeholder="e.g., 'kidney damage' or 'bleeding risk'"
        )
        n_results = st.slider("Number of results:", min_value=1, max_value=20, value=5, key="slider_tab1")

    if not query_text:
        st.info("Enter a symptom, mechanism, or drug name in the search bar to begin.")
        st.stop()

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
    except Exception as e:
        if "No document found" in str(e) or "does not contain" in str(e):
            st.warning("No results found for your query. Try different keywords.")
            st.stop()
        else:
            st.error(f"An error occurred during the query: {e}")
            st.stop()
    
    if not results or not results['ids'][0]:
        st.warning("No results found for your query. Try different keywords.")
        st.stop()

    st.subheader(f"Found {len(results['ids'][0])} matching interactions:")
    
    for i, data in enumerate(results['metadatas'][0]):
        explanation = results['documents'][0][i]
        drug_a = data.get('drug_a_name', "N/A")
        drug_b = data.get('drug_b_name', "N/A")
        score = data.get('risk_score', 0)
        
        expander_title = f"{drug_a} + {drug_b}** (Risk Score: {score:.4f})"
        
        with st.expander(expander_title):
            st.metric(
                label="GNN Risk Score",
                value=f"{score:.4f}",
                delta="High Risk" if score > 0.9 else ("Moderate Risk" if score > 0.7 else "Low Risk"),
                delta_color="inverse" if score > 0.9 else "normal"
            )
            st.markdown("*Clinical Explanation:*")
            st.markdown(explanation)

# --- Page 3: Browse All ---
elif selected_page == "Browse All":
    st.header("Browse All 200 Predicted Interactions")
    st.markdown("You can search, filter, and sort the entire dataset.")
    
    if not df_all_data.empty:
        display_columns = [
            "drug_a_name",
            "drug_b_name",
            "risk_score",
            "drug_a_id",
            "drug_b_id",
            "explanation"
        ]
        available_columns = [col for col in display_columns if col in df_all_data.columns]
        
        st.dataframe(
            df_all_data[available_columns],
            use_container_width=True,
            column_config={
                "risk_score": st.column_config.ProgressColumn(
                    "GNN Risk Score",
                    help="The risk score (0-1) predicted by the GNN.",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "drug_a_name": "Drug A",
                "drug_b_name": "Drug B",
                "explanation": st.column_config.TextColumn(
                    "LLM Explanation", width="large"
                ),
                "drug_a_id": "Drug A ID",
                "drug_b_id": "Drug B ID",
            },
            height=600
        )
    else:
        st.error("Data could not be loaded for the table.")

# --- Page 4: Graph ---
elif selected_page == "Graph":
    st.header("Drug Interaction Network Graph")
    
    with st.sidebar:
        st.header("Graph Filters")
        min_risk_for_graph = st.slider(
            "Minimum Risk Score to Display:",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.01
        )

    def sanitize_for_html(text):
        if text is None:
            return "Unknown"
        text = str(text).replace('α', 'alpha').replace('β', 'beta').replace('γ', 'gamma')
        text = text.replace('ö', 'o').replace('é', 'e').replace('ü', 'u')
        return text.encode('ascii', 'ignore').decode('ascii').strip()

    df_graph = df_all_data[df_all_data['risk_score'] >= min_risk_for_graph]

    if df_graph.empty:
        st.warning(f"No interactions found with a risk score of {min_risk_for_graph} or higher. Try lowering the slider.")
    else:
        st.markdown(f"Displaying {len(df_graph)} high-risk interactions.")

        net = Network(
            height="850px",
            width="100%",
            bgcolor="#1e1e1e",
            font_color="#ffffff",
            notebook=True,
            cdn_resources='in_line'
        )

        net.repulsion(
            node_distance=220,
            central_gravity=0.02,
            spring_length=160,
            spring_strength=0.05,
            damping=0.09
        )

        net.set_options("""
        const options = {
          "nodes": {
            "font": {"color": "#ffffff", "size": 18},
            "color": {"background": "#0077b6", "border": "#00b4d8"},
            "shape": "dot",
            "scaling": {"min": 10, "max": 30}
          },
          "edges": {
            "color": {"color": "#e63946"},
            "smooth": false
          },
          "physics": {
            "repulsion": {
              "nodeDistance": 220,
              "springLength": 160
            }
          },
          "layout": {"improvedLayout": true}
        }
        """)

        unique_drugs = set(df_graph['drug_a_name']).union(set(df_graph['drug_b_name']))
        node_map = {}
        for drug in unique_drugs:
            if drug and pd.notna(drug):
                clean_id = sanitize_for_html(drug)
                if not clean_id:
                    clean_id = f"drug_{hash(drug)}"
                label = sanitize_for_html(drug)
                node_map[drug] = clean_id
                net.add_node(clean_id, label=label, title=label)

        for _, row in df_graph.iterrows():
            a, b = row['drug_a_name'], row['drug_b_name']
            if not (a in node_map and b in node_map):
                continue
            score = row['risk_score']
            title = f"{sanitize_for_html(a)} + {sanitize_for_html(b)}<br>Risk: {score:.4f}"
            net.add_edge(
                node_map[a],
                node_map[b],
                title=title,
                width=(score * 5) + 1,
                color="#e63946"
            )

        import tempfile, io, os, time
        import streamlit.components.v1 as components

        try:
            tmp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
            html_content = net.generate_html()
            with io.open(tmp_file_path, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(html_content)
            with io.open(tmp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_data = f.read()
            components.html(html_data, height=900, scrolling=False)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generating graph (UTF-8 safe): {e}")
        finally:
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                time.sleep(1)
                try:
                    os.unlink(tmp_file_path)
                except PermissionError:
                    pass