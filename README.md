# 🧬 DrugXplain: Adverse Drug Interaction Prediction & Clinical Explanation using Graph Neural Networks and Medical LLMs

**🏆 Winner – Swasthya Avishkar Hackathon 2025** *Organized by Center for Data Modelling, Analytics and Visualization (CoDMAV), PES University and powered by Carelon Global Solutions India.* 
## **Track: AI Pipelines for Drug Repurposing, Efficacy Prediction, and Development**

---

## Overview
**DrugXplain** is an AI-powered healthcare decision-support system designed to **predict, analyze, and explain adverse drug–drug interactions (DDIs)** using Graph Neural Networks (GNNs) and Large Language Models (LLMs).

The system helps clinicians move beyond static, rule-based interaction checks by identifying **hidden, high-risk drug combinations** and generating **human-readable clinical explanations** for why those interactions occur.

## Features
* **Intelligent Search:** Search by drug names, symptoms, or biological mechanisms.
* **GNN-based Risk Scoring:** Predicts the probability of harmful interactions using graph structures.
* **LLM Explanations:** Translates complex AI predictions into clinician-friendly medical insights.
* **Interactive Network Graph:** Visualize drug interaction networks using PyVis.
* **Fast Retrieval:** Semantic search enabled by vector embeddings (ChromaDB).

## Technology Stack
* **Language:** Python
* **Deep Learning:** PyTorch, PyTorch Geometric (Graph Convolutional Networks)
* **Vector Database:** ChromaDB
* **LLM:** Local Medical LLM for clinical reasoning
* **Frontend:** Streamlit
* **Visualization:** PyVis

---

## System Architecture
1. **Data Processing:** Mapping drugs to molecular components and constructing interaction graphs.
2. **GNN Training:** Link prediction on the drug graph to generate interaction risk scores.
3. **Inference:** Predicting top high-risk drug pairs and exporting to CSV.
4. **LLM Explanation Layer:** Generating insights on mechanism, symptoms, and patient risk.
5. **Vector Database:** Storing metadata and explanations for semantic search.
6. **Dashboard:** Exploring interactions and visualizing networks via Streamlit.

---

## How to Run the Project

Follow these steps to set up and launch the application:

### 1. Install dependencies
```
pip install -r requirements.txt
```
### 2. Train the GNN model
```
cd src
python train_gnn.py
```
### 3. Run inference
```
python inference_gnn.py
```
### 4. Generate explanations
```
python llm_explainer.py
```
### 5. Store in vector DB
```
python vector_store.py
```
### 6. Launch the dashboard
```
streamlit run app.py
```
---
## Team DrugXplain
* **Abhishek P**
* **Aditya Dwaraki Rao**
* **Bhuvi Prashanth**

## Future Scope
* **Polypharmacy Prediction:** Extending support for multi-drug (3+) interactions.
* **EHR Integration:** Integration with hospital Electronic Health Record systems.
* **Real-time Support:** Deployment-ready backend APIs for clinical settings.
* **Regulatory Compliance:** Developing modules for safety and explainability certification.








