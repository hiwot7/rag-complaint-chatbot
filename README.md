# Enterprise RAG Complaint Management Chatbot
10 Academy Week 7 Capstone Project Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) system engineered to clean, serialize, index, and query unstructured enterprise customer grievance records accurately without model hallucinations.

## 🚀 Key System Features
- **Task 1 (Data Preprocessing):** Automates cleaning, filters data anomalies, and transforms disjointed multi-column records into single unified natural language context strings (`src/eda_preprocessing.py`).
- **Task 2 (Vector Indexing):** Uses a local `all-MiniLM-L6-v2` transformer model to create localized vector spaces, executing real-time semantic document search using Facebook AI Similarity Search (`FAISS`).
- **Interactive UI Dashboard:** A functional user application featuring custom retrieval controls and full conversation session state tracking.

## 📁 Repository Structure
```text
├── data/
│   ├── raw/                 # Incoming raw enterprise datasets
│   └── processed/           # Serialized, clean context targets
├── notebooks/               # Sandbox validation and engineering logs
├── src/                     # Core system production execution layer
│   ├── eda_preprocessing.py # Pipeline ingestion stage
│   ├── vector_store.py      # Embedding generation and retrieval engine
│   └── app.py               # Streamlit application portal
└── requirements.txt         # Package configuration manifest