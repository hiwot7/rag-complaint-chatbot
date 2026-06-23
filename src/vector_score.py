import os
import pandas as pd
from typing import List, Dict

# Using standard lightweight vector search dependencies
# Run: pip install sentence-transformers faiss-cpu
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

class VectorStoreManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_dir: str = "data/index"):
        self.index_dir = index_dir
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata: List[Dict] = []
        os.makedirs(self.index_dir, exist_ok=True)

    def build_and_index(self, csv_path: str, text_column: str = "rag_text", chunk_size: int = 500, overlap: int = 50):
        """Loads processed data, slices text into overlapping windows, and embeds them."""
        if not os.path.exists(csv_path):
            print(f"❌ Error: Processed data file missing at {csv_path}")
            return

        print("📖 Reading processed datasets...")
        df = pd.read_csv(csv_path)
        
        chunks = []
        chunk_metadata = []

        print("✂️ Splitting serialized text into uniform chunks...")
        for idx, row in df.iterrows():
            text = str(row[text_column])
            words = text.split(" ")
            
            # Slide a window across the words based on chunk size and overlap parameters
            start = 0
            while start < len(words):
                end = start + chunk_size
                chunk_str = " ".join(words[start:end])
                chunks.append(chunk_str)
                
                # Keep track of metadata to map search results back to original rows
                chunk_metadata.append({
                    "text": chunk_str,
                    "product": row.get("product", "Unknown"),
                    "issue": row.get("issue", "Unknown")
                })
                start += (chunk_size - overlap)

        print(f"🧬 Generating text embeddings for {len(chunks)} fragments...")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        embeddings_array = np.array(embeddings).astype('float32')

        # Initialize FAISS Index (L2 Distance metric)
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
        self.metadata = chunk_metadata

        # Save index parameters locally
        self.save_index()
        print("💾 Vector database index compiled and saved successfully!")

    def save_index(self):
        faiss.write_index(self.index, os.path.join(self.index_dir, "faiss.index"))
        with open(os.path.join(self.index_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load_index(self):
        """Loads local index configurations into memory for ultra-fast queries."""
        index_path = os.path.join(self.index_dir, "faiss.index")
        meta_path = os.path.join(self.index_dir, "metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        return False

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Performs real-time cosine/L2 semantic search over indexed documents."""
        if self.index is None and not self.load_index():
            print("⚠️ Vector store index is empty or not yet compiled.")
            return []

        query_vector = np.array([self.model.encode(query)]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

if __name__ == "__main__":
    # Test script verification loop
    vstore = VectorStoreManager()
    vstore.build_and_index("data/processed/filtered_complaints.csv")