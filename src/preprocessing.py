import os
import re
from typing import List

def load_documents(directory_path: str) -> str:
    """Reads all text files in a directory and combines them."""
    combined_text = ""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"📁 Created folder: '{directory_path}'. Place your raw context files here!")
        return combined_text

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as f:
                combined_text += f.read() + "\n"
    return combined_text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Splits raw text into overlapping chunks based on word counts."""
    # Clean up excessive white space
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    words = cleaned_text.split(" ")
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(words):
        end_idx = start_idx + chunk_size
        chunk = " ".join(words[start_idx:end_idx])
        chunks.append(chunk)
        # Shift forward by chunk_size minus the overlap
        start_idx += (chunk_size - overlap)
        
    return chunks

if __name__ == "__main__":
    # Test our parsing system
    sample_dir = "./data"
    print("⏳ Testing document processing framework...")
    raw_content = load_documents(sample_dir)
    
    if raw_content:
        processed_chunks = chunk_text(raw_content)
        print(f"✅ Successfully broken into {len(processed_chunks)} clean text chunks!")
    else:
        print("💡 The data directory is currently empty. Drop a text file inside './data' to test it.")