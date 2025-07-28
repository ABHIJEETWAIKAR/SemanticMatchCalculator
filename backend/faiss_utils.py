import faiss
import numpy as np
from typing import List

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    print(f"[faiss_utils] Creating FAISS index with embeddings shape: {embeddings.shape}")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print(f"[faiss_utils] Index created and embeddings added. Index ntotal: {index.ntotal}")
    return index

def search_faiss_index(index: faiss.IndexFlatL2, query_embedding: np.ndarray, top_k: int = 1):
    print(f"[faiss_utils] Searching FAISS index with query_embedding shape: {query_embedding.shape}, top_k: {top_k}")
    D, I = index.search(query_embedding, top_k)
    print(f"[faiss_utils] Search results - Distances: {D}, Indices: {I}")
    return D, I 