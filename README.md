# SemanticMatchCalculator

A semantic and keyword-based text similarity calculator with support for FAISS and Qdrant vector search backends. Includes a FastAPI backend and a simple frontend.

## Features
- Semantic similarity using sentence-transformers
- Keyword similarity (BM25, token overlap)
- Hybrid scoring
- Supports FAISS (in-memory) and Qdrant (vector DB) backends

## Setup Instructions

### 1. Clone the Repository
```
git clone <repo-url>
cd SemanticMatchCalculator
```

### 2. Python Environment
Create and activate a virtual environment:
```
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```
pip install -r SemanticMatchCalculator/requirements.txt
```

### 4. Qdrant (Optional, for Qdrant backend)
Run Qdrant using Docker:
```
docker run -p 6333:6333 qdrant/qdrant
```
Access the Qdrant dashboard at http://localhost:6333/dashboard

### 5. Run the Backend
From the `SemanticMatchCalculator/SemanticMatchCalculator` directory (where `backend/` is located):
```
# Activate venv if not already
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## API Usage

### Endpoint
`POST /similarity`

#### Request JSON
```
{
  "text_a": "string",
  "text_b": "string",
  "search_mode": "semantic" | "keyword" | "hybrid",
  "embedding_model": "all-MiniLM-L6-v2",  // optional
  "vector_db": "FAISS" | "Qdrant"         // optional
}
```

#### Response JSON
```
{
  "score": float,
  "debug": { ... }
}
```

## Troubleshooting

### Qdrant Errors
- Ensure Docker is running and Qdrant is started before using the Qdrant backend.
- If you see `ValueError: Unsupported points selector type`, update your code to use `points_selector=[id1, id2]` (list of IDs).

### Vector Dimension Errors
- Qdrant collections are created with a fixed vector dimension (e.g., 384 for MiniLM). If you change the embedding model, you may need to delete and recreate the collection.
- Error example: `Vector dimension error: expected dim: 384, got 768`

### Backend Not Starting
- Make sure you are in the correct directory and the virtual environment is activated.
- Install `uvicorn` if not present: `pip install uvicorn`

## Logging
- The backend prints log statements for all major operations (requests, similarity calculations, FAISS, Qdrant actions) to the console for debugging.

