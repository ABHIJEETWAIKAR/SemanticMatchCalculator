from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any
from backend.similarity import keyword_similarity, semantic_similarity, hybrid_similarity

app = FastAPI()

class SimilarityRequest(BaseModel):
    text_a: str
    text_b: str
    search_mode: Literal['keyword', 'semantic', 'hybrid'] = 'semantic'
    embedding_model: Optional[str] = 'all-MiniLM-L6-v2'
    vector_db: Optional[str] = 'FAISS'

class SimilarityResponse(BaseModel):
    score: float
    debug: Dict[str, Any]

@app.post('/similarity', response_model=SimilarityResponse)
def similarity_endpoint(req: SimilarityRequest):
    print(f"Received request: {req}")
    if req.search_mode == 'keyword':
        score, debug = keyword_similarity(req.text_a, req.text_b)
    elif req.search_mode == 'semantic':
        score, debug = semantic_similarity(req.text_a, req.text_b, req.embedding_model, vector_db=req.vector_db)
    elif req.search_mode == 'hybrid':
        score, debug = hybrid_similarity(req.text_a, req.text_b, req.embedding_model, vector_db=req.vector_db)
    else:
        score, debug = 0.0, {"error": "Invalid search mode"}
    debug['vector_db'] = req.vector_db
    print(f"Returning response: score={score}, debug={debug}")
    return SimilarityResponse(score=score, debug=debug) 