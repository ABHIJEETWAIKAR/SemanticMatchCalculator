from typing import Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import uuid

# Download NLTK data if not already present
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

try:
    _ = nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    _ = nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


def tokenize(text):
    # Simpler tokenization: lowercase and remove punctuation only
    tokens = text.lower().split()
    tokens = [t.strip(string.punctuation) for t in tokens if t.strip(string.punctuation)]
    return tokens


def keyword_similarity(text_a: str, text_b: str, bm25: bool = False, top_n: int = 5) -> Tuple[float, Dict[str, Any]]:
    print(f"[keyword_similarity] text_a: {text_a}, text_b: {text_b}, bm25: {bm25}, top_n: {top_n}")
    # Tokenize
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    # Also show raw tokens for debug
    raw_tokens_a = text_a.split()
    raw_tokens_b = text_b.split()
    debug = {}
    if bm25:
        # BM25
        corpus = [tokens_a, tokens_b]
        bm25_model = BM25Okapi(corpus)
        score_ab = bm25_model.get_score(tokens_b, 0)
        score_ba = bm25_model.get_score(tokens_a, 1)
        score = float(np.clip((score_ab + score_ba) / 2, 0, 100))
        # Top tokens by BM25
        top_tokens = bm25_model.get_top_n(tokens_b, tokens_a, n=top_n)
        debug.update({
            'bm25_score_ab': float(score_ab),
            'bm25_score_ba': float(score_ba),
            'top_bm25_tokens': top_tokens,
        })
    else:
        # TF-IDF
        vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=True)
        tfidf = vectorizer.fit_transform([text_a, text_b])
        cos_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        score = float(np.clip(cos_sim * 100, 0, 100))
        # Top tokens by TF-IDF
        feature_names = vectorizer.get_feature_names_out()
        top_a = sorted(zip(tfidf[0].toarray()[0], feature_names), reverse=True)[:top_n]
        top_b = sorted(zip(tfidf[1].toarray()[0], feature_names), reverse=True)[:top_n]
        debug.update({
            'cosine_similarity': float(cos_sim),
            'top_tfidf_tokens_a': [t for _, t in top_a],
            'top_tfidf_tokens_b': [t for _, t in top_b],
        })
    # Token overlap
    overlap = set(tokens_a) & set(tokens_b)
    debug.update({
        'token_overlap': list(overlap),
        'num_overlap': len(overlap),
        'tokens_a': tokens_a,
        'tokens_b': tokens_b,
        'raw_tokens_a': raw_tokens_a,
        'raw_tokens_b': raw_tokens_b,
        'bm25_used': bm25,
    })
    print(f"[keyword_similarity] score: {score}, debug: {debug}")
    return score, debug

# Qdrant helper
QDRANT_COLLECTION = "semantic_match"
def get_qdrant_client():
    # Try local Qdrant instance
    return QdrantClient(host="localhost", port=6333)

def ensure_qdrant_collection(client, dim):
    if QDRANT_COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

def qdrant_similarity(text_a, text_b, embedding_model):
    print(f"[qdrant_similarity] text_a: {text_a}, text_b: {text_b}, embedding_model: {embedding_model}")
    model = SentenceTransformer(embedding_model)
    emb_a = model.encode([text_a])[0]
    emb_b = model.encode([text_b])[0]
    client = get_qdrant_client()
    ensure_qdrant_collection(client, len(emb_a))
    # Upsert both as points
    id_a = str(uuid.uuid4())
    id_b = str(uuid.uuid4())
    client.upsert(collection_name=QDRANT_COLLECTION, points=[
        PointStruct(id=id_a, vector=emb_a, payload={"text": text_a}),
        PointStruct(id=id_b, vector=emb_b, payload={"text": text_b}),
    ])
    # Query similarity (search b in a)
    search_result = client.search(collection_name=QDRANT_COLLECTION, query_vector=emb_a, limit=2)
    # Find the other point's score
    sim_score = 0.0
    for hit in search_result:
        if hit.id == id_b:
            sim_score = hit.score
    # Clean up (optional: delete points)
    client.delete(collection_name=QDRANT_COLLECTION, points_selector=[id_a, id_b])
    print(f"[qdrant_similarity] sim_score: {sim_score}")
    return sim_score, {'qdrant_score': sim_score, 'ids': [id_a, id_b]}

def semantic_similarity(text_a: str, text_b: str, embedding_model: str = 'all-MiniLM-L6-v2', top_n: int = 5, vector_db: str = 'FAISS') -> Tuple[float, Dict[str, Any]]:
    print(f"[semantic_similarity] text_a: {text_a}, text_b: {text_b}, embedding_model: {embedding_model}, top_n: {top_n}, vector_db: {vector_db}")
    if vector_db == 'Qdrant':
        sim_score, qdrant_debug = qdrant_similarity(text_a, text_b, embedding_model)
        score = float(np.clip(sim_score * 100, 0, 100))
        debug = {'vector_db': 'Qdrant', **qdrant_debug}
        print(f"[semantic_similarity] score: {score}, debug: {debug}")
        return score, debug
    # Handle empty or very short input
    if not text_a or not text_b or len(text_a.strip()) < 3 or len(text_b.strip()) < 3:
        print(f"[semantic_similarity] Returning 0.0 due to empty or short input.")
        return 0.0, {'error': 'One or both texts are empty or too short.'}
    # Load embedding model (cache for performance)
    if not hasattr(semantic_similarity, '_model') or semantic_similarity._model_name != embedding_model:
        semantic_similarity._model = SentenceTransformer(embedding_model)
        semantic_similarity._model_name = embedding_model
    model = semantic_similarity._model
    # Compute embeddings
    emb_a = model.encode([text_a])[0]
    emb_b = model.encode([text_b])[0]
    if vector_db == 'FAISS':
        from backend.faiss_utils import create_faiss_index, search_faiss_index
        # For pairwise similarity, create index with emb_b, search with emb_a
        index = create_faiss_index(np.expand_dims(emb_b, axis=0))
        D, I = search_faiss_index(index, np.expand_dims(emb_a, axis=0), top_k=1)
        # FAISS returns squared L2 distance; convert to similarity (here, negative distance for demonstration)
        faiss_score = float(-D[0][0])
        score = float(np.clip((1 / (1 + D[0][0])) * 100, 0, 100))
        debug = {
            'vector_db': 'FAISS',
            'faiss_distance': float(D[0][0]),
            'faiss_index': int(I[0][0]),
            'embedding_model': embedding_model,
            'embedding_a_shape': str(np.shape(emb_a)),
            'embedding_b_shape': str(np.shape(emb_b)),
            'embedding_a_norm': float(np.linalg.norm(emb_a)),
            'embedding_b_norm': float(np.linalg.norm(emb_b)),
        }
        print(f"[semantic_similarity] score: {score}, debug: {debug}")
        return score, debug
    # Cosine similarity fallback
    cos_sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
    score = float(np.clip(cos_sim * 100, 0, 100))
    tokens_a = text_a.split()
    tokens_b = text_b.split()
    overlap = set(tokens_a) & set(tokens_b)
    top_overlap = list(overlap)[:top_n]
    debug = {
        'embedding_model': embedding_model,
        'cosine_similarity': cos_sim,
        'embedding_a_shape': str(np.shape(emb_a)),
        'embedding_b_shape': str(np.shape(emb_b)),
        'embedding_a_norm': float(np.linalg.norm(emb_a)),
        'embedding_b_norm': float(np.linalg.norm(emb_b)),
        'top_overlap_tokens': top_overlap,
        'num_overlap_tokens': len(overlap),
        'tokens_a': tokens_a,
        'tokens_b': tokens_b,
    }
    print(f"[semantic_similarity] score: {score}, debug: {debug}")
    return score, debug

def hybrid_similarity(text_a: str, text_b: str, embedding_model: str = 'all-MiniLM-L6-v2', weight_semantic: float = 0.5, vector_db: str = 'FAISS') -> Tuple[float, Dict[str, Any]]:
    print(f"[hybrid_similarity] text_a: {text_a}, text_b: {text_b}, embedding_model: {embedding_model}, weight_semantic: {weight_semantic}, vector_db: {vector_db}")
    score_keyword, debug_keyword = keyword_similarity(text_a, text_b)
    score_semantic, debug_semantic = semantic_similarity(text_a, text_b, embedding_model, vector_db=vector_db)
    score = float(weight_semantic * score_semantic + (1 - weight_semantic) * score_keyword)
    debug = {
        'score_keyword': score_keyword,
        'score_semantic': score_semantic,
        'weight_semantic': weight_semantic,
        'debug_keyword': debug_keyword,
        'debug_semantic': debug_semantic,
        'vector_db': vector_db,
    }
    print(f"[hybrid_similarity] score: {score}, debug: {debug}")
    return score, debug 