import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

MODELS = {
    "router": {
        "model_id": "llama-3.1-8b-instant",
        "purpose": "routing, planning, cheap classification"
    },
    "advisor": {
        "model_id": "llama-3.3-70b-versatile",
        "purpose": "deep reasoning, final advice, reflection"
    }
}

RAG_CONFIG = {
    "chunk_size": 700,
    "chunk_overlap": 120,
    "top_k": 4,
    "collection_name": "bakewise_lk_kb",
    "embedding_model": "all-MiniLM-L6-v2"
}