import logging
from sentence_transformers import SentenceTransformer
from config import RAG_CONFIG

logger = logging.getLogger(__name__)
_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading: {RAG_CONFIG['embedding_model']}")
        _model = SentenceTransformer(RAG_CONFIG["embedding_model"])
        logger.info("Embedding model loaded.")
    return _model


def embed_texts(texts: list) -> list:
    model = get_embedding_model()
    return model.encode(
        texts, normalize_embeddings=True
    ).tolist()


def embed_query(query: str) -> list:
    model = get_embedding_model()
    return model.encode(
        query, normalize_embeddings=True
    ).tolist()