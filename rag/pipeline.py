import logging
from rag.document_loader import DocumentLoader
from rag.vector_store import VectorStore
from rag.retriever import Retriever

logger = logging.getLogger(__name__)
_retriever = None


def get_retriever(force_reindex: bool = False) -> Retriever:
    global _retriever
    if _retriever is not None and not force_reindex:
        return _retriever

    logger.info("Initializing RAG pipeline...")
    loader = DocumentLoader()
    vs = VectorStore()
    stats = vs.get_stats()
    existing = stats.get("total_chunks", 0)

    if existing > 0 and not force_reindex:
        logger.info(f"Using existing: {existing} chunks")
    else:
        chunks = loader.load_and_chunk()
        if chunks:
            vs.index_documents(chunks)
        else:
            logger.warning("No chunks found!")

    _retriever = Retriever(vs)
    logger.info("RAG pipeline ready.")
    return _retriever


def get_pipeline_stats() -> dict:
    vs = VectorStore()
    return vs.get_stats()