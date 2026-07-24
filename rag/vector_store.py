import logging
from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from config import RAG_CONFIG, CHROMA_DB_PATH
from rag.embeddings import embed_texts, embed_query

logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(self):
        Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = RAG_CONFIG["collection_name"]
        self.collection = None

    def _get_or_create_collection(self):
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        return self.collection

    def index_documents(self, chunks: List[Document]) -> int:
        collection = self._get_or_create_collection()
        existing = collection.count()
        if existing > 0:
            logger.info(f"Already indexed: {existing} chunks.")
            return existing
        logger.info(f"Indexing {len(chunks)} chunks...")
        texts = [c.page_content for c in chunks]
        embeddings = embed_texts(texts)
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [c.metadata for c in chunks]
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            collection.add(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                documents=texts[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )
        total = collection.count()
        logger.info(f"Done. Total: {total}")
        return total

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Document]:
        collection = self._get_or_create_collection()
        if collection.count() == 0:
            return []
        k = top_k or RAG_CONFIG["top_k"]
        query_emb = embed_query(query)
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        docs = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            score = round(1 - dist, 4)
            meta["retrieval_score"] = score
            docs.append(
                Document(page_content=text, metadata=meta)
            )
        return docs

    def get_stats(self) -> dict:
        try:
            col = self._get_or_create_collection()
            return {
                "total_chunks": col.count(),
                "collection_name": self.collection_name,
                "embedding_model": RAG_CONFIG["embedding_model"]
            }
        except Exception as e:
            return {"error": str(e)}

    def reset(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = None