from typing import List, Tuple
from langchain.schema import Document
from rag.vector_store import VectorStore


class Retriever:

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    def retrieve(
        self, query: str, top_k: int = 4
    ) -> List[Document]:
        return self.vs.retrieve(query, top_k=top_k)

    def format_context(self, docs: List[Document]) -> str:
        if not docs:
            return "No relevant information found."
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            score = doc.metadata.get("retrieval_score", "N/A")
            parts.append(
                f"[{i}] Source: {source} | Relevance: {score}\n"
                f"{doc.page_content.strip()}"
            )
        return "\n\n---\n\n".join(parts)

    def retrieve_and_format(
        self, query: str, top_k: int = 4
    ) -> Tuple[List[Document], str]:
        docs = self.retrieve(query, top_k)
        context = self.format_context(docs)
        return docs, context