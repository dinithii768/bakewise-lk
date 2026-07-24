"""
Research Agent for BakeWise LK

PATTERN: Worker (in Orchestrator-Worker pattern)
ROLE: Retrieves relevant information from RAG knowledge base
MODEL: llama-3.1-8b-instant via Groq
WHY: Fast retrieval summarization does not need heavy reasoning
"""

import logging
from groq import Groq
from agents.base_agent import BaseAgent
from agents.message import AgentMessage
from config import GROQ_API_KEY, MODELS

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Worker agent that retrieves and summarizes
    relevant knowledge base content.
    """

    def __init__(self, retriever):
        super().__init__("ResearchAgent")
        self.retriever = retriever
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = MODELS["router"]["model_id"]

    def handle_task(self, message: AgentMessage) -> AgentMessage:
        query = message.content.get("query", "")
        intent = message.content.get("intent", "general")

        # RAG Retrieval
        docs, context = self.retriever.retrieve_and_format(query, top_k=4)

        if not docs:
            return self._reply(message, {
                "research": "No relevant information found in knowledge base.",
                "sources": [],
                "context": ""
            })

        # Summarize retrieved context
        prompt = f"""You are a research assistant for Sri Lankan home food businesses.

User Question: {query}
Intent Category: {intent}

Retrieved Knowledge Base Content:
{context}

Summarize the most relevant information from the knowledge base
to help answer the user question.
Be concise, factual, and specific to Sri Lanka.
Only use information from the provided content."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=MODELS["router"]["temperature"],
            max_tokens=MODELS["router"]["max_tokens"]
        )

        summary = response.choices[0].message.content

        sources = list(set([
            d.metadata.get("source", "unknown") for d in docs
        ]))

        self.logger.info(
            f"[ResearchAgent] Retrieved {len(docs)} chunks "
            f"from {len(sources)} sources"
        )

        return self._reply(message, {
            "research": summary,
            "sources": sources,
            "context": context,
            "chunk_count": len(docs)
        })