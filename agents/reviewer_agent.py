"""
Reviewer Agent for BakeWise LK

PATTERN: Reflection / Self-Critique
ROLE: Reviews and improves the advisor output for quality
MODEL: llama-3.3-70b-versatile via Groq
WHY: Quality critique needs strong reasoning capability
"""

import logging
import json
from groq import Groq
from agents.base_agent import BaseAgent
from agents.message import AgentMessage
from config import GROQ_API_KEY, MODELS

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    """
    REFLECTION PATTERN agent.
    Critiques and improves the AdvisorAgent output.
    Acts as quality gate before response reaches user.
    """

    def __init__(self):
        super().__init__("ReviewerAgent")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = MODELS["reviewer"]["model_id"]

    def handle_task(self, message: AgentMessage) -> AgentMessage:
        return self.handle_reflection(message)

    def handle_reflection(self, message: AgentMessage) -> AgentMessage:
        query = message.content.get("query", "")
        draft = message.content.get("draft", "")
        research = message.content.get("research", "")

        prompt = f"""You are a quality reviewer for AI-generated advice
about Sri Lankan home food businesses.

Original Question: {query}

Research Used:
{research[:800]}

Draft Response to Review:
{draft}

Review the draft on these criteria and respond with JSON only:
{{
  "accuracy_score": 1-5,
  "completeness_score": 1-5,
  "actionability_score": 1-5,
  "sri_lanka_relevance_score": 1-5,
  "overall": "good" or "needs_improvement",
  "issues": "brief description of any issues",
  "improved_response": "improved version if needed, else null"
}}

Score meanings:
5 = Excellent
4 = Good
3 = Acceptable
2 = Needs work
1 = Poor

Only provide improved_response if overall is needs_improvement."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=MODELS["reviewer"]["temperature"],
                max_tokens=MODELS["reviewer"]["max_tokens"]
            )

            raw = response.choices[0].message.content.strip()

            # Clean JSON if wrapped in markdown
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            review = json.loads(raw)

            overall = review.get("overall", "good")
            improved = review.get("improved_response")

            final_response = (
                improved
                if overall == "needs_improvement"
                and improved
                and improved != "null"
                else draft
            )

            final_response += (
                "\n\n---\n"
                "*This advice is for general guidance only. "
                "Always verify with relevant Sri Lankan authorities "
                "before making business decisions.*"
            )

            self.logger.info(
                f"[ReviewerAgent] Review complete. "
                f"Overall: {overall}"
            )

            return self._reply(message, {
                "final_response": final_response,
                "review_scores": {
                    "accuracy": review.get("accuracy_score"),
                    "completeness": review.get("completeness_score"),
                    "actionability": review.get("actionability_score"),
                    "sri_lanka_relevance": review.get(
                        "sri_lanka_relevance_score"
                    )
                },
                "overall_quality": overall,
                "issues": review.get("issues", "none"),
                "was_improved": (
                    overall == "needs_improvement" and bool(improved)
                )
            })

        except Exception as e:
            self.logger.error(f"[ReviewerAgent] Review failed: {e}")
            final_response = (
                draft +
                "\n\n---\n"
                "*Always verify with relevant Sri Lankan authorities.*"
            )
            return self._reply(message, {
                "final_response": final_response,
                "overall_quality": "unknown",
                "issues": str(e),
                "was_improved": False
            })