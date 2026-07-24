"""
Advisor Agent for BakeWise LK

PATTERN: Worker (in Orchestrator-Worker pattern)
ROLE: Generates final detailed advice for the user
MODEL: llama-3.3-70b-versatile via Groq
WHY: Final advice needs strong reasoning and detailed response
     More expensive model justified for quality output
"""

import logging
from groq import Groq
from agents.base_agent import BaseAgent
from agents.message import AgentMessage
from config import GROQ_API_KEY, MODELS

logger = logging.getLogger(__name__)


class AdvisorAgent(BaseAgent):
    """
    Worker agent that generates final actionable advice
    for Sri Lankan home food business owners.
    """

    def __init__(self):
        super().__init__("AdvisorAgent")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = MODELS["advisor"]["model_id"]

    def handle_task(self, message: AgentMessage) -> AgentMessage:
        query = message.content.get("query", "")
        research = message.content.get("research", "")
        intent = message.content.get("intent", "general")
        plan = message.content.get("plan", [])

        plan_text = "\n".join(
            f"  {i+1}. {step}" for i, step in enumerate(plan)
        )

        system_prompt = """You are BakeWise LK, an expert AI advisor for
Sri Lankan home-based food businesses and home bakers.

You provide practical, accurate, and actionable advice about:
- Business registration and legal compliance in Sri Lanka
- Food safety and hygiene standards
- Food labeling requirements
- Pricing and costing strategies
- SME loans and financing options
- Online selling and social media marketing
- Packaging and storage best practices

Always reference Sri Lankan laws, institutions, and context.
Be warm, encouraging, and practical.
Format your response clearly with sections where helpful."""

        user_prompt = f"""User Question: {query}
Intent: {intent}

Execution Plan followed:
{plan_text}

Research Findings from Knowledge Base:
{research}

Please provide a comprehensive, practical response that directly
answers the question using the research findings above.
Include specific Sri Lankan institutions, regulations, costs in LKR,
and actionable next steps where relevant."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=MODELS["advisor"]["temperature"],
            max_tokens=MODELS["advisor"]["max_tokens"]
        )

        advice = response.choices[0].message.content

        self.logger.info(
            f"[AdvisorAgent] Generated advice using {self.model}"
        )

        return self._reply(message, {
            "advice": advice,
            "model_used": self.model
        })