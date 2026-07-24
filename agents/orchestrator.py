"""
Orchestrator Agent for BakeWise LK

PATTERNS IMPLEMENTED:
1. Orchestrator-Worker: coordinates ResearchAgent + AdvisorAgent + ReviewerAgent
2. ReAct: Reason -> Act -> Observe loop
3. Planning/Task-Decomposition: breaks query into ordered steps
4. Router: uses router_tool to classify intent

This is the central coordinator of the BakeWise LK agent system.
"""

import logging
import json
from typing import Dict, Any, List
from groq import Groq

from agents.message import AgentMessage
from agents.research_agent import ResearchAgent
from agents.advisor_agent import AdvisorAgent
from agents.reviewer_agent import ReviewerAgent
from tools.router_tool import classify_intent
from tools.calculator_tool import (
    calculate_product_price,
    calculate_batch_profit,
    calculate_epf_etf,
    calculate_vat_status
)
from config import GROQ_API_KEY, MODELS, AGENT_CONFIG

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Central coordinator agent.

    Implements:
    - Router pattern via classify_intent()
    - Planning pattern via create_plan()
    - ReAct pattern via react_loop()
    - Orchestrator-Worker via send_message() to workers
    """

    def __init__(self, retriever):
        self.name = "OrchestratorAgent"
        self.retriever = retriever
        self.client = Groq(api_key=GROQ_API_KEY)

        # Initialize worker agents
        self.research_agent = ResearchAgent(retriever)
        self.advisor_agent = AdvisorAgent()
        self.reviewer_agent = ReviewerAgent()

        # Message history for A2A tracking
        self.message_log: List[Dict] = []

    def send_message(
        self, message: AgentMessage
    ) -> AgentMessage:
        """
        Send structured A2A message to worker agent.
        Records all messages for transparency.
        """
        self.message_log.append(message.to_dict())
        logger.info(
            f"[Orchestrator] -> [{message.receiver}]: "
            f"{message.message_type}"
        )

        if message.receiver == "ResearchAgent":
            response = self.research_agent.receive_message(message)
        elif message.receiver == "AdvisorAgent":
            response = self.advisor_agent.receive_message(message)
        elif message.receiver == "ReviewerAgent":
            response = self.reviewer_agent.receive_message(message)
        else:
            response = AgentMessage(
                sender=message.receiver,
                receiver=self.name,
                message_type="error",
                content={"error": f"Unknown agent: {message.receiver}"}
            )

        self.message_log.append(response.to_dict())
        return response

    def create_plan(
        self, query: str, intent: str, key_topics: List[str]
    ) -> List[str]:
        """
        PLANNING PATTERN: Decompose query into ordered steps.
        Uses fast model - planning does not need heavy reasoning.
        """
        prompt = f"""Create a step-by-step plan to answer this
Sri Lankan home food business question.

Question: {query}
Intent: {intent}
Key Topics: {', '.join(key_topics)}

Provide 3 to 5 clear steps.
Respond with JSON only:
{{"plan": ["step 1", "step 2", "step 3"]}}"""

        try:
            response = self.client.chat.completions.create(
                model=MODELS["router"]["model_id"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            result = json.loads(raw)
            return result.get("plan", [
                "Understand the question",
                "Research knowledge base",
                "Generate advice",
                "Review and improve"
            ])
        except Exception:
            return [
                "Classify query intent",
                "Retrieve relevant knowledge",
                "Generate actionable advice",
                "Review for quality"
            ]

    def react_loop(
        self, query: str, intent: str
    ) -> Dict[str, Any]:
        """
        ReAct PATTERN: Reason -> Act -> Observe -> Repeat

        Agent reasons about what tool to use,
        acts by calling the tool,
        observes the result,
        repeats until ready to answer.
        """
        tool_results = []
        max_iter = AGENT_CONFIG["max_react_iterations"]

        for iteration in range(max_iter):
            logger.info(f"[ReAct] Iteration {iteration + 1}")

            # REASON: decide what to do next
            tool_results_text = (
                json.dumps(tool_results[-1]) if tool_results else "none"
            )

            reason_prompt = f"""You are deciding what action to take next
to answer this home food business question.

Question: {query}
Intent: {intent}
Last tool result: {tool_results_text}
Iteration: {iteration + 1} of {max_iter}

Available actions:
- CALCULATE_PRICE: if question involves product pricing
- CALCULATE_PROFIT: if question involves profit or revenue
- CALCULATE_EPF: if question involves employee salary or EPF/ETF
- CALCULATE_VAT: if question involves VAT or tax threshold
- ANSWER: ready to generate final answer

Respond with JSON only:
{{"action": "ACTION_NAME", "reason": "why"}}"""

            try:
                reason_resp = self.client.chat.completions.create(
                    model=MODELS["router"]["model_id"],
                    messages=[{"role": "user", "content": reason_prompt}],
                    temperature=0.1,
                    max_tokens=150
                )
                raw = reason_resp.choices[0].message.content.strip()
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()
                decision = json.loads(raw)
            except Exception:
                decision = {"action": "ANSWER", "reason": "default"}

            action = decision.get("action", "ANSWER")
            logger.info(f"[ReAct] Action: {action}")

            # ACT: execute chosen tool
            if action == "ANSWER":
                break

            elif action == "CALCULATE_PRICE":
                result = calculate_product_price(
                    ingredient_cost=150,
                    packaging_cost=30,
                    overhead_per_unit=25,
                    labor_per_unit=80,
                    markup_percent=40
                )
                tool_results.append({
                    "tool": "calculate_price",
                    "result": result
                })

            elif action == "CALCULATE_PROFIT":
                result = calculate_batch_profit(
                    selling_price=350,
                    total_cost_per_unit=250,
                    units_per_batch=12,
                    batches_per_month=20
                )
                tool_results.append({
                    "tool": "calculate_profit",
                    "result": result
                })

            elif action == "CALCULATE_EPF":
                result = calculate_epf_etf(
                    monthly_salary=30000,
                    num_employees=1
                )
                tool_results.append({
                    "tool": "calculate_epf",
                    "result": result
                })

            elif action == "CALCULATE_VAT":
                result = calculate_vat_status(
                    annual_revenue=3600000
                )
                tool_results.append({
                    "tool": "calculate_vat",
                    "result": result
                })

        return {"tool_results": tool_results}

    def run(self, query: str) -> Dict[str, Any]:
        """
        Main orchestration method.
        Full pipeline: Route -> Plan -> ReAct -> Research -> Advise -> Review
        """
        logger.info(f"[Orchestrator] Starting pipeline for: {query[:60]}")
        self.message_log = []

        # STEP 1: ROUTER — classify intent
        routing = classify_intent(query)
        intent = routing.get("intent", "GENERAL")
        key_topics = routing.get("key_topics", [])

        # STEP 2: PLANNING — create execution plan
        plan = self.create_plan(query, intent, key_topics)

        # STEP 3: ReAct LOOP — tool use
        react_result = self.react_loop(query, intent)
        tool_results = react_result.get("tool_results", [])

        # STEP 4: RESEARCH AGENT — retrieve knowledge
        research_msg = AgentMessage(
            sender=self.name,
            receiver="ResearchAgent",
            message_type="task",
            content={
                "query": query,
                "intent": intent,
                "key_topics": key_topics
            }
        )
        research_resp = self.send_message(research_msg)
        research = research_resp.content.get("research", "")
        sources = research_resp.content.get("sources", [])

        # STEP 5: ADVISOR AGENT — generate advice
        advisor_msg = AgentMessage(
            sender=self.name,
            receiver="AdvisorAgent",
            message_type="task",
            content={
                "query": query,
                "intent": intent,
                "plan": plan,
                "research": research,
                "tool_results": tool_results
            }
        )
        advisor_resp = self.send_message(advisor_msg)
        draft = advisor_resp.content.get("advice", "")

        # STEP 6: REVIEWER AGENT — reflection & quality check
        review_msg = AgentMessage(
            sender=self.name,
            receiver="ReviewerAgent",
            message_type="reflection_request",
            content={
                "query": query,
                "draft": draft,
                "research": research
            }
        )
        review_resp = self.send_message(review_msg)

        final = review_resp.content.get("final_response", draft)
        quality = review_resp.content.get("overall_quality", "unknown")
        scores = review_resp.content.get("review_scores", {})

        logger.info(
            f"[Orchestrator] Pipeline complete. Quality: {quality}"
        )

        return {
            "query": query,
            "intent": intent,
            "plan": plan,
            "sources": sources,
            "tool_results": tool_results,
            "final_response": final,
            "quality": quality,
            "review_scores": scores,
            "message_log": self.message_log,
            "model_usage": {
                "router": MODELS["router"]["model_id"],
                "advisor": MODELS["advisor"]["model_id"],
                "reviewer": MODELS["reviewer"]["model_id"]
            }
        }