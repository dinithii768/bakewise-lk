"""
Router Tool for BakeWise LK

PATTERN: Router
ROLE: Classify user intent into categories
MODEL: llama-3.1-8b-instant (fast + cheap for classification)
"""

import json
import logging
from groq import Groq
from config import GROQ_API_KEY, MODELS

logger = logging.getLogger(__name__)


INTENT_CATEGORIES = {
    "REGISTRATION": "Business registration, licensing, legal setup",
    "FOOD_SAFETY": "Food hygiene, PHI, health standards",
    "LABELING": "Product labels, allergens, packaging info",
    "PRICING": "Cost calculation, markup, pricing strategy",
    "FINANCING": "Loans, grants, microfinance, funding",
    "MARKETING": "Social media, online selling, promotions",
    "COMPLIANCE": "CAA, consumer rights, regulations",
    "EQUIPMENT": "Baking tools, kitchen setup",
    "GENERAL": "General home food business advice"
}


def classify_intent(query: str) -> dict:
    """
    ROUTER PATTERN: Classify user query into intent category.
    Uses fast cheap model for this simple classification task.
    """
    client = Groq(api_key=GROQ_API_KEY)

    categories_text = "\n".join([
        f"- {k}: {v}" for k, v in INTENT_CATEGORIES.items()
    ])

    prompt = f"""Classify this home food business question into ONE category.

Question: "{query}"

Categories:
{categories_text}

Respond with JSON only:
{{"intent": "CATEGORY_NAME", "confidence": 0.95, "key_topics": ["topic1", "topic2"]}}"""

    try:
        response = client.chat.completions.create(
            model=MODELS["router"]["model_id"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )

        raw = response.choices[0].message.content.strip()

        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        logger.info(f"[Router] Intent: {result.get('intent')}")
        return result

    except Exception as e:
        logger.error(f"[Router] Classification failed: {e}")
        return {
            "intent": "GENERAL",
            "confidence": 0.5,
            "key_topics": []
        }