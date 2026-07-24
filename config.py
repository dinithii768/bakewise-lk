import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base Paths ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
CHROMA_DB_PATH = str(PROCESSED_DATA_DIR / "chroma_db")

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── API Keys ────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Model Selection Strategy ────────────────────────────────
# Deliberately using 2 different Groq models for different tasks
# Model 1: llama-3.1-8b-instant  → fast, cheap, routing/planning
# Model 2: llama-3.3-70b-versatile → strong reasoning, final advice
MODELS = {
    "router": {
        "model_id": "llama-3.1-8b-instant",
        "temperature": 0.1,
        "max_tokens": 512,
        "purpose": "Intent routing, classification, planning - low cost, ultra fast"
    },
    "advisor": {
        "model_id": "llama-3.3-70b-versatile",
        "temperature": 0.4,
        "max_tokens": 2048,
        "purpose": "Final advice generation, deep reasoning, synthesis"
    },
    "reviewer": {
        "model_id": "llama-3.3-70b-versatile",
        "temperature": 0.2,
        "max_tokens": 1024,
        "purpose": "Reflection, self-critique, quality checking"
    }
}

# ── RAG Configuration ───────────────────────────────────────
RAG_CONFIG = {
    "chunk_size": 700,
    "chunk_overlap": 120,
    "top_k": 4,
    "collection_name": "bakewise_lk_kb",
    "embedding_model": "all-MiniLM-L6-v2",
    "similarity_threshold": 0.3
}

# ── Agent Configuration ─────────────────────────────────────
AGENT_CONFIG = {
    "max_react_iterations": 3,
    "max_reflection_loops": 1
}

# ── Agentic Patterns Used ───────────────────────────────────
PATTERNS = {
    "Router":             "tools/router_tool.py  →  classify_intent()",
    "Planning":           "agents/orchestrator.py →  create_plan()",
    "ReAct":              "agents/orchestrator.py →  react_loop()",
    "Orchestrator-Worker":"agents/orchestrator.py →  coordinates ResearchAgent + AdvisorAgent",
    "Reflection":         "agents/reviewer.py     →  reflect_and_improve()"
}