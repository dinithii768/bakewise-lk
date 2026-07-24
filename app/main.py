"""
BakeWise LK - Streamlit Application
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Fix Python path - add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

st.set_page_config(
    page_title="BakeWise LK",
    page_icon="🍰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load API key safely
try:
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

def load_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #D4813A 0%, #F5A623 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 800;
    }
    .main-header p {
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .agent-step {
        background: #fff8f0;
        border-left: 4px solid #D4813A;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }
    .quality-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-good {
        background: #d4edda;
        color: #155724;
    }
    .badge-improve {
        background: #fff3cd;
        color: #856404;
    }
    .source-tag {
        display: inline-block;
        background: #e8f4fd;
        color: #0c5460;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 0.2rem;
    }
    .tool-result {
        background: #f0fff4;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def initialize_system():
    """Initialize RAG pipeline and agent system (cached)."""
    try:
        from rag.pipeline import get_retriever
        from agents.orchestrator import OrchestratorAgent

        retriever = get_retriever()
        orchestrator = OrchestratorAgent(retriever)

        stats = retriever.vs.get_stats()
        return orchestrator, stats, True, None

    except Exception as e:
        return None, {}, False, str(e)


def init_session():
    """Initialize session state variables."""
    defaults = {
        "chat_history": [],
        "total_queries": 0,
        "last_result": None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🍰 BakeWise LK</h1>
        <p>Agentic AI Advisor for Sri Lankan Home Food Businesses</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(stats, ready):
    with st.sidebar:
        st.markdown("## 🍰 BakeWise LK")
        st.caption("AI Advisor for Home Food Entrepreneurs")
        st.divider()

        if ready:
            st.success("✅ System Ready")
            chunks = stats.get("total_chunks", 0)
            st.info(f"📚 {chunks} knowledge chunks loaded")
        else:
            st.error("❌ System Not Ready")
            st.warning("Check API key in secrets.toml")

        st.divider()
        st.markdown("### 💡 Sample Questions")

        questions = [
            "How do I register my home bakery?",
            "What must I put on my food label?",
            "How to price my cupcakes correctly?",
            "What loans are available for women bakers?",
            "Do I need to register for VAT?",
            "What allergens must I declare?",
            "How do I sell food on Instagram?",
            "What are EPF obligations for my helper?"
        ]

        for q in questions:
            if st.button(
                f"💬 {q[:38]}...",
                key=f"sq_{hash(q)}",
                use_container_width=True
            ):
                st.session_state["pending_query"] = q
                st.rerun()

        st.divider()
        st.markdown("### 🤖 Agent System")
        agents_info = {
            "🔀 Router": "Classifies intent",
            "📋 Planner": "Creates task plan",
            "🔄 ReAct": "Tool selection loop",
            "📚 Research": "RAG retrieval",
            "💡 Advisor": "Generates advice",
            "🪞 Reviewer": "Quality check"
        }
        for agent, role in agents_info.items():
            st.markdown(f"**{agent}**: {role}")

        st.divider()
        st.markdown("### 🧠 Models Used")
        st.markdown("**Router/Planner:**")
        st.code("llama-3.1-8b-instant\n(Groq - fast + cheap)")
        st.markdown("**Advisor/Reviewer:**")
        st.code("llama-3.3-70b-versatile\n(Groq - high quality)")

        st.divider()
        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True
        ):
            st.session_state["chat_history"] = []
            st.session_state["last_result"] = None
            st.rerun()


def render_chat(orchestrator, ready):
    """Main chat interface."""

    if not st.session_state["chat_history"]:
        st.info("""
        **ආයුබෝවන්! Welcome to BakeWise LK!** 🍰

        I can help you with:
        - 🏢 **Registration** — How to legally start your home food business
        - 🏷️ **Labeling** — What to put on your product labels
        - 💰 **Pricing** — How to price your baked goods profitably
        - 💳 **Financing** — Loans and grants available in Sri Lanka
        - 🛡️ **Compliance** — Food safety and consumer regulations
        - 📱 **Marketing** — Selling online and on social media

        *Ask me anything about running your home food business in Sri Lanka!*
        """)

    # Display chat history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("metadata"):
                meta = msg["metadata"]
                cols = st.columns(4)
                cols[0].caption(
                    f"Intent: `{meta.get('intent', 'N/A')}`"
                )
                cols[1].caption(
                    f"Sources: `{len(meta.get('sources', []))}`"
                )
                cols[2].caption(
                    f"Tools: `{len(meta.get('tool_results', []))}`"
                )
                quality = meta.get("quality", "N/A")
                badge = (
                    "badge-good"
                    if quality == "good"
                    else "badge-improve"
                )
                cols[3].markdown(
                    f'<span class="quality-badge {badge}">'
                    f'✅ {quality}</span>',
                    unsafe_allow_html=True
                )

    # Handle pending query from sidebar buttons
    pending = st.session_state.pop("pending_query", None)
    user_input = st.chat_input(
        "Ask your home food business question..."
    )
    query = pending or user_input

    if not query:
        return

    if not ready or not orchestrator:
        st.error("❌ System not ready. Check API key!")
        return

    # Show user message
    st.session_state["chat_history"].append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    # Process with agents
    with st.chat_message("assistant"):
        status_box = st.status(
            "🤖 Agent pipeline running...",
            expanded=True
        )

        try:
            with status_box:
                st.write("🔀 **Router** — Classifying intent...")
                st.write("📋 **Planner** — Creating execution plan...")
                st.write("🔄 **ReAct** — Checking tools needed...")
                st.write("📚 **Research Agent** — Searching knowledge base...")
                st.write("💡 **Advisor Agent** — Generating advice...")
                st.write("🪞 **Reviewer Agent** — Quality checking...")

            result = orchestrator.run(query)

            status_box.update(
                label="✅ Pipeline Complete",
                state="complete",
                expanded=False
            )

            # Display final response
            final = result.get("final_response", "")
            st.markdown(final)

            # Show intent + plan
            with st.expander(
                f"📋 Execution Plan | Intent: {result.get('intent', 'N/A')}"
            ):
                plan = result.get("plan", [])
                for i, step in enumerate(plan, 1):
                    st.markdown(
                        f'<div class="agent-step">'
                        f'{i}. {step}</div>',
                        unsafe_allow_html=True
                    )

            # Show sources
            sources = result.get("sources", [])
            if sources:
                with st.expander(f"📚 Sources Used ({len(sources)})"):
                    for src in sources:
                        st.markdown(
                            f'<span class="source-tag">📄 {src}</span>',
                            unsafe_allow_html=True
                        )

            # Show tool results if any
            tool_results = result.get("tool_results", [])
            if tool_results:
                with st.expander(
                    f"🔧 Calculator Results ({len(tool_results)})"
                ):
                    for tr in tool_results:
                        tool_name = tr.get("tool", "tool")
                        tool_data = tr.get("result", {})
                        st.markdown(
                            f'<div class="tool-result">'
                            f'<strong>🔧 {tool_name}</strong><br>',
                            unsafe_allow_html=True
                        )
                        for k, v in tool_data.items():
                            if isinstance(v, dict):
                                continue
                            st.markdown(f"- **{k}**: {v}")
                        st.markdown(
                            '</div>',
                            unsafe_allow_html=True
                        )

            # Show agent message flow
            msg_log = result.get("message_log", [])
            if msg_log:
                with st.expander(
                    f"📨 Agent Communication Flow ({len(msg_log)} messages)"
                ):
                    for m in msg_log:
                        sender = m.get("sender", "?")
                        receiver = m.get("receiver", "?")
                        mtype = m.get("message_type", "?")
                        mid = m.get("message_id", "")
                        st.markdown(
                            f"**{sender}** → **{receiver}** "
                            f"| `{mtype}` | `{mid}`"
                        )

            # Show review scores
            scores = result.get("review_scores", {})
            if scores:
                with st.expander("🪞 Review Scores"):
                    cols = st.columns(4)
                    score_items = list(scores.items())
                    for i, (k, v) in enumerate(score_items):
                        if v is not None:
                            cols[i % 4].metric(
                                k.replace("_", " ").title(),
                                f"{v}/5"
                            )

            # Save to history
            metadata = {
                "intent": result.get("intent", ""),
                "sources": result.get("sources", []),
                "tool_results": result.get("tool_results", []),
                "quality": result.get("quality", "unknown")
            }

            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": final,
                "metadata": metadata
            })
            st.session_state["last_result"] = result
            st.session_state["total_queries"] += 1

        except Exception as e:
            status_box.update(
                label="❌ Error",
                state="error",
                expanded=False
            )
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": error_msg
            })


def render_rag_tab(stats):
    """RAG evaluation tab."""
    st.header("🔍 RAG Pipeline Info")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Total Chunks",
        stats.get("total_chunks", 0)
    )
    col2.metric(
        "Embedding Model",
        "MiniLM-L6-v2"
    )
    col3.metric(
        "Vector Store",
        "ChromaDB"
    )

    st.divider()
    st.subheader("📊 Chunking Strategy")
    st.markdown("""
    | Setting | Value |
    |---------|-------|
    | Chunk Size | 700 characters |
    | Chunk Overlap | 120 characters |
    | Splitter | RecursiveCharacterTextSplitter |
    | Separators | paragraph → sentence → word |
    | Embedding Model | all-MiniLM-L6-v2 (384 dim) |
    | Vector Store | ChromaDB (cosine similarity) |
    """)

    st.divider()
    st.subheader("🧪 Retrieval Evaluation")
    st.markdown("""
    | # | Query | Relevant? | Comment |
    |---|-------|-----------|---------|
    | 1 | How to register home food business? | ✅ Yes | Registration doc retrieved |
    | 2 | What allergens to declare on label? | ✅ Yes | Allergen doc retrieved |
    | 3 | How to price cupcakes? | ✅ Yes | Pricing guide retrieved |
    | 4 | Loans for women entrepreneurs? | ✅ Yes | Finance doc retrieved |
    | 5 | Food labeling requirements? | ✅ Yes | Labeling doc retrieved |

    **Average Relevance: 5/5 — GOOD ✅**
    """)

    st.divider()
    st.subheader("🔍 Test Retrieval Live")

    test_q = st.text_input(
        "Enter test query:",
        placeholder="e.g. food labeling requirements"
    )

    if st.button("🔍 Retrieve") and test_q:
        try:
            from rag.pipeline import get_retriever
            retriever = get_retriever()
            docs, context = retriever.retrieve_and_format(
                test_q, top_k=3
            )
            st.success(f"Retrieved {len(docs)} chunks")
            for i, doc in enumerate(docs, 1):
                with st.expander(
                    f"Chunk {i} | "
                    f"{doc.metadata.get('source', 'unknown')} | "
                    f"Score: {doc.metadata.get('retrieval_score', 'N/A')}"
                ):
                    st.text(doc.page_content[:500])
        except Exception as e:
            st.error(f"Retrieval error: {e}")


def render_about_tab():
    """About / architecture tab."""
    st.header("ℹ️ About BakeWise LK")

    st.markdown("""
    ## What is BakeWise LK?

    BakeWise LK is a multi-agent AI system that provides
    **compliance and pricing advice** for Sri Lankan home-based
    food entrepreneurs and home bakers.

    ---

    ## 🏗️ Agentic Design Patterns

    | Pattern | Location | Description |
    |---------|----------|-------------|
    | **Router** | `tools/router_tool.py` | Classifies query intent |
    | **Planning** | `agents/orchestrator.py` | Creates task plan |
    | **ReAct** | `agents/orchestrator.py` | Reason-Act-Observe loop |
    | **Orchestrator-Worker** | `agents/orchestrator.py` | Coordinates agents |
    | **Reflection** | `agents/reviewer_agent.py` | Quality critique |

    ---

    ## 📨 Agent Communication Flow

    ```
    User Query
        ↓
    OrchestratorAgent
        ├── Router Tool     (intent classification)
        ├── create_plan()   (task decomposition)
        ├── react_loop()    (tool selection)
        ├── ResearchAgent   (RAG retrieval)
        ├── AdvisorAgent    (generate advice)
        └── ReviewerAgent   (quality check)
                ↓
        Final Response → User
    ```

    ---

    ## 🧠 Model Selection Strategy

    | Sub-task | Model | Why |
    |----------|-------|-----|
    | Routing & Planning | llama-3.1-8b-instant (Groq) | Ultra fast, cheap |
    | ReAct decisions | llama-3.1-8b-instant (Groq) | Simple decisions |
    | Final advice | llama-3.3-70b-versatile (Groq) | High quality needed |
    | Reflection | llama-3.3-70b-versatile (Groq) | Strong reasoning |

    ---

    ## 📚 RAG Pipeline

    - **Corpus:** 20 Sri Lankan home food business documents
    - **Chunking:** RecursiveCharacterTextSplitter
      (700 chars, 120 overlap)
    - **Embedding:** all-MiniLM-L6-v2 (384 dimensions, FREE, local)
    - **Vector Store:** ChromaDB (persistent, cosine similarity)
    - **Retrieval:** Top-4 semantic search

    ---

    ## ⚠️ Disclaimer

    This AI advisor provides general guidance only.
    Always verify with relevant Sri Lankan authorities
    (IRD, CAA, Department of Labour, etc.)
    before making business decisions.
    """)


def main():
    load_css()
    init_session()
    render_header()

    # Initialize system
    with st.spinner("🔄 Loading BakeWise LK system..."):
        orchestrator, stats, ready, error = initialize_system()

    if error:
        st.error(f"❌ System initialization failed: {error}")
        st.info("""
        **Setup Required:**
        1. Add GROQ_API_KEY to `.streamlit/secrets.toml`
        2. Add documents to `data/raw/`
        3. Run: `streamlit run app/main.py`
        """)

    render_sidebar(stats, ready)

    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "💬 Ask BakeWise",
        "🔍 RAG Pipeline",
        "ℹ️ About"
    ])

    with tab1:
        render_chat(orchestrator, ready)

    with tab2:
        render_rag_tab(stats)

    with tab3:
        render_about_tab()


if __name__ == "__main__":
    main()