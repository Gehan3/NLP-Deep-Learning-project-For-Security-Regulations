from __future__ import annotations

import html
import os
import sys

import streamlit as st

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    import prompt as rag
except ImportError:
    from src import prompt as rag

try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag.OPENROUTER_MODEL = st.secrets.get(
        "OPENROUTER_MODEL",
        rag.OPENROUTER_MODEL,
    )
except Exception:
    # Local runs without a Streamlit secrets file continue to use .env.
    pass

st.set_page_config(
    page_title="ISO 27002 RAG Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background-color: #0B0E14;
        color: #F0F2F6;
        font-family: 'Inter', sans-serif;
    }

    h1 {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.2rem !important;
    }

    h3 {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #151A23;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    .source-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 18px;
        backdrop-filter: blur(12px);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }

    .source-box:hover {
        transform: translateY(-3px);
        border-color: #00C9FF;
    }

    .source-tag {
        display: inline-block;
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        color: #000;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .metric-row {
        color: #94A3B8;
        font-size: 0.78rem;
        line-height: 1.5;
        margin-top: 10px;
    }

    hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_sources" not in st.session_state:
    st.session_state.latest_sources = []

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/000000/security-checked.png",
        width=64,
    )
    st.markdown("## ISO 27002 System")
    st.markdown(
        "An AI-powered dashboard designed to query and explain retrieved "
        "ISO/IEC 27002:2022 context."
    )

    st.divider()
    st.markdown("### System Status")
    st.success("✅ ChromaDB Connected")
    st.success("✅ RAG Pipeline Active")

    st.divider()
    st.caption("Retrieval defaults: 12 candidates → up to 4 reranked sources")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.latest_sources = []
        st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("Powered by Streamlit & OpenRouter")

st.title("ISO 27002 Security Assistant")
st.markdown("*Grounded answers from the retrieved ISO 27002 context.*")
st.divider()

chat_column, sources_column = st.columns([2, 1], gap="large")

with chat_column:
    st.markdown("### 💬 Chat Interface")
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

user_query = st.chat_input(
    "Ask a question or enter a topic such as 'unauthorized access'..."
)

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with chat_column:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_query)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Retrieving controls and generating a grounded response..."):
                try:
                    answer, sources = rag.answer_question(user_query)
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.session_state.latest_sources = sources
                except Exception as error:
                    st.session_state.latest_sources = []
                    st.error(f"An error occurred: {error}")

with sources_column:
    st.markdown("### 📚 Retrieved Sources")

    if not st.session_state.latest_sources:
        st.info(
            "Relevant regulation context and retrieval diagnostics will "
            "appear here after a query."
        )
    else:
        source_cards: list[str] = []

        for index, source in enumerate(st.session_state.latest_sources, start=1):
            control_id = html.escape(str(source.get("control_id") or "Unknown"))
            section = html.escape(str(source.get("section") or "Unknown"))
            parent_id = html.escape(str(source.get("parent_id") or "Not specified"))
            metadata_context = html.escape(
                str(source.get("metadata_context") or "")
            )
            chunk_text = html.escape(
                str(source.get("chunk_text", source.get("text", "")))
            ).replace("\n", "<br>")

            distance = float(source.get("distance", 0.0))
            embedding_score = float(source.get("embedding_score", source.get("score", 0.0)))
            rerank_logit = float(source.get("rerank_logit", 0.0))
            rerank_score = float(source.get("rerank_score", 0.0))
            rerank_rank = source.get("rerank_rank", "?")
            selection_reason = html.escape(
                str(source.get("selection_reason") or "Selected after reranking")
            )

            metadata_line = (
                f'<p style="margin-bottom:10px; font-size:0.85em; '
                f'color:#94A3B8;"><em>{metadata_context}</em></p>'
                if metadata_context
                else ""
            )

            # update for retriever: show raw and normalized retrieval diagnostics
            # so score-related failures are visible in the UI.
            source_cards.append(
                f"""
                <div class="source-box">
                    <div class="source-tag">Source {index}</div>
                    <p style="margin-bottom:6px; font-size:0.95em; color:#E2E8F0;">
                        <strong>Control ID:</strong> {control_id}<br>
                        <strong>Section:</strong> {section}<br>
                        <strong>Parent ID:</strong> {parent_id}
                    </p>
                    {metadata_line}
                    <div style="font-size:0.9em; border-left:3px solid #00C9FF;
                                padding-left:12px; margin-top:8px; color:#CBD5E1;
                                line-height:1.5;">
                        {chunk_text}
                    </div>
                    <div class="metric-row">
                        Chroma distance: {distance:.4f}<br>
                        Embedding diagnostic score: {embedding_score:.4f}<br>
                        Reranker rank: {rerank_rank}<br>
                        Raw reranker logit: {rerank_logit:.4f}<br>
                        Sigmoid reranker score: {rerank_score:.4f}<br>
                        Selection: {selection_reason}
                    </div>
                </div>
                """
            )

        st.markdown("".join(source_cards), unsafe_allow_html=True)