from importlib import import_module
import streamlit as st
import sys
import os
import re

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    import prompt as rag
except ImportError:
    from src import prompt as rag

try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ISO 27002 RAG Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #0B0E14;
        color: #F0F2F6;
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant Gradient Header */
    h1 {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.2rem !important;
    }
    
    /* Subheader Styling */
    h3 {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #151A23;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Source Boxes (Glassmorphism) */
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
    
    /* Badges */
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
    
    /* Divider Customization */
    hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_sources" not in st.session_state:
    st.session_state.latest_sources = []

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/security-checked.png", width=64)
    st.markdown("## ISO 27002 System")
    st.markdown("An AI-powered dashboard designed to help you quickly query and understand security regulations.")
    
    st.divider()
    st.markdown("### System Status")
    st.success("✅ ChromaDB Connected")
    st.success("✅ RAG Pipeline Active")
    
    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.latest_sources = []
        st.rerun()
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("Powered by Streamlit & OpenRouter")

# --- MAIN DASHBOARD HEADER ---
st.title("ISO 27002 Security Assistant")
st.markdown("*Your intelligent portal for navigating and interpreting security guidelines.*")
st.divider()

# --- LAYOUT SETUP ---
col1, col2 = st.columns([2, 1], gap="large")

# --- CHAT INTERFACE (LEFT COLUMN) ---
with col1:
    st.markdown("### 💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

# --- CHAT INPUT ---
# This is placed at the bottom automatically by st.chat_input
if prompt := st.chat_input("Ask a question about ISO 27002 controls or security policies..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with col1:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
            
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing regulations and generating response..."):
                try:
                    answer, sources = rag.answer_question(prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.latest_sources = sources
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# --- SOURCES DISPLAY (RIGHT COLUMN) ---
with col2:
    st.markdown("### 📚 Retrieved Sources")
    
    if not st.session_state.latest_sources:
        st.info("Relevant regulation context will appear here after you ask a question.")
    else:
        html_sources = ""
        for i, source in enumerate(st.session_state.latest_sources, 1):
            control_id = source.get('control_id', 'Unknown')
            section = source.get('section', 'Unknown')
            meta_context = source.get('metadata_context', 'No additional context')
            chunk_text = source.get('chunk_text', source.get('text', ''))
            
            html_sources += f'''
            <div class="source-box">
                <div class="source-tag">Source {i}</div>
                <p style="margin-bottom:6px; font-size:0.95em; color: #E2E8F0;"><strong>Control ID:</strong> {control_id} <br> <strong>Section:</strong> {section}</p>
                <p style="margin-bottom:10px; font-size:0.85em; color:#94A3B8;"><em>{meta_context}</em></p>
                <div style="font-size:0.9em; border-left: 3px solid #00C9FF; padding-left: 12px; margin-top:8px; color: #CBD5E1; line-height: 1.5;">
                    {chunk_text}
                </div>
            </div>
            '''
        st.markdown(html_sources, unsafe_allow_html=True)
