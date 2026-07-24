from importlib import import_module

import streamlit as st

rag = import_module("prompt")

try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass

st.title("ISO 27002 Security Regulations RAG Assistant")

question = st.text_area("Question")

if st.button("Answer") and question.strip():
    answer, sources = rag.answer_question(question)
    st.text_area("Answer", value=answer, height=220)

    with st.expander("Sources"):
        for source in sources:
            st.write(source["title"])
            st.write(source["chunk_text"])

    with st.expander("View Retrieved Sources"):
        for i, source in enumerate(sources, 1):
            st.markdown(f"Source No{i}")
        
            control_id = source.get('control_id')
            section = source.get('section')
            meta_context = source.get('metadata_context')
            
            st.markdown(f"**Control ID:** `{control_id}` | **Section:** `{section}`")
            st.markdown(f"**Context:** {meta_context}")
            
            chunk_text = source.get('chunk_text', source.get('text', ''))
            st.info(chunk_text)
            st.divider()