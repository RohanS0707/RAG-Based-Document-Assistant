"""Streamlit interface for the PDF RAG assistant."""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_assistant.pipeline import answer_question, build_index, load_pdf_documents, make_embeddings, split_documents

load_dotenv()
st.set_page_config(page_title="Document Intelligence Assistant", page_icon="📄", layout="wide")
st.title("📄 Document Intelligence Assistant")
st.caption("Ask questions about a PDF. Answers are grounded in retrieved passages with page citations.")

with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])
    st.caption("Indexing uses local sentence-transformer embeddings and FAISS. First model load may take a few minutes.")
    llm_enabled = st.toggle("Use configured LLM for answers", value=False)
    st.caption("Requires OPENAI_API_KEY in .env. Without it, an extractive fallback is used.")

if uploaded is None:
    st.info("Upload a PDF to create a searchable index.")
    st.stop()

file_signature = f"{uploaded.name}:{uploaded.size}"
if st.session_state.get("indexed_file") != file_signature:
    try:
        with st.spinner("Reading PDF, loading local embeddings, and building the FAISS index..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                pdf_path = Path(tmp_dir) / Path(uploaded.name).name
                pdf_path.write_bytes(uploaded.getvalue())
                documents = load_pdf_documents(pdf_path)
                chunks = split_documents(documents)
                if not chunks:
                    raise ValueError("No text chunks could be created from this PDF.")
                index = build_index(chunks, make_embeddings())
        st.session_state["rag_index"] = index
        st.session_state["indexed_file"] = file_signature
        st.session_state["chunk_count"] = len(chunks)
        st.session_state["page_count"] = len(documents)
        st.session_state["messages"] = []
    except Exception as exc:
        st.error(f"Could not index this PDF: {exc}")
        st.stop()

st.success(f"Indexed {st.session_state['page_count']} text pages into {st.session_state['chunk_count']} chunks. Retrieval returns the top 4 passages.")

for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"Retrieved evidence ({len(message['sources'])})"):
                for i, src in enumerate(message["sources"], 1):
                    st.markdown(f"**{i}. {src['source']} — page {src['page']}**")
                    st.caption(src["text"])
        if message.get("mode"):
            st.caption(f"Answer mode: {message['mode']}")

question = st.chat_input("Ask something about this PDF")
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving relevant passages and preparing an answer..."):
                result = answer_question(st.session_state["rag_index"], question, use_llm=llm_enabled)
            st.markdown(result.answer)
            st.caption(f"Answer mode: {result.answer_mode}")
            with st.expander(f"Retrieved evidence ({len(result.sources)})"):
                for i, src in enumerate(result.sources, 1):
                    st.markdown(f"**{i}. {src['source']} — page {src['page']}**")
                    st.caption(src["text"])
            st.session_state["messages"].append({
                "role": "assistant", "content": result.answer,
                "sources": result.sources, "mode": result.answer_mode,
            })
        except Exception as exc:
            st.error(f"Could not answer this question: {exc}")
