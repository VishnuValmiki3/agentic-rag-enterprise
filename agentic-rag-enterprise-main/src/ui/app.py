"""
Streamlit UI — Enterprise Document Q&A.
Run: streamlit run src/ui/app.py
"""
import streamlit as st
import time
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid import HybridRetriever
from src.agent.graph import build_agent, query


@st.cache_resource
def load_system():
    """Load the RAG system (cached across reruns)."""
    vector_store = VectorStore(persist_dir="./data/chroma_db")
    retriever = HybridRetriever(vector_store=vector_store)
    agent = build_agent(retriever)
    return agent, vector_store


def main():
    st.set_page_config(
        page_title="Enterprise Document Q&A",
        page_icon="📄",
        layout="wide",
    )

    st.title("📄 Enterprise Document Q&A")
    st.caption("Agentic RAG with self-correcting retrieval, hybrid search, and cited answers")

    # Sidebar
    with st.sidebar:
        st.header("System Info")
        try:
            agent, vector_store = load_system()
            doc_count = vector_store.count
            st.success(f"✅ System loaded — {doc_count} chunks indexed")
        except Exception as e:
            st.error(f"❌ Error loading system: {e}")
            st.info("Run the ingestion pipeline first:\n`python -m src.ingestion.pipeline --pdf-dir data/sample_pdfs/`")
            return

        st.divider()
        st.header("Settings")
        show_sources = st.toggle("Show source chunks", value=True)
        show_latency = st.toggle("Show latency breakdown", value=False)

    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and "sources" in msg:
                if show_sources and msg["sources"]:
                    with st.expander(f"📚 Sources ({len(msg['sources'])} chunks)"):
                        for i, source in enumerate(msg["sources"]):
                            st.markdown(f"**Chunk {i+1}** — Page {source['page']}, {source['section']}")
                            st.text(source["text"][:300] + "...")
                            st.caption(f"Relevance score: {source['score']:.3f}")
                            st.divider()

                if show_latency and "latency" in msg:
                    with st.expander("⏱️ Latency breakdown"):
                        for step, duration in msg["latency"].items():
                            st.text(f"{step}: {duration*1000:.0f}ms")

    # User input
    if user_input := st.chat_input("Ask a question about your documents..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                result = query(agent, user_input)

            st.markdown(result["answer"])

            if result.get("retries", 0) > 0:
                st.info(f"ℹ️ Query was rewritten {result['retries']} time(s) to improve retrieval")

            # Show sources
            if show_sources and result.get("sources"):
                with st.expander(f"📚 Sources ({len(result['sources'])} chunks)"):
                    for i, source in enumerate(result["sources"]):
                        st.markdown(f"**Chunk {i+1}** — Page {source['page']}, {source['section']}")
                        st.text(source["text"][:300] + "...")
                        st.caption(f"Relevance score: {source['score']:.3f}")
                        st.divider()

            if show_latency:
                with st.expander("⏱️ Latency breakdown"):
                    for step, duration in result.get("latency", {}).items():
                        st.text(f"{step}: {duration*1000:.0f}ms")

        # Store in session
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "latency": result.get("latency", {}),
        })


if __name__ == "__main__":
    main()
