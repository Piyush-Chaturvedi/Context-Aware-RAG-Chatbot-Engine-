"""
Streamlit demo UI for the dual-mode RAG chatbot.

Run with:
    streamlit run app.py

Two modes (selectable in the sidebar):
  - Customer Support   -> answers from product manuals only
  - Internal Engineer  -> answers from internal standards only (password-gated demo)
"""
import os
import streamlit as st

from src.vectorstore import VectorStore
from src.rag_pipeline import answer_query
from src.config import INDEX_DIR, INTERNAL_MODE_PASSWORD, MODEL_PROVIDER

st.set_page_config(page_title="Appliance Knowledge Assistant", page_icon="🔌", layout="centered")


@st.cache_resource
def load_store(index_filename):
    path = os.path.join(INDEX_DIR, index_filename)
    if not os.path.exists(path):
        return None
    return VectorStore().load(path)


def render_chat(store, mode_label, session_key):
    if store is None:
        st.error(
            "Index not found. Please run `python build_index.py` first to build "
            "the search index from the documents in the data/ folder."
        )
        return

    if session_key not in st.session_state:
        st.session_state[session_key] = []

    for msg in st.session_state[session_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(f"Ask the {mode_label} assistant...")
    if user_input:
        st.session_state[session_key].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                result = answer_query(store, user_input)
            st.markdown(result["answer"])
            if result["mode"] != "no_match":
                with st.expander("View retrieved source excerpts"):
                    for chunk, score in result["retrieved_chunks"]:
                        st.markdown(f"**{chunk['source']}** (relevance: {score:.2f})")
                        st.caption(chunk["text"])

        st.session_state[session_key].append({"role": "assistant", "content": result["answer"]})


def main():
    st.title("🔌 Appliance Knowledge Assistant")
    st.caption(
        "A dual-mode RAG chatbot: customer support from product manuals, "
        "and an internal assistant for engineering standards."
    )

    if MODEL_PROVIDER == "none":
        st.info(
            "Running in **retrieval-only demo mode** (no LLM API key set). "
            "The bot will return the best-matching manual excerpt directly. "
            "Add an API key in `.env` to enable full generated answers — see README.",
            icon="ℹ️",
        )

    mode = st.sidebar.radio("Select mode", ["Customer Support", "Internal Engineering (employees only)"])

    if mode == "Customer Support":
        st.subheader("💬 Customer Support Assistant")
        st.caption("Ask about installation, operation, or troubleshooting for your product.")
        store = load_store("customer_index.pkl")
        render_chat(store, "customer support", "customer_chat_history")

    else:
        st.subheader("🛠️ Internal Engineering Assistant")
        if "internal_authed" not in st.session_state:
            st.session_state.internal_authed = False

        if not st.session_state.internal_authed:
            pw = st.text_input("Employee access password", type="password")
            if st.button("Login"):
                if pw == INTERNAL_MODE_PASSWORD:
                    st.session_state.internal_authed = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            st.caption(
                "Demo-only auth. In production this would be SSO / company login, "
                "not a shared password."
            )
        else:
            st.caption("Ask about component selection, design standards, or specifications.")
            store = load_store("internal_index.pkl")
            render_chat(store, "internal engineering", "internal_chat_history")


if __name__ == "__main__":
    main()
