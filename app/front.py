import streamlit as st
from src.retrieval.simple_rag import simple_chain
from src.retrieval.rag_agent import rag_agent
from src.util.vectorstore import get_all_collection_names
import sys, os
from dotenv import load_dotenv
load_dotenv() 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
available_collections = get_all_collection_names()

CHAIN_OPTIONS = {
    "Simple RAG (Standard)": simple_chain,
    "Agentic RAG (Tool-Calling)": rag_agent
}

st.title("RAG tutor")

with st.sidebar:
    st.header("Data source")
    if available_collections:
        selected_collection = st.selectbox(
            "Select Qdrant Collection:", 
            options=available_collections
        )
    else:
        st.error("No collections found")
        selected_collection = None

    st.divider()

    st.header("RAG approach")

    selected_chain_name = st.selectbox(
        "Select Reasoning Engine:",
        options=list(CHAIN_OPTIONS.keys()),
        help="Simple RAG always uses context; Agentic RAG decides if it needs the book."
    )
    chosen_chain_func = CHAIN_OPTIONS[selected_chain_name]

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you with Data Mining?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(chosen_chain_func(prompt,collection_name=selected_collection))

    st.session_state.messages.append({"role": "assistant", "content": response})

