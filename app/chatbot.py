import streamlit as st
from src.retrieval.simple_rag import simple_chain
from src.retrieval.rag_agent import rag_agent
from src.util.vectorstore import get_all_collection_names
from langchain_core.messages import HumanMessage, AIMessage
import sys, os
from dotenv import load_dotenv
load_dotenv() 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
available_collections = get_all_collection_names()

CHAIN_OPTIONS = {
    "Simple RAG (Standard)": simple_chain,
    "Agentic RAG (Tool-Calling)": rag_agent
}



st.set_page_config(page_title="RAG Tutor",layout="wide")

st.markdown("""
    <style>
        [data-testid="stVerticalBlockBorderWrapper"] > div:first-child {
            height: 80vh !important;
            max-height: 80vh !important;
        }
    </style>
""", unsafe_allow_html=True)

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

    st.header("Search Type")
    search_type = st.selectbox(
        "Select search type:",
        options=["hybrid", "dense", "sparse"],
        index=0,
        help="hybrid = keyword + semantic, dense = semantic only, sparse = keyword only."
    )
    st.divider()
    st.header("RAG approach")

    selected_chain_name = st.selectbox(
        "Select Reasoning Engine:",
        options=list(CHAIN_OPTIONS.keys()),
        help="Simple RAG always uses context; Agentic RAG decides if it needs the book."
    )
    
    use_history = st.checkbox(
        "Use Chat History", 
        value=True,
        help="Toggle this off if you are using a small model with limited context window."
    )
    
    st.divider()

    st.header("Retrieval Settings")
    top_k = st.slider(
        "Number of chunks to retrieve (k):",
        min_value=2,
        max_value=10,
        value=4,
        help="Higher values provide more context but can confuse the LLM or hit token limits."
    )


    chosen_chain_func = CHAIN_OPTIONS[selected_chain_name]

if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

def stream_handler(generator):
    for item in generator:
        if isinstance(item, str):
            yield item
        elif isinstance(item, list):
            st.session_state.last_chunks = item

main_col, side_col = st.columns([3, 1],gap="medium")
with main_col:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    chat_container = st.container(height=600)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("How can I help you with Data Mining? (English only)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):

                history = []
                if use_history:
                    for msg in st.session_state.messages[:-1]:
                        if msg["role"] == "user":
                            history.append(HumanMessage(content=msg["content"]))
                        else:
                            history.append(AIMessage(content=msg["content"]))

                try:
                    response_gen = chosen_chain_func(prompt, selected_collection, top_k, search_type=search_type, chat_history=history)
                    response = st.write_stream(stream_handler(response_gen))
                    if isinstance(response, list):
                        st.session_state.last_chunks = response

                except ValueError as e:
                    st.error(f"Embedding Error: {str(e)}")
                    st.info("Please select the embedding model that matches this collection or create a new collection.")
                    st.session_state.messages.pop() 
                    st.stop()
                    
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.stop()

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
with side_col:
    st.subheader("Retrieved Chunks")
    if st.session_state.last_chunks:
        for doc in st.session_state.last_chunks:
            source_name = doc.metadata.get('source', 'Unknown').split('/')[-1]
            score = doc.metadata.get('relevance_score', 0)

            with st.expander(f"Source: {source_name}  |  Score: {score:.4f}"):
                # st.progress(min(max(score, 0.0), 1.0))
                st.write(doc.page_content)
    else:
        st.info("Chunks used for the answer will appear here. "\
                "If no chunks appear the model didn't use the retriever tool and search the database, "\
                "it answered from its own knowledge. Try again or use a smarter (larger) model."\
                "Using the 'Simple RAG' option will return the chunks every time since it always searches the database.")