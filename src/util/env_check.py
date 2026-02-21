import os
from src.util.embeddings import get_embedding_model as _get_embedding_model
from src.util.llm import get_llm as _get_llm
from dotenv import load_dotenv
load_dotenv() 

def _validate_env():
    """Checks if all required env variables are set."""
    if not os.getenv("LLM_MODEL_NAME") or not os.getenv("EMBEDDING_MODEL_NAME"):
        raise ValueError("LLM_MODEL_NAME and EMBEDDING_MODEL_NAME must be set.")

def get_llm_model():
    _validate_env()
    return _get_llm(os.getenv("LLM_MODE", "local"), os.getenv("LLM_MODEL_NAME"))

def get_embed_model():
    _validate_env()
    return _get_embedding_model(os.getenv("EMBEDDING_MODE", "local"), os.getenv("EMBEDDING_MODEL_NAME"))

def get_rag_models():
    """Returns both for the main RAG chain."""
    return get_llm_model(), get_embed_model()

