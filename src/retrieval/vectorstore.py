from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

load_dotenv(find_dotenv())


def get_vectorstore(embedding_model : OllamaEmbeddings | OpenAIEmbeddings,collection_name : str):
    """
    Return vectorstore connected to a local collection. 

    Args:
        embedding_model (OllamaEmbeddings,OpenAIEmbeddings) : Embedding model used to embed documents and queries.
        collection_name (str) : Name of qdrant collection we are connecting to. If the collection doesn't exist creates it.

    """
    # Ensuring database path stays consistent
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent.parent.parent
    db_path = project_root / "data" / "vector_db" / "qdrant"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    client = QdrantClient(path=db_path)
    embedding_dim = len(embedding_model.embed_query("hello world"))

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )

    return vector_store