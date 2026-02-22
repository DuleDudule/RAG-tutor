from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from pathlib import Path

_QDRANT_CLIENT = None  

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
db_path = project_root / "data" / "vector_db" / "qdrant"
db_path.parent.mkdir(parents=True, exist_ok=True)

def _get_client() -> QdrantClient:
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is not None:
        try:
            _QDRANT_CLIENT.get_collections()
        except RuntimeError as e:
            if "closed" in str(e).lower():
                _QDRANT_CLIENT = None

    if _QDRANT_CLIENT is None:
        _QDRANT_CLIENT = QdrantClient(path=db_path)
    return _QDRANT_CLIENT

def get_vectorstore(embedding_model : OllamaEmbeddings | OpenAIEmbeddings,collection_name : str):
    """
    Return vectorstore connected to a local collection. 

    Args:
        embedding_model (OllamaEmbeddings,OpenAIEmbeddings) : Embedding model used to embed documents and queries.
        collection_name (str) : Name of qdrant collection we are connecting to. If the collection doesn't exist creates it.

    """
    global _QDRANT_CLIENT

    client = _get_client()
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


def get_all_collection_names():
    """Fetch names of all collections available."""
    global _QDRANT_CLIENT
    
    client=_get_client()  
    collections = client.get_collections().collections
    return [c.name for c in collections]