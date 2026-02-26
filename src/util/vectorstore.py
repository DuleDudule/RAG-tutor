from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
import atexit

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
        _QDRANT_CLIENT = QdrantClient(path=db_path,force_disable_check_same_thread= True)
        atexit.register(close_qdrant_client)
    return _QDRANT_CLIENT

def close_qdrant_client():
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is not None:
        try:
            _QDRANT_CLIENT.close()
        except Exception:
            pass
        _QDRANT_CLIENT = None

mode_mapping = {
    "dense": RetrievalMode.DENSE,
    "sparse": RetrievalMode.SPARSE,
    "hybrid": RetrievalMode.HYBRID,
}

def get_vectorstore(
    embedding_model: OllamaEmbeddings | OpenAIEmbeddings,
    sparse_embedding_model: FastEmbedSparse,
    collection_name: str,
    search_type: str = "hybrid",
):
    """
    Return vectorstore connected to a local collection. 

    Args:
        embedding_model: Dense embedding model (Ollama or OpenAI).
        sparse_embedding_model: Sparse BM25 model (FastEmbedSparse).
        collection_name: Qdrant collection name. Created if it doesnt exist.
        search_type: One of 'dense', 'sparse', or 'hybrid'. 
    """
    global _QDRANT_CLIENT

    client = _get_client()
    embedding_dim = len(embedding_model.embed_query("hello world"))

    selected_mode = mode_mapping.get(search_type, RetrievalMode.HYBRID)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=True)
                )
            },
        )
    else:
        collection_info = client.get_collection(collection_name)
        existing_size = collection_info.config.params.vectors.size

        if existing_size != embedding_dim:
            raise ValueError(
                f"Dimension Mismatch! Collection '{collection_name}' expects {existing_size} "
                f"dimensions, but the current model provides {embedding_dim}. "
            )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
        sparse_embedding=sparse_embedding_model,
        retrieval_mode=selected_mode,
        sparse_vector_name="sparse",
    )

    return vector_store


def get_all_collection_names():
    """Fetch names of all collections available."""
    global _QDRANT_CLIENT
    
    client=_get_client()  
    collections = client.get_collections().collections
    return [c.name for c in collections]