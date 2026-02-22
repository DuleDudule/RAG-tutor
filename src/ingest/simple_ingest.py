from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.util.env_check import get_embed_model
from src.util.vectorstore import get_vectorstore
from uuid import uuid4

def simple_ingest(path: str, collection_name: str):
    """
    Ingests a PDF into Qdrant using RecursiveCharacter splitting.
    Returns the count of documents ingested.
    """
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        
        if not docs:
            raise ValueError("The PDF appears to be empty or unreadable.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200,
        )
        texts = text_splitter.split_documents(docs)

        embedding_model = get_embed_model()
        vector_store = get_vectorstore(embedding_model, collection_name)

        uuids = [str(uuid4()) for _ in range(len(texts))]
        vector_store.add_documents(documents=texts, ids=uuids)
        
        return len(texts)

    except Exception as e:
        raise e