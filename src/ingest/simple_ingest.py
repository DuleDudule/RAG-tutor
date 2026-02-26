from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.util.env_check import get_rag_models
from src.util.vectorstore import get_vectorstore
from uuid import uuid4

def simple_ingest(path: str, collection_name: str,stem_and_stop: bool = False, chunk_size: int = 2000, chunk_overlap: int = 200,):
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
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
        )
        texts = text_splitter.split_documents(docs)

        if stem_and_stop:
            from src.util.stemming import preprocess_text
            for text in texts:
                text.metadata["raw_text"] = text.page_content
                text.metadata["preprocessed"] = True
                text.page_content = preprocess_text(text.page_content)
                
        _, embedding_model, sparse_model = get_rag_models()
        vector_store = get_vectorstore(embedding_model, sparse_model, collection_name)

        uuids = [str(uuid4()) for _ in range(len(texts))]

        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_docs = texts[i : i + batch_size]
            batch_ids = uuids[i : i + batch_size]
            
            for idx, doc in enumerate(batch_docs):
                if len(doc.page_content) > 7000: 
                     print(f"Large chunk detected (idx {idx}): {len(doc.page_content)} chars")
            
            try:
                vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            except Exception as e:
                print(f"Error in batch {i // batch_size + 1}: {e}")
                # Print the largest chunk in this batch for inspection
                largest_chunk = max(batch_docs, key=lambda d: len(d.page_content))
                print(f"Largest chunk in failed batch: {len(largest_chunk.page_content)} chars. "\
                      "Try using smaller chunk size and change collection name to avoid errors.")
                print(f"Context preview: {largest_chunk.page_content[:200]}...")
                raise e
        
        return len(texts)

    except Exception as e:
        raise e