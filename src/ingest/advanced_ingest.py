import json
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.util.env_check import get_rag_models
from src.util.vectorstore import get_vectorstore
from pathlib import Path

def advanced_ingest(path: str, collection_name: str,stem_and_stop: bool = False, chunk_size: int = 2000, chunk_overlap: int = 200,  page_offset: int = 26):
    """
    Ingests a PDF into Qdrant by first splitting it into chapters based on a JSON mapping,
    merging chapter pages, chunking them, and injecting metadata into the text.
    Returns the count of documents ingested.
    """
    try:
        project_root = Path(__file__).resolve().parents[2]
        json_path = project_root / "data" / "processed" / "contents.json"

        with open(json_path, "r") as f:
            chapters_json = json.load(f)
            
        loader = PyPDFLoader(path)
        pages = loader.load()
        
        if not pages:
            raise ValueError("The PDF appears to be empty or unreadable.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        all_chunks = []
        
        for chapter in chapters_json:
            start_page = chapter["start_page"]
            end_page = chapter["end_page"]
            
            start_idx = start_page + page_offset - 1
            
            if end_page is not None:
                end_idx = end_page + page_offset - 1
                chapter_pages = pages[start_idx : end_idx + 1] 
            else:
                chapter_pages = pages[start_idx : ]
                
            chapter_text = "\n".join([page.page_content for page in chapter_pages])
            
            if len(chapter_pages) > 0:
                chapter_metadata = chapter_pages[0].metadata.copy()
            else:
                chapter_metadata = {}
            
            keys_to_remove = [
                "page", "page_label", "subject", "producer", 
                "creator", "creationdate", "author", "moddate", "title"
            ]
            for key in keys_to_remove:
                chapter_metadata.pop(key, None)

            chapter_metadata["chapter_number"] = chapter["chapter_number"]
            chapter_metadata["chapter_title"] = chapter["title"]
            
            chapter_doc = Document(page_content=chapter_text, metadata=chapter_metadata)
            chapter_chunks = text_splitter.split_documents([chapter_doc])
            
            
            all_chunks.extend(chapter_chunks)    
            
        for chunk in all_chunks:
            ch_num = chunk.metadata.get("chapter_number", "Unknown")
            ch_title = chunk.metadata.get("chapter_title", "Unknown Title")
            source_file = chunk.metadata.get("source", "Unknown Source")
            
            metadata_header = (
                f"Chapter {ch_num}: {ch_title}\n"
                f"Source: {source_file.split('/')[-1]}\n"
                f"----------\n"
            )

            if stem_and_stop:
                from src.util.stemming import preprocess_text
                chunk.metadata["raw_text"] = chunk.page_content
                chunk.metadata["preprocessed"] = True
                chunk.page_content = preprocess_text(chunk.page_content)
            chunk.page_content = metadata_header + chunk.page_content

        _, embedding_model, sparse_model = get_rag_models()
        vector_store = get_vectorstore(embedding_model, sparse_model, collection_name)

        uuids = [str(uuid4()) for _ in range(len(all_chunks))]
        vector_store.add_documents(documents=all_chunks, ids=uuids)
        
        return len(all_chunks)

    except Exception as e:
        raise e