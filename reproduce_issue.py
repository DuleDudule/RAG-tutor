import os
from src.util.vectorstore import get_vectorstore
from src.util.env_check import get_rag_models
from src.util.stemming import preprocess_text
from langchain_core.documents import Document

# Setup
llm, embed_model, sparse_model = get_rag_models()
collection_name = "test_stemming_issue"

# Clean up if exists
from qdrant_client import QdrantClient
from pathlib import Path
db_path = Path("data/vector_db/qdrant")
client = QdrantClient(path=str(db_path))
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

# Ingest stemmed document
vector_store = get_vectorstore(embed_model, sparse_model, collection_name, search_type="sparse")
text = "The clustering algorithm is very efficient."
stemmed_text = preprocess_text(text)
print(f"Original: {text}")
print(f"Stemmed: {stemmed_text}")

doc = Document(page_content=stemmed_text, metadata={"raw_text": text, "preprocessed": True})
vector_store.add_documents([doc])

# Search with raw query
query = "What is clustering?"
results_raw = vector_store.similarity_search_with_score(query, k=1)
print(f"Query: '{query}' -> Results: {len(results_raw)}")

# Search with stemmed query
stemmed_query = preprocess_text(query)
results_stemmed = vector_store.similarity_search_with_score(stemmed_query, k=1)
print(f"Stemmed Query: '{stemmed_query}' -> Results: {len(results_stemmed)}")

if len(results_raw) == 0 and len(results_stemmed) > 0:
    print("REPRODUCED: Raw query returned nothing, but stemmed query found the document.")
else:
    print("NOT REPRODUCED or different behavior.")
