from src.util.vectorstore import get_vectorstore
from src.util.env_check import get_rag_models
from langchain_core.messages import SystemMessage, HumanMessage


llm, embedding_model = get_rag_models()

def simple_chain(query : str,collection_name: str,top_k: int):
    """
    Simple rag implementation where the user question is used to 
    find the most similar chunks of the book. 
    These are passed to the llm as context from which it should answer
    """
    vector_store = get_vectorstore(embedding_model,collection_name)
    retrieved_docs = vector_store.similarity_search(query,k=top_k)

    
    
    docs_content = "\n<TEXT CHUNK>\n".join(
        doc.metadata.get("raw_text", doc.page_content) 
        if "preprocessed" in doc.metadata else doc.page_content 
        for doc in retrieved_docs
    )
    
    system_message = (
        "You are a helpful assistant that uses information in "\
        "Data Mining: The Textbook to answer user questions. Use the following context ONLY to answer the users question."\
        "Make sure to base your answer solely on the following snippets of the book."
        "When writing mathematical formulas, you MUST use LaTeX notation: \n"
        "- Use double dollar signs for standalone equations (e.g., $$E=mc^2$$).\n"
        "- Use single dollar signs for inline math (e.g., $x^2$).\n"
        "Do not use brackets like [ ] or ( ) for math."
        "If there isn't enough information to answer the question say you don't know." \
        "Here is the context:"\
        f"\n\n{docs_content}"
    )
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=query)
    ]


    for chunk in llm.stream(messages):
        yield chunk.content

    yield retrieved_docs



