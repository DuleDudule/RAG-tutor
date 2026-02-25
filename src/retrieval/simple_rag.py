from src.util.vectorstore import get_vectorstore
from src.util.env_check import get_rag_models
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from typing import List, Optional


llm, embedding_model, sparse_model = get_rag_models()

def simple_chain(
    query: str,
    collection_name: str,
    top_k: int,
    search_type: str = "hybrid",
    chat_history: Optional[List[BaseMessage]] = None,
):
    """
    Simple rag implementation where the user question is used to
    find the most similar chunks of the book.
    These are passed to the llm as context from which it should answer
    """
    vector_store = get_vectorstore(embedding_model, sparse_model, collection_name, search_type)
    retrieved_docs = vector_store.similarity_search_with_relevance_scores(query,k=top_k)

    formatted_docs = []
    docs_content_list = []

    for doc, score in retrieved_docs:
        doc.metadata["relevance_score"] = score
        formatted_docs.append(doc)

        content = doc.metadata.get("raw_text", doc.page_content) if "preprocessed" in doc.metadata else doc.page_content
        docs_content_list.append(content)

    docs_content = "\n<TEXT CHUNK>\n".join(docs_content_list)


    system_message = (
        "You are an expert Data Mining Tutor helping a student study from Charu C. Aggarwal's 'Data Mining: The Textbook'.\n\n"

        "Your goal is to provide clear, educational responses structured into two distinct parts: \n"
        "1. Theoretical Explanation \n"
        "2. Python Code Implementation \n\n"

        "### Guidelines:\n"
        "- STRICT CONTEXT FOR THEORY: You MUST base your theoretical explanation ONLY on the provided context snippets. Do not invent theories, formulas, or include concepts not found in the text. "
        "If the context does not contain enough information to answer the question, state clearly: 'The provided text does not contain enough information to answer this.'\n"
        "- EXTERNAL KNOWLEDGE FOR CODE: Because the textbook focuses on mathematical theory, you are explicitly allowed and encouraged to use your general programming knowledge to write Python code (e.g., using pandas, numpy, scikit-learn). The code must accurately practically demonstrate the specific theoretical concepts discussed in the context.\n"
        "- MATH FORMATTING: When writing mathematical formulas, you MUST use LaTeX notation:\n"
        "  - Use double dollar signs for standalone equations (e.g., $$E=mc^2$$).\n"
        "  - Use single dollar signs for inline math (e.g., $x^2$).\n"
        "  - Do not use brackets like \\[ \\] or \\( \\) for math.\n"
        "- TONE AND STRUCTURE: Be encouraging, clear, and pedagogical. Use Markdown formatting, clear headings, and bullet points to make your explanations scannable and easy to digest.\n\n"

        "Here is the context:\n"
        f"{docs_content}"
    )

    messages = [SystemMessage(content=system_message)]

    if chat_history:
        messages.extend(chat_history)

    messages.append(HumanMessage(content=query))


    for chunk in llm.stream(messages):
        yield chunk.content

    yield formatted_docs



