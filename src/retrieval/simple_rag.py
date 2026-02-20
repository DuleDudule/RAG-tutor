from src.util.embeddings import get_embedding_model
from src.util.vectorstore import get_vectorstore
from src.util.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage


def simple_chain(query : str):

    embedding_model = get_embedding_model("local")
    vector_store = get_vectorstore(embedding_model,"simple_chunking_whole_book")
    retrieved_docs = vector_store.similarity_search(query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant. Use the following context to answer the users question."\
        "If there isn't enough information to answer the question say you don't know." \
        "Here is the context:"\
        f"\n\n{docs_content}"
    )
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=query)
    ]

    llm = get_llm(mode="local",model_name="qwen3:1.7b")

    for chunk in llm.stream(messages):
        yield chunk.content



