from ..util.embeddings import get_embedding_model
from ..util.vectorstore import get_vectorstore
from ..util.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage


def simple_chain(query : str):

    embedding_model = get_embedding_model("local")
    vector_store = get_vectorstore(embedding_model,"simple_chunking_first_50")
    retrieved_docs = vector_store.similarity_search(query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=query)
    ]

    llm = get_llm("local","llama3.2:1b")

    response = llm.invoke(messages)

    return response.content

print(simple_chain("Who does the author acknowledge?"))