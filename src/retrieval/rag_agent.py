from src.util.embeddings import get_embedding_model
from src.util.vectorstore import get_vectorstore
from src.util.llm import get_llm
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent,AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

embedding_model = get_embedding_model("local")
llm = get_llm(mode="local",model_name="qwen3:1.7b")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use the available tool to answer the users question."\
        "If there isn't enough information to answer the question say that you don't know."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)


def rag_agent(query: str,collection_name: str):
    vectorstore = get_vectorstore(embedding_model,collection_name)
    retriever = vectorstore.as_retriever()
    
    @tool
    def retrieve_book_context(query: str) -> str:
        """Search and return information from the Data Mining Textbook."""
        docs = retriever.invoke(query)
        return "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in docs
        )
    tools = [retrieve_book_context]
    
    
    agent = create_tool_calling_agent(llm, tools,prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    for chunk in agent_executor.stream({"input": query}):
        if "output" in chunk:
            yield chunk["output"]
