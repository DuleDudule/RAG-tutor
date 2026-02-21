from src.util.vectorstore import get_vectorstore
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent,AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.util.env_check import get_rag_models

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        "You are a helpful assistant that uses information in "\
        "Data Mining: The Textbook to answer user questions. "\
        "Use the available tool to search for the answer in the book."
        "Make sure to base your answer solely on the snippets from the book."\
        "When writing mathematical formulas, you MUST use LaTeX notation: \n"\
        "- Use double dollar signs for standalone equations (e.g., $$E=mc^2$$).\n"\
        "- Use single dollar signs for inline math (e.g., $x^2$).\n"\
        "Do not use brackets like [ ] or ( ) for math."\
        "If there isn't enough information to answer the question say you don't know." \
        ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

llm, embedding_model = get_rag_models()

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
