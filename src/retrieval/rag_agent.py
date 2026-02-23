from src.util.vectorstore import get_vectorstore
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent,AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.util.env_check import get_rag_models

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", (
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
            
            "You MUST use the available tool to search for the answer in the book."
        )),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

llm, embedding_model = get_rag_models()

def rag_agent(query: str,collection_name: str,top_k: int):
    vectorstore = get_vectorstore(embedding_model,collection_name)
    retriever = vectorstore.as_retriever(search_kwargs ={"k":top_k})
    retrieved_docs = []
    @tool
    def retrieve_book_context(query: str) -> str:
        """Search and return information from the Data Mining Textbook."""
        docs = retriever.invoke(query)
        retrieved_docs.extend(docs)
        return "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.metadata.get('raw_text', doc.page_content) if doc.metadata.get('preprocessed') else doc.page_content}"
            for doc in docs
        )
    tools = [retrieve_book_context]
    
    
    agent = create_tool_calling_agent(llm, tools,prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    for chunk in agent_executor.stream({"input": query}):
        if "output" in chunk:
            yield chunk["output"]

    yield retrieved_docs