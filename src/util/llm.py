from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import os
from typing import Literal
from .ollama import require_ollama

def get_llm(mode : Literal["local","cloud"],model_name : str = "qwen3-embedding:0.6b"):
    """
    Return an LLM instance based on .env setup and args.

    Args:
        mode  ("local","cloud") : Local uses Ollama, cloud uses OpenAI api and needs an api_key set.
        model_name (str) : Passed through to Ollama if using local mode or OpenAI if using cloud.

    """
    if mode=="cloud":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in .env to use the openai api")
        try:
            llm = ChatOpenAI(model=model_name,api_key=api_key)
        except FileNotFoundError:
            raise RuntimeError("Failed calling OpenAI, check your api key and model name")

    elif mode=="local":
        # require_ollama(model_name)
        llm = ChatOllama(model=model_name)
 
    else:
        raise ValueError("mode must be 'local' or 'cloud'")
    
    return llm



