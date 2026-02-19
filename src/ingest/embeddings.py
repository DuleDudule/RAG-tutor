from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
import os
from typing import Literal
import shutil
import subprocess
import json

def require_ollama(model_name):
    if shutil.which("ollama") is None:
        raise RuntimeError("Ollama CLI not found. Install it first.")

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )

        lines = result.stdout.strip().split("\n")[1:] 
        models = [line.split()[0] for line in lines if line.strip()] 

        if model_name not in models:
            raise RuntimeError(
                f"Ollama installed but model '{model_name}' not pulled.\n"
                f"Run: ollama pull {model_name}"
            )
    except FileNotFoundError:
        raise RuntimeError("Ollama is not installed")
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to query Ollama models")


def get_embedding_model(mode : Literal["local","cloud"],model_name : str = "qwen3-embedding:0.6b"):
    """
    Return an embedding model based on .env setup and args.

    Args:
        mode  ("local","cloud") : Local uses Ollama, cloud uses OpenAI api and needs an api_key set.
        model_name (str) : Passed through to Ollama if using local mode, if using cloud set to text-embedding-3-small.

    """
    if mode=="cloud":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in .env to use the openai api")
        model_name = "text-embedding-3-small"
        return OpenAIEmbeddings(model=model_name,api_key=api_key)

    elif mode=="local":
        require_ollama(model_name)
        return OllamaEmbeddings(model=model_name)
 
    else:
        raise ValueError("mode must be 'local' or 'cloud'")



