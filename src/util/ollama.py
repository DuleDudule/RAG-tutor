import shutil
import subprocess

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

