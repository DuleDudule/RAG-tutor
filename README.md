#  RAG-Tutor: Mining Text Data & RAG

**Student:** Dušan Jevtović (408/21)  
**Course:** Istraživanje podataka 2 (Data Mining 2)  
**Institution:** Faculty of Mathematics (MATF), University of Belgrade  


---

##  Description
Demonstrating Retrieval Augmented Generation (RAG) and the power of embedding models to capture semantic meaning behind text data.
The final product is an LLM chatbot augmented with the knowledge from course literature ([Data Mining The Textbook](https://link.springer.com/book/10.1007/978-3-319-14142-8)) that students can use to study Data Mining.  
You can find a more detailed description of the project in the [description.md](description.md) file.



---

##  Setup & Installation

### 1. Prerequisites
- **Python** (3.12)
- **Poetry** (for dependency management)
- **Ollama** (optional, for local LLM support)

### 2. Installation
If you do not have Poetry installed, you can install it via the official installer:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
After installation, ensure that the Poetry binary is in your $PATH. You can verify the installation by running:
```bash
poetry --version
```
Clone the repository and install dependencies:
```bash
git clone git@github.com:DuleDudule/RAG-tutor.git
cd RAG-tutor
poetry install
```

### 3. Setup
Copy the .env.example file and set the required variables
```bash
cp .env.example .env
```
If you're going for local execution set the LLM_MODE and EMBEDDING_MODE to "local". Pull the model you want to use and set that model name in .env :
```bash
ollama pull <model_name>
```

If you want to stick to small default models i chose (the ones set in .env.example) do:
```bash
ollama pull qwen3-embedding:0.6b && ollama pull qwen3:1.7b
```
If you want to use the OpenAI api you need to set the appropriate variables to "cloud" and configure and set an api key and model name supported by the OpenAI api

### 4. How to use
To run the app run the following in the terminal:
```bash
poetry run python -m streamlit run app/chatbot.py
```
The repo comes with the book already ingested using different strategies and embedding models. Select them from the dropdown menu on the chat page and get to studying.

If you wish to experiment with your own model or chunking parameters then navigate to the ingest page in the sidebar, choose your preprocessing strategy and upload the Data Mining Textbook.

Now on the chatbot page select the collection you just uploaded, adjust parameters and chat with the LLM equiped with the knowledge from the book.

By uploading the book using different strategies and choosing those collections on the chat page you can compare the quality of the answers.

---
## 5. Considerations
- **Embedding model** - Different embedding models produce embeddings (vectors) of different sizes. 
If you use one model to process the book and save it to the vector database and then change the model later you might run into errors. Make sure to match the embedding model you use for RAG with the one used to ingest the document.
The pre-ingested collections labeled with "openai" were ingested using OpenAI's `text-embedding-3-small` model. The ones labeled with "ollama" were ingested using the default embedding model set in .env.example (`qwen3-embedding:0.6b`).

- **RAG** - Answer quality changes drastically based on the capabilities of the LLM you use. 
For best results use the largest model your system can handle locally or the OpenAI api. 
If you're running a model localy make sure it supports "tool calling". That is how the LLM interacts with the database. 
Sometimes a smaller model won't listen to instructions well and it responds to the user question from its own knowledge instead of using the data we ingested into the database.