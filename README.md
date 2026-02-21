#  RAG-Tutor: Minig Text Data & RAG

**Student:** Dušan Jevtović (408/21)  
**Course:** Istraživanje podataka 2 (Data Mining 2)  
**Institution:** Faculty of Mathematics (MATF), University of Belgrade  


---

##  Project Description
My goal is to compare and combine traditional data mining techniques for text and Retrieval Augmented Generation. 
The final product is a chatbot augmented with the knowledge from course literature [Data Mining The Textbook](https://link.springer.com/book/10.1007/978-3-319-14142-8) that students can use to study Data Mining.

##  Project Goals
1. **Preprocessing:** Implement text cleaning and chunking pipelines.
2. **Analysis:** Compare different retrieval algorithms and preprocessing techniques.
3. **Visualization:** 2D/3D visualization of the book's vector space.
4. **Interactive UI:** A chatbot interface allowing users to toggle between different data mining strategies.

---

##  Setup & Installation

### 1. Prerequisites
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
If you're going for local execution set the appropriate variables to "local". Pull the model you want to use and set that model name in .env. :
```bash
ollama pull <model_name>
```

If you want to stick to small default models i chose (the ones set in .env.example) do:
```bash
ollama pull qwen3-embedding:0.6b && ollama pull qwen3:1.7b
```
If you want to use te OpenAI api you need to set the appropriate variables to "cloud" and configure and set an api key and model name supported by the OpenAI api

### 4. Considerations
- **Embedding model** - Different embedding models produce embeddings (vectors) of different sizes. If you use one model to process the book and save it to the vector database and then change the model later you might run into errors. Make sure to match the embedding model you pass to the vectorstore in the retrieval phase to the one used to ingest the document.
- **RAG** - Answer quality changes drastically based on the capabilities of the LLM you use. For best results use the largest model your system can handle locally or the OpenAI api. If you're running a model localy make sure it supports "tool calling". That is how the llm interacts with the database.