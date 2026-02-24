
## What is RAG and Why Use It?

Large Language Models (LLMs) are trained on vast datasets, but they possess two major limitations: **knowledge cutoff** (they don't know about data published after their training) and **lack of private context** (they are unaware of your specific data, such as private company documents, proprietary code, or specialized textbooks not in their training set). 

**Retrieval-Augmented Generation (RAG)** addresses these issues by providing the LLM with an intelligent way to access external information based on its semantic meaning. Instead of relying solely on its pre-trained internal weights, the model is supplied with specific, relevant snippets of text retrieved from a trusted source—in our case, Charu C. Aggarwal's *Data Mining: The Textbook*.

### How it works:
1.  **User Query:** The student asks a question.
2.  **Retrieval:** The system searches a database for the most relevant sections of the textbook.
3.  **Augmentation:** The retrieved sections are prepended to the user's question as context.
4.  **Generation:** The LLM generates a response based *only* on the provided context (for theory) while using its general capabilities for practical tasks (like writing Python code).

---

## Embedding Models

As demonstrated in `notebooks/embedding_demo.ipynb`, computers cannot  read text the way humans do. They require numerical representations. 

**Embedding Models** are specialized neural networks that transform unstructured text into dense numerical vectors. These vectors represent **semantic meaning**. In a high-dimensional vector space, sentences like *"Clustering groups similar points"* and *"Partitioning algorithms find data segments"* will be mathematically closer to each other than to a sentence about cats or dogs.

In this project, we support two modes:
- **Local:** Using Ollama (defaulting to `qwen3-embedding:0.6b`), which allows for private, offline processing.
- **Cloud:** Using OpenAI’s `text-embedding-3-small` for higher performance and dimensionality.

---

## Implementation Details

### Step 1: Ingestion (`src/ingest/`)
Ingestion is the process of preparing the textbook for retrieval. We implemented two distinct strategies to compare performance:

*   **Simple Ingest (`simple_ingest.py`):** Uses a `RecursiveCharacterTextSplitter` to break the PDF into chunks of a fixed size (2000 characters). This is universal but blind to the book's structure.
*   **Advanced Ingest (`advanced_ingest.py`):** We use a custom `contents.json` (generated with LLM assistance) to map chapter boundaries. 
    *   **Chapter Awareness:** Chunks are created within chapter boundaries, ensuring a chunk doesn't bridge two unrelated topics.
    *   **Metadata Injection:** We explicitly prepend chapter titles and numbers to the text of each chunk. Since the embedding model only sees the text content, this ensures that the semantic vector includes the context of where the information came from.
    *   **Preprocessing:** Optional stemming and stop-word removal (`src/util/stemming.py`) can be applied to reduce noise. This is included to demonstrate how a more traditional data preprocessing technique could be paired with more modern approaches. However, since modern embedding models are trained on text data consisting of full, grammatically correct senteces this doesnt necessarily improve the retrieval performance.

### Step 2: Retrieval (`src/retrieval/`)
Once the book is vectorized and stored in **Qdrant** (our vector database), we need to find the right information.

*   **Vector Database:** Qdrant stores the embeddings and performs Cosine Similarity searches. When a query comes in, it's embedded, and the database returns the $k$ most similar chunks.
*   **Simple Chain (`simple_rag.py`):** In this simple approach we first explicitly fetch similar documents from the database and then inject them into a prompt along with the user question. We make one LLM call with this prompt and expect an answer grounded in the retrieved context. This simple approach is usefull in a Q&A system where we dont want/need to have an interactive conversation with the LLM - we just want it to answer the question using the contents of the database. While simple and cheap this has its limitations.
*   **Agentic Retrieval (`rag_agent.py`):** We use a Tool Calling agent. Instead of a linear search, the LLM is given a tool (`retrieve_book_context`). The LLM *decides* when it needs more information and what search query to use, allowing for more nuanced multi-step reasoning. This approach lets as ask unrelated questions without confusing the LLM while also allowing it to search the database multiple times with different queries to find the most relevant data about the question.

---

## Notebooks

---

## UI

