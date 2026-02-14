# RAG-Tutor: A Comparative Study of Text Data Mining & RAG

**Student:** Dušan Jevtović (408/21)  
**Course:** Istraživanje podataka 2 (Data Mining 2)  
**Institution:** Faculty of Mathematics (MATF), University of Belgrade  


---

##  Project Description
My goal is to compare and combine traditional text data mining techniques and Retrieval Augmented Generation. 
The final product is a chatbot augmented with the knowledge from the course literature [Charu C. Aggarwal: Data Mining The Textbook](https://link.springer.com/book/10.1007/978-3-319-14142-8) that students can use to study Data Mining.

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