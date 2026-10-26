# 📚 Local Hybrid RAG Assistant

A privacy-first, fully local Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **Ollama (Llama 3.2)**, **ChromaDB**, and **BM25**. It features multi-page PDF ingestion, custom Reciprocal Rank Fusion (RRF) hybrid search, and strict page-level source provenance tracking.

---

## 🚀 Key Features
- **100% Local & Private:** Operates entirely offline using Ollama and local HuggingFace embeddings (`BAAI/bge-small-en-v1.5`).
- **RRF Hybrid Search:** Combines dense vector retrieval (ChromaDB) with sparse keyword matching (BM25) for high precision.
- **Smart PDF Normalization:** Cleans formatting artifacts before chunking to preserve context quality.
- **Provenance Citations:** Forces the LLM to cite exact chunk IDs and page numbers (e.g., `[Chunk_17] (Page 9)`) for every claim.
- **Multi-Turn Chat:** Stateful chat memory managed seamlessly via Streamlit session state.

---

## 🛠️ Tech Stack
- **Frontend & App Framework:** Streamlit
- **LLM Engine:** Ollama (`llama3.2`)
- **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`)
- **Vector Database:** ChromaDB
- **Keyword Search:** Rank-BM25
- **Orchestration:** LangChain

---

## ⚙️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/local-rag-app.git](https://github.com/YOUR_USERNAME/local-rag-app.git)
   cd local-rag-app