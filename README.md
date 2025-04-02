# Advanced Local RAG Pipeline with Sentence Transformers, FAISS & IBM Granite3.2 LLM

> Fully local Retrieval-Augmented Generation system for document-based Q&A.

---

## 📚 Project Description

This project implements an **end-to-end RAG pipeline** that supports:

- Sentence-level chunking.
- FAISS vector store for efficient similarity search.
- BAAI `bge-m3` embedding model.
- IBM `granite-3.2-8b-instruct` LLM using Hugging Face.
- Local model caching and loading.
- LangChain integration for chaining retrieval and generation.

---

## ✨ Features

- Automatic document loader (TXT).
- Sentence-wise tokenized chunking with sliding window.
- Embeddings generation with `SentenceTransformer` (BAAI/bge-m3).
- Vector database with FAISS.
- Query answering using IBM Granite 3.2 model.
- Caching and reloading of vector database (avoids recomputation).
- Hash-based change detection.
- Supports CUDA acceleration (tested on CUDA:1).
- Full LangChain-powered pipeline.

---

## 🧩 Components

- **Chunker**: Custom sliding window, sentence-aware chunking.
- **FAISS Vector Store**: Fast nearest neighbor search.
- **BAAI Embedding Model**: Generates high-quality vector embeddings.
- **Granite 3.2 Model**: Locally hosted IBM LLM for text generation.
- **LangChain**: Chains retrieval and generation into a full RAG pipeline.
- **Docstore**: Stores meta-information of all document chunks.

---

## 🛠️ Technologies Used

- Python
- LangChain
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- Numpy
- PyTorch (CUDA)
- Logging

---

## 🗃️ Directory Structure

```
rag_system/
├── data/                          # Your raw text documents
├── faiss_index/                   # Auto-generated vector DB & docstore
│   ├── index.faiss
│   ├── docstore.pkl
│   └── data_hash.txt
├── granite_model/                 # IBM Granite model cache
├── document_mapping.txt           # Optional for human-readable doc info
├── main.py
└── README.md
```

---

## ⚙️ How It Works

1. Load and chunk documents sentence-wise.
2. Generate vector embeddings.
3. Build or load existing FAISS index.
4. Load IBM Granite model (local or auto-download).
5. Assemble retrieval + generation chain.
6. Answer user queries interactively.

---

## 🟢 How to Run

1. Download IBM `granite-3.2-8b-instruct` from Hugging Face.
2. Ensure CUDA is available for GPU acceleration.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place TXT documents inside `/data` folder.
5. Execute:
   ```bash
   python main.py
   ```
6. Query interactively in the terminal.

---

## ✅ Notes

- You can increase `chunk_size`, `overlap`, and `min_chunk_size` inside the `Chunker` for performance tuning.
- Embeddings are cached using FAISS.
- The system automatically detects document changes via hash.
- Use CUDA:1 by default (modify if required).

---

## ✨ Future Improvements

- Multi-turn conversation support.
- Persistent vector store with FAISS.
- Configurable YAML for all parameters.
- Web UI (Streamlit / FastAPI).

---

## ✨ Credits

Developed by Sagar Shankaran as part of IBM Granite & LangChain based RAG experiments.

