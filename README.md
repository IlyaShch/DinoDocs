# 🦕 RAG-Based Code Documentation Assistant

This project is a full-stack **Retrieval-Augmented Generation (RAG)** application designed to parse and understand technical code documentation from PDF files and answer natural language questions about them. It uses **FastAPI**, **PineconeDB**, **minishlab/potion-base-8M**, and **Gemini LLM**, with a **React frontend**.

---

## 🚀 Features

- 📝 Parses PDF documentation using `PyPDF2`
- 🔍 Stores and retrieves semantic chunks via `Pinecone`
- 🤖 Uses Gemini to generate natural language answers from retrieved context
- 🌐 FastAPI backend with a React frontend for interactive querying

---

## 🗂 Project Structure

```bash
├── model.py             # Core RAG pipeline (embeddings, vector search, Gemini query)
├── modelManager.py      # PineconeModelManager (index upsert/query logic)
├── api.py               # FastAPI server (serves frontend, handles query requests)
├── index.html           # React frontend (user interface for querying)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```