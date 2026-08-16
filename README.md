# Enterprise RAG Application with FastAPI & Gemini 3.5 Flash

A production-ready Retrieval-Augmented Generation (RAG) backend API built with FastAPI, PostgreSQL (`pgvector`), Sentence-Transformers, and Google Gemini 3.5 Flash LLM.

## 🚀 Features

- **Document Ingestion:** PDF upload, text extraction, and smart chunking (`pypdf`).
- **Vector Search:** High-performance vector embeddings using HuggingFace `sentence-transformers` stored in PostgreSQL with `pgvector` extension.
- **RAG Pipeline:** Context-aware grounding and answer generation powered by Google Gemini 3.5 Flash via `google-genai` SDK.
- **Strict Grounding:** Zero-hallucination prompts strictly enforcing answers based on uploaded document context.
- **Modular Architecture:** Layered project structure (Routers, Services, Schemas, Database Models).

---

## 🛠️ Tech Stack

- **Framework:** FastAPI
- **LLM:** Google Gemini 3.5 Flash (`google-genai` SDK)
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database:** PostgreSQL with `pgvector` extension
- **ORM & Migrations:** SQLAlchemy
- **Language & Runtime:** Python 3.10+ / Uvicorn

---
