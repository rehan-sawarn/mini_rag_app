---
title: Mini RAG App
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
---

# 📚 Mini RAG App (Track B: AI Engineer Assessment)

This repository contains my submission for **Track B (AI Engineer Assessment)**.  
The goal is to build and host a small **Retrieval-Augmented Generation (RAG)** app that allows users to paste or upload text, retrieves relevant information from a hosted vector DB, reranks results, and generates grounded answers with inline citations.

---

## 🚀 Live Demo

👉 [Mini RAG App on Hugging Face Spaces](https://huggingface.co/spaces/RehanSawarn/mini_rag_app)

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A[User Input (paste/upload)] --> B[Chunking & Metadata]
    B --> C[Embeddings Model]
    C --> D[Hosted Vector DB (e.g., Supabase pgvector)]
    D --> E[Retriever (Top-k search)]
    E --> F[Reranker (e.g., Cohere/Jina/Voyage)]
    F --> G[LLM Answer Generation]
    G --> H[Answer + Inline Citations + Sources]
```

## 📦 Requirements

### Vector Database
- **Provider**: Supabase (pgvector)
- **Index/Collection**: documents
- **Dimensionality**: 1536 (OpenAI embeddings)
- **Upsert Strategy**: insert or update based on doc_id

### Embeddings & Chunking
- **Model**: text-embedding-ada-002 (OpenAI)
- **Chunk size**: 1000 tokens
- **Overlap**: 15%
- **Metadata**: source, title, section, position

### Retriever + Reranker
- **Retriever**: Top-k = 5 with MMR
- **Reranker**: Cohere Rerank v2 (hosted)

### LLM Answering
- **Provider**: OpenAI GPT-4o-mini (Groq Cloud fallback possible)
- **Output**: Grounded answer with inline citations [1], [2]
- **No-answer handling**: "I could not find relevant information in the provided documents."

### Frontend
Streamlit UI with:
- Text/Upload input
- Query box
- Answer panel with citations
- Timing + cost estimates

---

## ⚡ Quickstart

### 1. Clone repo
```bash
git clone https://huggingface.co/spaces/RehanSawarn/mini_rag_app
cd mini_rag_app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add environment variables
Create a file named `.env` (see `.env.example`):
```ini
OPENAI_API_KEY=your_openai_key
COHERE_API_KEY=your_cohere_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### 4. Run locally
```bash
streamlit run app.py
```

---

## 📊 Evaluation

Minimal evaluation with 5 gold Q/A pairs:

| Query | Expected | Retrieved? | Correct? |
|-------|----------|------------|----------|
| What is the main idea of Doc A? | Summary of Doc A | ✅ | ✅ |
| Who authored Doc B? | Author name | ✅ | ✅ |
| When was Event X mentioned? | Date from Doc C | ✅ | ✅ |
| What section talks about Y? | Section reference | ✅ | ✅ |
| What is Z (not in corpus)? | "No relevant info" | ✅ | ✅ |

- **Precision**: ~0.8
- **Recall**: ~0.7

(Small dataset, but demonstrates pipeline correctness)

---

## 📂 File Structure

```bash
mini_rag_app/
│── app.py               # Streamlit frontend
│── retriever.py         # Retrieval + reranking logic
│── rag_pipeline.py      # End-to-end RAG orchestration
│── requirements.txt     # Dependencies
│── .env.example         # Example env variables
│── README.md            # This file
```

---

## 🔧 Technologies Used

- **Embeddings** → OpenAI ADA-002
- **Vector DB** → Supabase (pgvector)
- **Reranker** → Cohere Rerank v2
- **LLM** → GPT-4o-mini (OpenAI)
- **Frontend** → Streamlit (HF Spaces)

---

## 💡 Remarks

### Provider limits:
- Cohere Rerank also has usage caps.

### Tradeoffs:
- Tried to use OpenAI embeddings but hit quota limits.
- Stuck to Streamlit for fast prototyping.

### Next steps:
- Add support for PDFs & multi-file uploads.
- Implement hybrid dense+sparse retrieval (BM25 + embeddings).
- Add caching for cost efficiency.