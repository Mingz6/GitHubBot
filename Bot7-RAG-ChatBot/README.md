---
title: ETAP CHAT
emoji: 👁
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
license: apache-2.0
---

# Bot7 - RAG ChatBot (ETAP)

Cloned from [murilofarias10/RAG-ETAP](https://github.com/murilofarias10/RAG-ETAP) — Murilo's project from VAM (Vancouver AI Meetup).

## What it does

Full RAG pipeline for querying ETAP (Electrical Transient Analyzer Program) reports:

1. **`1extract_md_from_pdf_improved.py`** — PDF → Markdown → chunks (LangChain RecursiveCharacterTextSplitter) → embeddings (BGE-large via Together AI)
2. **`2generate_faiss_from_md.py`** — Builds FAISS vector index + BM25 keyword index. Includes Reciprocal Rank Fusion (RRF) and cross-encoder reranking.
3. **`3new_chat.py`** — Flask chat UI. Hybrid retrieval: vector search + BM25 → RRF → cross-encoder rerank → Llama 3.3 70B (Together AI) generates answer.

## Key patterns

- **Hybrid retrieval**: FAISS (semantic) + BM25 (keyword) run in parallel
- **Reciprocal Rank Fusion**: Combines ranked lists with `1/(k+rank)` scoring
- **Cross-encoder reranking**: `ms-marco-MiniLM-L-6-v2` re-scores top 10 after fusion
- **Together AI as OpenAI drop-in**: `openai` SDK with `base_url="https://api.together.xyz/v1"`

## Stack

Flask, PyMuPDF, LangChain, FAISS, BM25 (rank_bm25), sentence-transformers, Together AI (BGE-large embeddings + Llama 3.3 70B chat)
