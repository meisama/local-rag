# Simple Local RAG: Offline PDF Chat on GPU

## Overview
A lightweight Retrieval-Augmented Generation (RAG) system for querying local PDFs using open-source LLMs on NVIDIA GPUs. Ideal for secure, offline scientific doc analysis (e.g., research papers in HPC workflows).

## Features
- Ingests PDFs into a local vector store (FAISS).
- GPU-accelerated embeddings and inference (via HuggingFace/CuPy).
- Simple CLI for "chat with your PDF."

## Architecture
- Ingestion: PDF → Text → Embeddings (SentenceTransformer).
- Retrieval: FAISS index on GPU.
- Generation: Local LLM (e.g., Llama via Ollama).

## Installation
```bash
git clone https://github.com/meisama/local-rag.git
cd local-rag
pip install -r requirements.txt
```

## License
MIT
