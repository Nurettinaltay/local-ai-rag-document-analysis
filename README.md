# Local AI RAG Document Analysis

A privacy-first local RAG application for analyzing PDF documents and scanned images.

The application runs fully locally and combines document processing, vector search, hybrid retrieval, reranking and local language models without using cloud APIs.

## Key Features

- PDF document ingestion and chunking
- Local embeddings with Ollama `nomic-embed-text`
- Multilingual embeddings with `paraphrase-multilingual-MiniLM-L12-v2`
- PostgreSQL + pgvector vector storage
- Semantic search
- Hybrid search with PostgreSQL full-text search
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- HNSW, GIN and B-tree indexing
- Query embedding caching
- SHA-256 file deduplication
- Local document Q&A with Qwen
- Scanned-document Q&A with Donut
- Prompt injection detection and control-token neutralization
- Structured JSON output for downstream use
- Retrieval benchmarking with Hit-rate@k, MRR@k and page coverage
- Fully local and privacy-first architecture

## Tech Stack

- Python
- Streamlit
- PostgreSQL
- pgvector
- Ollama
- Qwen
- Nomic Embeddings
- Sentence Transformers
- Hugging Face Transformers
- Donut
- psycopg3

## Architecture

```text
PDF / scanned image
        |
        v
Document processing
        |
        v
Chunking
        |
        v
Embeddings
  |             |
Nomic         MiniLM
  |             |
  +------ PostgreSQL / pgvector
                    |
                    v
        Semantic / Hybrid Search
                    |
                    v
                 Reranking
                    |
                    v
                  Qwen
                    |
                    v
          Dutch grounded answer
Retrieval

The application supports multiple retrieval strategies:

Semantic search using vector similarity
Keyword search using PostgreSQL full-text search
Hybrid search using Reciprocal Rank Fusion
Cross-encoder reranking for improved relevance
Structured Output

The application can also generate schema-constrained JSON output.

This allows AI results to be processed further in systems such as:

AI
 |
JSON
 |
Python
 |
Database
 |
Power BI
Benchmarking

Retrieval strategies can be evaluated using:

Hit-rate@k
MRR@k
Page coverage
Search duration
Privacy

The application is designed to run locally.

No cloud APIs are required for:

embeddings
language model inference
vector search
document storage

This makes the project suitable for privacy-sensitive document analysis.

Language

The application interface and generated answers are in Dutch.

Project Status

Portfolio / learning project focused on local AI, RAG, retrieval quality and document analysis.
