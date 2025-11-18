# Multimodal Clinical Trial RAG

**Status:** 🔬 **Phase 2 (Querying) - In Progress**

This is an end-to-end multimodal RAG and agentic system for querying complex clinical trial documents. The goal is to build a system that can accurately answer questions about trial protocols by indexing and understanding text, tables, and images.

---

## Project Phases

1.  **Extraction (Complete):** Used the `marker` library to parse PDFs into clean Markdown, tables, and images.
2.  **Ingestion (Complete):** Built a robust, idempotent pipeline (`ingest_data_robust.py`) that:
    * Generates summaries for all text, tables, and images using `gpt-4o-mini`.
    * Uses `@backoff` to handle API rate limits.
    * Indexes summaries into `ChromaDB` and stores original docs in a `MultiVectorRetriever` architecture.
    * Saves progress incrementally, so it can be resumed after a crash.
3.  **Querying (In Progress):** Building the RAG query engine.
    * [ ] Implement Hybrid Search (`BM25` + Semantic)
    * [ ] Add a Cohere `Reranker`
    * [ ] Build the final LLM chain
4.  **Agentic Layer:** Build agents that can perform tasks (e.g., "Compare two trials").
5.  **App:** Build a Streamlit app to demo the final system.

---

## How to Run

1.  Create and activate the environment: `conda create -n clinical-rag python=3.11 -y && conda activate clinical-rag`
2.  Install dependencies: `pip install -r requirements.txt`
3.  Set your API key: `echo 'OPENAI_API_KEY="sk-..."' > .env`
4.  (To-Do: Add data download link or ingestion steps)