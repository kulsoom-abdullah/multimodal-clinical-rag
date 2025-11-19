
# Agentic Multimodal Clinical Trial RAG

![Status](https://img.shields.io/badge/Status-Phase_3:_Advanced_Architecture-green) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![Docker](https://img.shields.io/badge/Docker-Ready-blue)

**Current Status:** 🔬 **Phase 3: Advanced Architecture (Metadata-Driven Agent)**

This is an end-to-end **Agentic RAG system** designed to answer complex questions about clinical trial protocols with high precision. Unlike standard RAG demos, this project solves the specific challenges of **"Filter-then-Search"** workflows in the medical domain by combining structured metadata extraction (via **Claude Sonnet 4.5**) with a Dynamic Router and Semantic Retrieval.

---

## 💡 Why RAG? (The Business Case)

Why build a complex RAG system instead of fine-tuning a model or using a long-context window?

1.  **Data Velocity:** Clinical trials change constantly (amendments, new phases). A pre-trained model is frozen in time. RAG allows us to ingest a new protocol PDF today and query it immediately without expensive re-training.
2.  **Hallucinations vs. Grounding:** In the medical domain, "guessing" is unacceptable. This system forces the LLM to answer *only* based on retrieved evidence, providing strict citations (e.g., `[Source: Prot_000/SAP]`) for auditability.
3.  **The "Needle in the Haystack":** Clinical protocols are dense, 200+ page documents. Simple vector search often retrieves irrelevant sections. By using an **Agentic Router**, we distinguish between looking up a specific ID (Keyword Search) and understanding a mechanism (Semantic Search).
4.  **Cost Efficiency:** Fine-tuning teaches "style," not "facts." RAG is the correct architectural pattern for knowledge-intensive tasks, costing a fraction of fine-tuning.

---

## 📊 Performance Benchmarks

I built a custom evaluation pipeline (`scripts/evaluate_retrieval.py`) to benchmark the system against a "Golden Dataset" of tricky clinical queries.

| Query Type | Challenge | Logic Used | Recall@3 | Latency |
| :--- | :--- | :--- | :--- | :--- |
| **ID Lookup** | *"Objective of 3000-02-005"* | **Strict Mode** (Hard Filter) | **100%** | 1.08s |
| **Concept** | *"Side effects of PARP..."* | **Hybrid** (Semantic + BM25) | **100%** | 1.01s |
| **Filtering** | *"Phase 3 trials for NSCLC"* | **Metadata Filter** | **100%** | 0.85s |

> **Note on Latency:** First-token latency includes a one-time model loading cost (Cold Start) of ~8-10s. Subsequent queries utilize cached resources in RAM, executing in <1.5s.

---

## 📖 The Engineering Journey

### 1. The Problem: "The Semantic Noise Trap"
My initial MVP used a standard RAG pipeline (Semantic Search with ChromaDB). While it worked for general questions, it failed catastrophically on specific clinical queries:
* **Query:** *"What is the primary objective of study 3000-02-005?"*
* **Failure:** The retriever returned "primary objective" sections from *other* trials (semantic matches) but missed the specific protocol ID because it was buried in the text.
* **Diagnosis:** "BM25 Leaks" — high-frequency keywords drowned out the specific ID.

### 2. The Pivot: Agentic Architecture
I re-architected the pipeline into an **Agentic System**:
* **Smart Ingestion:** Built an ingestion agent (`ingest_data_advanced.py`) using **Claude Sonnet 4.5** to extract 7 key metadata fields (Phase, Indication, Drug) into a structured schema before chunking.
* **Dynamic Routing Agent:** Implemented a Regex-based Router in the query engine to detect intent.
    * **If ID is detected:** System enters **Strict Mode** (Disables BM25, enforces Hard Metadata Filters).
    * **If Concept is detected:** System enters **Hybrid Mode** (70% Semantic / 30% Keyword).
* **Precision Reranking:** Added a `CrossEncoder` (MS-MARCO) step to re-score the final results.

---

## 🏗️ System Architecture

### Phase 1: Intelligent Extraction (The "Generator")
* **Tool:** `marker-pdf`
* **Logic:** Converts complex, column-heavy clinical PDFs into clean Markdown and extracts figures/tables as separate artifacts.

### Phase 2: Advanced Ingestion (The "Brain")
* **Script:** `scripts/ingest_data_advanced.py`
* **Model:** `claude-sonnet-4-5-20250929` (Metadata Extraction) + `gpt-5-mini` (Summarization).
* **Embedding:** `NeuML/pubmedbert-base-embeddings` (Domain-specific biomedical embeddings).
* **Robustness:** Uses `@backoff` decorators and Idempotent Checkpointing to resume long jobs.

### Phase 3: Agentic Retrieval (The "Router")
* **Script:** `scripts/query_rag.py` / `app_v2.py`
* **Router:** Dynamic Router detects ID-heavy queries vs. concept queries.
* **Retriever:** Hybrid Ensemble (`BM25` + `Chroma`) with dynamic weighting.
* **Multimodal:** The UI renders retrieved images (tables/figures) directly in the chat stream.

---

## 💰 Resource & Cost Analysis

### 1. Compute Infrastructure (RunPod)
* **Hardware:** NVIDIA RTX 3090 (24GB VRAM).
* **Cost:** **~$0.22 / hour**.
* **Performance:** `PubMedBERT` encodes ~100 chunks/second on GPU.

### 2. The "Hybrid Brain" Strategy
To optimize costs, tasks are assigned to the most appropriate model tier:
* **Reasoning (Claude Sonnet 4.5):** Used *only* for high-value metadata extraction (Superior reasoning for context).
* **Scale (GPT-5-mini):** Used for summarizing thousands of text chunks (Cost-efficiency for high volume).
* **Total Ingestion Cost:** Processed 7 full trials (4,800+ requests) for **~$6.50**.

---

## 📂 Repository Structure

```bash
├── app_v2.py                     # Streamlit UI with Router Visualization
├── Dockerfile                    # Production-ready container config
├── scripts/
│   ├── ingest_data_advanced.py   # The "Brain": Metadata extraction & Indexing
│   ├── query_rag.py              # The "Logic": Dynamic Router & RAG Chain
│   └── evaluate_retrieval.py     # The "Proof": Calculating Recall@K
├── data/                         # (Ignored) ChromaDB & Docstore
└── requirements.txt              # Dependencies
````

-----

## 🚀 How to Run

### Option A: Docker (Recommended)

The easiest way to run the full system with all dependencies.

```bash
# 1. Build the image
docker build -t clinical-rag .

# 2. Run the container (Exposes port 8501)
docker run -p 8501:8501 clinical-rag
```

*Navigate to `http://localhost:8501`*

### Option B: Local Development

```bash
# 1. Create environment
conda create -n clinical-rag python=3.11 -y 
conda activate clinical-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the App
streamlit run app_v2.py
```

```
```