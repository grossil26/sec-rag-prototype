# SecRAG: A Sovereign, Multi-Agent Graph-RAG Architecture for Multi-Framework Compliance Validation

## Abstract
This repository contains the finalized Phase 2 architecture for SecRAG, a Retrieval-Augmented Generation (RAG) system engineered for complex legal and cybersecurity frameworks (e.g., GDPR). To address the limitations of standard RAG—such as Document-Level Retrieval Mismatch (DRM), hallucination risks, and data sovereignty concerns—this system utilizes a 100% local, multi-agent architecture powered by Gemma 4 models, PySpark distributed ingestion, and Knowledge Graph reasoning.

## Architecture & Innovations
* **Distributed Ingestion & SAC:** Utilizes Apache PySpark to ingest massive PDF mandates. It employs Summary-Augmented Chunking (SAC), prepending a generative global summary to every chunk to preserve context.
* **Knowledge Graph (KG) Fusion:** During ingestion, the system extracts explicit logical legal linkages (`Subject -> Predicate -> Object`) using `NetworkX`. These triplets are fused with vector retrieval to enable multi-hop reasoning.
* **Adaptive Query Routing:** An initial agent classifies query complexity, dynamically adjusting the retrieval parameter ($k=5$ for simple, $k=15$ for complex) to optimize compute load.
* **Cross-Encoder Hybrid Retrieval:** Fuses Dense Semantic Search (Qdrant) and Sparse Keyword Search (BM25), subsequently passing results through a neural Cross-Encoder (`BAAI/bge-reranker-base`) for rigorous precision filtering.
* **Generator Agent:** Strictly bound to context answer generator.
* **Auditor Agent:** An independent LLM-as-a-Judge that evaluates drafts via JSON-enforced logic to block hallucinations.

## Repository Structure
* `app.py`: The core Streamlit application housing the Multi-Agent Evaluation Loop, Router, and Hybrid Retriever.
* `spark_ingest.py`: The PySpark ETL pipeline for distributed bulk-PDF ingestion, chunking, and Qdrant population.
* `ragas_eval.py`: Autonomous "LLM-as-a-Judge" semantic benchmarking script (Context Relevance & Faithfulness).
* `requirements.txt`: Python dependencies.
* `data/`: Directory for raw PDFs and text policies (Ignored in Git).
* `local_qdrant_db/`: Local vector database storage (Ignored in Git).
* `local_kg.graphml`: Exported Knowledge Graph structure (Ignored in Git).

## Setup & Installation

### 1. Prerequisites
* Python 3.10+
* Java 8 or 11 (Required for Apache PySpark)
* Access to a local Ollama cluster (e.g., University supercomputing node) running `gemma4:31b`.

### 2. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory to route inference to your local cluster (No external API keys required):
```env
OLLAMA_BASE_URL="http://localhost:xxxxx"
OLLAMA_MODEL="gemma4:31b"
```

## Deployment & Usage

### 1. Bulk Data Ingestion (Backend)
To process heavy PDF regulatory frameworks into the vector and graph databases:
```bash
python spark_ingest.py
```

### 2. Multi-Agent UI (Frontend)
Start the Streamlit web server:
```bash
streamlit run app.py
```
* **Administrator** (`admin` / `admin123`): Access the sidebar to process short `.txt` policies and extract Graph Triplets on the fly.
* **Standard User** (`user` / `user123`): Submit compliance queries and view the real-time execution of the Router, Reranker, Graph Search, and Auditor verification.

### 3. Autonomous Evaluation (Benchmarking)
Run the internal RAGAS-Lite evaluator to scientifically grade the system's accuracy and hallucination resistance:
```bash
python ragas_eval.py
```
