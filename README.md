# Sec_RAG-Prototype

## Abstract
This repository contains a prototype Retrieval-Augmented Generation (RAG) system specifically engineered for legal and cybersecurity frameworks (e.g., GDPR, NIS2). Standard RAG systems often fail on legal documents due to highly repetitive boilerplate language and complex formatting. This architecture resolves these issues by implementing advanced techniques from recent AI research.

## Architecture & Research Context
1. **Summary-Augmented Chunking (SAC):** Standard chunking destroys global document context. During ingestion, a Generative LLM creates a concise global summary which is prepended to every single text chunk. This drastically reduces Document-Level Retrieval Mismatch (DRM).
2. **Hybrid Retrieval:** Relies on an `EnsembleRetriever` combining Dense Semantic Search (Qdrant) to understand query intent, and Sparse Keyword Search (BM25) to capture exact legal acronyms.
3. **Multi-Agent Evaluation Loop:** 
    * **Generator Agent:** Drafts an initial response strictly using retrieved context.
    * **Auditor Agent:** An independent LLM-as-a-Judge that evaluates the drafted response. It outputs a programmatic JSON score assessing *Context Relevance* and *Faithfulness* to block hallucinations.
4. **Role-Based Access Control (RBAC):** A Streamlit web interface featuring separate Admin (ingestion) and User (chat) roles.

## Repository Structure
* `app.py`: Main Streamlit application and UI logic.
* `rag_pipeline.py`: Core logic for the Hybrid Retriever and Qdrant integration.
* `ingest.py`: Handles the Summary-Augmented Chunking (SAC) of new documents.
* `evaluation_agent.py`: The AI Auditor that verifies context relevance and blocks hallucinations.
* `retrieve_test.py`: Testing script for the retrieval pipeline.
* `requirements.txt`: Python dependencies for the project.
* `data/`: Directory for storing raw policy `.txt` files (Ignored in Git).
* `local_qdrant_db/`: Local vector database storage (Ignored in Git).

## Setup & Installation

### 1. Prerequisites
* Python 3.10+
* A free [Groq API Key](https://console.groq.com/keys)

### 2. Clone the Repository
	git clone git@git.mif.vu.lt:YOUR_USERNAME/sec-rag-prototype.git
	cd sec-rag-prototype

### 3. Environment Setup
Create an isolated virtual environment and install the required dependencies:

	python3 -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt

### 4. Environment Variables
Create a file named `.env` in the root directory and add your API key:

	GROQ_API_KEY="your_groq_api_key_here"

## Deployment & Usage
Start the Streamlit web server:

	streamlit run app.py

*(If running on a headless VM, create an SSH tunnel to port 8501 to view the UI locally).*

### User Roles:
* **Administrator (admin / admin123):** Access the sidebar to upload raw `.txt` policy files. The system will automatically generate SAC summaries, chunk the text, and inject it into the local Qdrant database.
* **Standard User (user / user123):** Ask legal/compliance questions. The UI will display the real-time execution of the Multi-Agent Evaluation Loop.

## Phase 2 Roadmap
Future upgrades for deployment on university cluster hardware:
* Transition to local `Llama3:70b` hosted via Ollama.
* PySpark integration for distributed PDF ingestion.
* Knowledge Graph (Neo4j) entity extraction for multi-hop legal reasoning.
