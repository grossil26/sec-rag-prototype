import os
import json
import re
import streamlit as st
from dotenv import load_dotenv
import networkx as nx

# --- UPDATED IMPORTS FOR OLLAMA ---
from langchain_ollama import ChatOllama 
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
# Replace the failing LangChain imports with this:
from sentence_transformers import CrossEncoder
import numpy as np

# Load Env
load_dotenv()

# ==========================================
# 1. ROLE-BASED ACCESS CONTROL (RBAC) SETUP
# ==========================================
USERS = {
    "user": {"password": "user123", "role": "Standard User"},
    "admin": {"password": "admin123", "role": "Administrator"}
}

def login():
    st.title("🔒 SecRAG - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state["logged_in"] = True
            st.session_state["role"] = USERS[username]["role"]
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid credentials.")

def logout():
    st.session_state.clear()
    st.rerun()

# ==========================================
# 2. CORE RAG & AGENT LOGIC (CACHED)
# ==========================================
@st.cache_resource
def load_models():
    """Loads LLM and Embeddings once to save memory"""
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # MAIN LLM: For Generator and Summarizer
    llm = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "gemma4:31b"),
        temperature=0.0
    )

    # AUDITOR LLM: Strictly locked to JSON output
    llm_json = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "gemma4:31b"),
        temperature=0.0,
        format="json"
    )
    
    # Load the native Cross-Encoder
    cross_encoder = CrossEncoder('BAAI/bge-reranker-base')
    
    return embeddings, llm, llm_json, cross_encoder

def get_hybrid_retriever(embeddings, k_val=5):
    """Connects to Qdrant and builds a dynamically sized Retriever"""
    try:
        # 1. ALWAYS load the Dense (Vector) Retriever from Qdrant
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=embeddings, collection_name="legal_policies", path="./local_qdrant_db"
        )
        dense_retriever = qdrant.as_retriever(search_kwargs={"k": k_val})

        # 2. Try to load Sparse (BM25), but DO NOT crash if the file is missing!
        try:
            loader = TextLoader("data/dummy_policy.txt")
            chunks = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200).split_documents(loader.load())
            sparse_retriever = BM25Retriever.from_documents(chunks)
            sparse_retriever.k = k_val
            return EnsembleRetriever(retrievers=[dense_retriever, sparse_retriever], weights=[0.7, 0.3])
        except Exception as bm25_error:
            # If the text file isn't there, just return the Dense Retriever!
            print(f"BM25 Skipped. Using Pure Vector Search. (Log: {bm25_error})")
            return dense_retriever 

    except Exception as e:
        print(f"Database Error: {e}")
        return None

# ==========================================
# 3. ADMIN INGESTION ENGINE (Now with KG!)
# ==========================================
def process_new_document(text_content, embeddings, llm, llm_json):
    """Runs SAC chunking AND Knowledge Graph Extraction"""
    
    # 1. Generate SAC Summary
    prompt = PromptTemplate.from_template("Summarize this legal document in 3 sentences focusing on scope:\n{text}\nSummary:")
    summary = (prompt | llm).invoke({"text": text_content}).content

    # 2. Extract Knowledge Graph Triplets (Using JSON LLM)
    st.sidebar.write("🕸️ Extracting Knowledge Graph Triplets...")
    kg_prompt = PromptTemplate.from_template(
        "You are a legal data extractor. Extract the 10 most important relationships from the following text.\n"
        "Output ONLY a JSON list of dictionaries, where each dictionary has keys: 'subject', 'predicate', 'object'.\n"
        "Keep them short (1-3 words each).\nTEXT:\n{text}"
    )
    kg_json = (kg_prompt | llm_json).invoke({"text": text_content}).content
    
    # Build the NetworkX Graph
    G = nx.DiGraph()
    try:
        # Regex to safely find the JSON array in the output
        match = re.search(r'\[.*\]', kg_json, re.DOTALL)
        clean_json = match.group(0) if match else kg_json
        triplets = json.loads(clean_json)
        
        for triplet in triplets:
            subj = triplet.get("subject", "").lower()
            pred = triplet.get("predicate", "").lower()
            obj = triplet.get("object", "").lower()
            if subj and obj:
                G.add_edge(subj, obj, relation=pred)
        
        # Save the graph to a local file
        nx.write_graphml(G, "local_kg.graphml")
        st.sidebar.write(f"✅ Extracted {len(G.edges)} relationship rules.")
    except Exception as e:
        print(f"KG Extraction Error: {e}")
        st.sidebar.error("Failed to parse KG Triplets. Proceeding with Vector only.")

    # 3. Standard Chunking & Vector Insertion
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    from langchain_core.documents import Document
    chunks = splitter.split_documents([Document(page_content=text_content)])

    for chunk in chunks:
        chunk.page_content = f"[GLOBAL CONTEXT: {summary}]\n\n[CHUNK CONTENT]:\n{chunk.page_content}"

    QdrantVectorStore.from_documents(
        chunks, embeddings, path="./local_qdrant_db", collection_name="legal_policies"
    )
    return summary

# ==========================================
# 4. MAIN STREAMLIT APPLICATION
# ==========================================
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login()
        return

    st.sidebar.title("System Control")
    st.sidebar.write(f"**User:** {st.session_state['username']}")
    st.sidebar.write(f"**Role:** {st.session_state['role']}")
    st.sidebar.button("Logout", on_click=logout)

    # Unpack the models (Now includes cross_encoder)
    embeddings, llm, llm_json, cross_encoder = load_models()

# --- Admin Only Panel ---
    if st.session_state["role"] == "Administrator":
        st.sidebar.divider()
        st.sidebar.subheader("⚙️ Admin: Policy Ingestion")
        uploaded_file = st.sidebar.file_uploader("Upload Policy (.txt)", type="txt")
        
        if st.sidebar.button("Process & Ingest") and uploaded_file:
            with st.spinner("Chunking & Generating SAC (via Local Cluster)..."):
                text = uploaded_file.read().decode("utf-8")
                
                # ==========================================
                # THE FIX: Save the uploaded file to disk
                # so the BM25 Retriever can find it later!
                # ==========================================
                os.makedirs("data", exist_ok=True) # Ensure the folder exists
                with open("data/dummy_policy.txt", "w", encoding="utf-8") as f:
                    f.write(text)
                
                # Now proceed with ingestion
                summary = process_new_document(text, embeddings, llm, llm_json)
                st.sidebar.success("Database Updated!")
                with st.sidebar.expander("View Generated SAC"):
                    st.write(summary)

    # --- Chat Interface ---
    st.title("Welcome to SecRAG")
    st.caption("Powered by Multi-Agent Hybrid RAG (Gemma4:31b on Nvidia Spark)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the security policies..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            # Start the status box FIRST
            with st.status("Running Sovereign Multi-Agent Pipeline...", expanded=True) as status:

                # ==========================================
                # AGENT 1: THE ROUTER (Adaptive Query Processing)
                # ==========================================
                st.write("🧠 **Router Agent:** Classifying query complexity...")
                router_prompt = PromptTemplate.from_template(
                    "You are an AI routing agent for a legal database.\n"
                    "Analyze this query: '{question}'\n"
                    "Classify its complexity. If it is a basic definition or fact, output 'simple'. "
                    "If it requires deep analysis, multi-step reasoning, or comparisons, output 'complex'.\n"
                    "Output ONLY a JSON object with a single key 'complexity' and string value."
                )
                router_json = (router_prompt | llm_json).invoke({"question": prompt}).content

                # Parse the Router's JSON decision
                try:
                    match = re.search(r'\{.*\}', router_json, re.DOTALL)
                    clean_json = match.group(0) if match else router_json
                    complexity = json.loads(clean_json).get("complexity", "simple").lower()
                except:
                    complexity = "simple" # Default fallback

                # Apply HyPA-RAG Adaptive Parameters
                if "complex" in complexity:
                    k_val = 15
                    top_n = 8
                    st.write(f"📈 **Routing Decision:** [{complexity.upper()}] - Widening search net (k={k_val})...")
                else:
                    k_val = 10
                    top_n = 5
                    st.write(f"📉 **Routing Decision:** [{complexity.upper()}] - Optimizing compute speed (k={k_val})...")

                # NOW we initialize the retriever with the correct k_val!
                retriever = get_hybrid_retriever(embeddings, k_val=k_val)
                if not retriever:
                    status.update(label="System Error: Database Empty", state="error", expanded=True)
                    st.error("Database is empty. Please ask an Admin to upload policies or run the PySpark pipeline.")
                    st.stop()

                # ==========================================
                # RETRIEVAL & RERANKING
                # ==========================================
                st.write("🔍 **Retriever:** Fetching chunks via Hybrid Search...")
                initial_docs = retriever.invoke(prompt)

                st.write("🎯 **Reranker:** Cross-Encoder scoring and filtering chunks...")
                pairs = [[prompt, doc.page_content] for doc in initial_docs]
                scores = cross_encoder.predict(pairs)
                scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)

                # Use the adaptive top_n parameter here!
                best_docs = [doc for score, doc in scored_docs[:top_n]]
                vector_context = "\n\n".join(d.page_content for d in best_docs)

                # ==========================================
                # KNOWLEDGE GRAPH SEARCH (Graph-RAG Fusion)
                # ==========================================
                graph_context = ""
                try:
                    if os.path.exists("local_kg.graphml"):
                        G = nx.read_graphml("local_kg.graphml")
                        found_triplets = []
                        # Simple entity matching: Check if graph nodes are mentioned in the prompt
                        for node in G.nodes():
                            if node in prompt.lower():
                                # Get immediate neighbors of this entity
                                for neighbor in G.neighbors(node):
                                    relation = G[node][neighbor].get('relation', 'is related to')
                                    found_triplets.append(f"- {node.upper()} {relation.upper()} {neighbor.upper()}")

                        if found_triplets:
                            st.write(f"🕸️ **Graph Search:** Found {len(found_triplets)} logical legal linkages!")
                            graph_context = "[EXPLICIT LOGICAL RELATIONSHIPS]:\n" + "\n".join(found_triplets) + "\n\n"
                except Exception as e:
                    print(f"Graph Search Error: {e}")

                # Fuse Vector Context and Graph Context
                context = graph_context + "[LEGAL TEXT CLAUSES]:\n" + vector_context

                # ==========================================
                # AGENT 2 & 3: GENERATOR AND AUDITOR
                # ==========================================
                st.write("✍️ **Generator Agent:** Drafting response based on context...")
                gen_prompt = PromptTemplate.from_template(
                    "You are a legal assistant. Answer the question using ONLY the context.\n"
                    "If the answer is not in the context, say 'I do not know'.\n"
                    "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
                )
                draft = (gen_prompt | llm).invoke({"context": context, "question": prompt}).content

                st.write("🛡️ **Auditor Agent:** Evaluating draft for hallucinations...")
                eval_prompt = PromptTemplate.from_template(
                    "You are an AI Auditor. Assess if the Drafted Answer is 100% faithful to the context.\n"
                    "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nDRAFTED ANSWER:\n{draft}\n\n"
                    "Output ONLY a JSON object with keys: 'is_faithful' (boolean) and 'reasoning' (string)."
                )

                eval_json = (eval_prompt | llm_json).invoke({"context": context, "question": prompt, "draft": draft}).content

                try:
                    match = re.search(r'\{.*\}', eval_json, re.DOTALL)
                    clean_json = match.group(0) if match else eval_json
                    evaluation = json.loads(clean_json)

                    if evaluation.get("is_faithful"):
                        status.update(label="Response Validated & Approved!", state="complete", expanded=False)
                        final_response = draft
                    else:
                        status.update(label="Security Block: Hallucination Detected!", state="error", expanded=True)
                        st.error(f"**Audit Reasoning:** {evaluation.get('reasoning')}")
                        final_response = "I am sorry, but my initial draft failed our internal hallucination audit. I cannot provide a safe answer based on the available policies."
                except Exception as e:
                    print(f"JSON Parsing Error: {e}")
                    final_response = "System Error: Auditor failed to respond in proper JSON format."

            # Print the final response outside the status box!
            st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})

if __name__ == "__main__":
    main()
