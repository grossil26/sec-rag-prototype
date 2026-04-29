import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Load Env
load_dotenv()

# ==========================================
# 1. ROLE-BASED ACCESS CONTROL (RBAC) SETUP
# ==========================================
# Hardcoded credentials for the prototype
USERS = {
    "user": {"password": "user123", "role": "Standard User"},
    "admin": {"password": "admin123", "role": "Administrator"}
}

def login():
    st.title("🔒 Legal RAG System - Login")
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
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant", temperature=0.0)
    return embeddings, llm

def get_hybrid_retriever(embeddings):
    """Connects to Qdrant and builds the Hybrid Retriever"""
    # Note: For BM25 to work dynamically in a real app, we'd load chunks from a JSON. 
    # For this prototype, we reload the local dummy text.
    try:
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=embeddings, collection_name="legal_policies", path="./local_qdrant_db"
        )
        dense_retriever = qdrant.as_retriever(search_kwargs={"k": 2})
        
        loader = TextLoader("data/dummy_policy.txt")
        chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(loader.load())
        sparse_retriever = BM25Retriever.from_documents(chunks)
        sparse_retriever.k = 2
        
        return EnsembleRetriever(retrievers=[dense_retriever, sparse_retriever], weights=[0.7, 0.3])
    except Exception as e:
        return None # Database might not exist yet

# ==========================================
# 3. ADMIN INGESTION ENGINE
# ==========================================
def process_new_document(text_content, embeddings, llm):
    """Runs the SAC chunking and saves to database"""
    # 1. Generate Summary (SAC)
    prompt = PromptTemplate.from_template("Summarize this legal document in 3 sentences focusing on scope:\n{text}\nSummary:")
    summary = (prompt | llm).invoke({"text": text_content}).content
    
    # 2. Chunk & Inject Summary
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    # Using a dummy document object for the text
    from langchain_core.documents import Document
    chunks = splitter.split_documents([Document(page_content=text_content)])
    
    for chunk in chunks:
        chunk.page_content = f"[GLOBAL CONTEXT: {summary}]\n\n[CHUNK CONTENT]:\n{chunk.page_content}"
    
    # 3. Save to Qdrant
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

    # --- Sidebar UI ---
    st.sidebar.title("System Control")
    st.sidebar.write(f"**User:** {st.session_state['username']}")
    st.sidebar.write(f"**Role:** {st.session_state['role']}")
    st.sidebar.button("Logout", on_click=logout)

    embeddings, llm = load_models()

    # --- Admin Only Panel ---
    if st.session_state["role"] == "Administrator":
        st.sidebar.divider()
        st.sidebar.subheader("⚙️ Admin: Policy Ingestion")
        uploaded_file = st.sidebar.file_uploader("Upload Policy (.txt)", type="txt")
        if st.sidebar.button("Process & Ingest") and uploaded_file:
            with st.spinner("Chunking & Generating SAC..."):
                text = uploaded_file.read().decode("utf-8")
                summary = process_new_document(text, embeddings, llm)
                st.sidebar.success("Database Updated!")
                with st.sidebar.expander("View Generated SAC"):
                    st.write(summary)

    # --- Chat Interface ---
    st.title("⚖️ AI Legal & Compliance Assistant")
    st.caption("Powered by Multi-Agent RAG (Llama 3.1 & Qdrant)")

    # Chat history state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    if prompt := st.chat_input("Ask a question about the security policies..."):
        # 1. Display User Message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Process AI Response
        with st.chat_message("assistant"):
            retriever = get_hybrid_retriever(embeddings)
            
            if not retriever:
                st.error("Database is empty. Ask an admin to upload policies.")
                return

            with st.status("Running Multi-Agent Pipeline...", expanded=True) as status:
                st.write("🔍 **Retriever:** Fetching chunks via Hybrid Search...")
                docs = retriever.invoke(prompt)
                context = "\n\n".join(d.page_content for d in docs)
                
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
                eval_json = (eval_prompt | llm).invoke({"context": context, "question": prompt, "draft": draft}).content
                
                try:
                    # Clean the LLM output: extract ONLY the dictionary using Regex
                    import re
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
                    # We print the actual error to the terminal to help us debug!
                    print(f"JSON Parsing Error: {e}") 
                    print(f"Raw LLM Output was: {eval_json}")
                    final_response = "System Error: Auditor failed to respond in proper JSON format."

            st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})

if __name__ == "__main__":
    main()
