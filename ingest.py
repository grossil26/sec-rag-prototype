import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate

# Load Groq Key
load_dotenv()

# 1. Initialize FastEmbed (Runs 100% locally on your VM, no API needed for embeddings!)
print("Loading Local Embedding Model...")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# 2. Initialize Groq LLM for Summarization
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

def generate_document_summary(text):
    """Uses the LLM to generate a global summary of the document for SAC."""
    print("Generating Document-Level Summary (SAC)...")
    prompt = PromptTemplate.from_template(
        "You are an AI assisting with legal document processing.\n"
        "Summarize the following document in under 3 sentences. Identify the title, core purpose, and key scope.\n"
        "Document Text:\n{text}\n\nSummary:"
    )
    chain = prompt | llm
    response = chain.invoke({"text": text})
    return response.content

def process_and_ingest():
    # Load our dummy policy
    loader = TextLoader("data/dummy_policy.txt")
    raw_docs = loader.load()
    full_text = raw_docs[0].page_content
    
    # Generate the global summary
    doc_summary = generate_document_summary(full_text)
    print(f"\n--- Generated Summary ---\n{doc_summary}\n-------------------------\n")
    
    # Chunk the document (Pattern/Recursive Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, # Small chunks for precise retrieval
        chunk_overlap=50,
        separators=["\n\n", "\nSection", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    
    # Implement SAC (Summary-Augmented Chunking)
    for i, chunk in enumerate(chunks):
        # We prepend the global summary to the chunk's text!
        chunk.page_content = f"[GLOBAL CONTEXT: {doc_summary}]\n\n[CHUNK CONTENT]:\n{chunk.page_content}"
        chunk.metadata["chunk_id"] = i
    
    print(f"Prepared {len(chunks)} SAC-augmented chunks. Ingesting into Qdrant...")
    
# 3. Initialize Local Qdrant & Store Data
    # By providing a path instead of a URL, Qdrant runs purely locally on disk!
    qdrant = QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        path="./local_qdrant_db", # Saves database in this folder
        collection_name="legal_policies",
    )
    
    print("Success! Data successfully embedded and stored in local Qdrant database.")

if __name__ == "__main__":
    process_and_ingest()
