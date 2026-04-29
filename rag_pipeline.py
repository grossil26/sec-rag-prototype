import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load Environment Variables
load_dotenv()

def setup_retriever():
    """Sets up our Dense + Sparse Hybrid Retriever"""
    # 1. Dense Retriever (Qdrant)
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="legal_policies",
        path="./local_qdrant_db"
    )
    dense_retriever = qdrant.as_retriever(search_kwargs={"k": 2})

    # 2. Sparse Retriever (BM25)
    loader = TextLoader("data/dummy_policy.txt")
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, separators=["\n\n", "\nSection", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    sparse_retriever = BM25Retriever.from_documents(chunks)
    sparse_retriever.k = 2

    # 3. Hybrid Ensemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.7, 0.3]
    )
    return ensemble_retriever

def main():
    # 1. Initialize the LLM
    print("Initializing Llama 3.1 via Groq...")
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"), 
        model_name="llama-3.1-8b-instant",
        temperature=0.0 # Strict and deterministic for legal answers!
    )

    # 2. Get our Hybrid Retriever
    retriever = setup_retriever()

    # 3. Create the Strict Legal System Prompt (Security Guardrail)
    prompt_template = """
    You are an AI legal and security compliance assistant. 
    Your strict protocol is to answer the user's question using ONLY the provided context below.
    If the context does not contain the answer, you must state: "I do not have the information required to answer this based on the provided policies."
    Do not use outside knowledge. Do not attempt to guess.

    CONTEXT:
    {context}

    USER QUESTION: 
    {question}

    FINAL ANSWER:
    """
    prompt = PromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 4. Build the LangChain RAG Pipeline
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser() # Extracts just the text from the LLM's response object
    )

    # 5. Let's ask our question!
    question = "Who needs to be notified if there is a PII breach, and within what timeframe?"
    
    print(f"\nProcessing User Query: '{question}'\n")
    print("--- SYSTEM RESPONSE ---")
    
    # Run the pipeline
    response = rag_chain.invoke(question)
    print(response)
    print("-----------------------")

    # Let's test the security guardrail with a trick question!
    trick_question = "What is the penalty for a GDPR violation according to the European Union?"
    print(f"\nProcessing Trick Query: '{trick_question}'\n")
    print("--- SYSTEM RESPONSE ---")
    trick_response = rag_chain.invoke(trick_question)
    print(trick_response)
    print("-----------------------")


if __name__ == "__main__":
    main()
