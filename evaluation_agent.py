import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Load Environment Variables
load_dotenv()

# --- REUSABLE PIPELINE SETUP ---
def setup_retriever():
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings, collection_name="legal_policies", path="./local_qdrant_db"
    )
    dense_retriever = qdrant.as_retriever(search_kwargs={"k": 2})

    loader = TextLoader("data/dummy_policy.txt")
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=["\n\n", "\nSection", "\n", " ", ""])
    chunks = text_splitter.split_documents(raw_docs)
    
    sparse_retriever = BM25Retriever.from_documents(chunks)
    sparse_retriever.k = 2

    return EnsembleRetriever(retrievers=[dense_retriever, sparse_retriever], weights=[0.7, 0.3])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- THE NEW MULTI-AGENT LOGIC ---
def main():
    # We can use the same Llama 3.1 model to act as both agents
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant", temperature=0.0)
    retriever = setup_retriever()

    # 1. GENERATOR PROMPT
    generator_prompt = PromptTemplate.from_template(
        "You are a legal assistant. Answer the question using ONLY the context.\n"
        "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )

    # 2. AUDITOR (EVALUATOR) PROMPT
    # We ask the LLM to output pure JSON so our Python code can parse the score!
    auditor_prompt = PromptTemplate.from_template(
        "You are an expert Legal AI Auditor. Your job is to evaluate a drafted answer.\n"
        "1. Read the Context.\n"
        "2. Read the Question.\n"
        "3. Read the Drafted Answer.\n"
        "Assess if the Drafted Answer is 100% faithful to the context (no outside knowledge added).\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n\n"
        "DRAFTED ANSWER:\n{draft}\n\n"
        "Output ONLY a JSON object with two keys: 'is_faithful' (boolean true/false) and 'reasoning' (string).\n"
        "JSON:"
    )

    question = "Who needs to be notified if there is a PII breach, and within what timeframe?"
    
    print("\n[System] Fetching Context via Hybrid Search...")
    retrieved_docs = retriever.invoke(question)
    context_text = format_docs(retrieved_docs)

    print("[Generator Agent] Drafting initial response...")
    gen_chain = generator_prompt | llm
    draft_response = gen_chain.invoke({"context": context_text, "question": question}).content

    print(f"\n--- DRAFTED RESPONSE ---\n{draft_response}\n------------------------\n")

    print("[Auditor Agent] Evaluating draft for hallucinations...")
    eval_chain = auditor_prompt | llm
    
    # We force the LLM to use JSON mode (Supported by Groq & Llama 3.1)
    eval_response = eval_chain.invoke(
        {"context": context_text, "question": question, "draft": draft_response}
    ).content

    try:
        # Parse the JSON response from the Auditor
        evaluation = json.loads(eval_response)
        print("\n--- AUDIT RESULTS ---")
        print(f"Faithful to Context: {evaluation.get('is_faithful')}")
        print(f"Reasoning: {evaluation.get('reasoning')}")
        print("---------------------")

        # Final Gateway Logic
        if evaluation.get('is_faithful'):
            print("\n✅ AUDIT PASSED: Delivering response to user.")
        else:
            print("\n❌ AUDIT FAILED: The AI hallucinated or used outside knowledge. Blocking response.")
            
    except Exception as e:
        print(f"Failed to parse evaluation JSON: {eval_response}")

if __name__ == "__main__":
    main()
