import os
import json
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("🚀 Initializing Sovereign RAGAS-Lite Evaluator...")

# 1. Load Models
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
cross_encoder = CrossEncoder('BAAI/bge-reranker-base')
llm_json = ChatOllama(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    model=os.getenv("OLLAMA_MODEL", "gemma4:31b"),
    temperature=0.0,
    format="json"
)
llm_gen = ChatOllama(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    model=os.getenv("OLLAMA_MODEL", "gemma4:31b"),
    temperature=0.0
)

# 2. Retriever Setup
def get_retriever():
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings, collection_name="legal_policies", path="./local_qdrant_db"
    )
    dense = qdrant.as_retriever(search_kwargs={"k": 10})
    try:
        loader = TextLoader("data/dummy_policy.txt")
        chunks = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200).split_documents(loader.load())
        sparse = BM25Retriever.from_documents(chunks)
        sparse.k = 10
        return EnsembleRetriever(retrievers=[dense, sparse], weights=[0.7, 0.3])
    except:
        return dense

retriever = get_retriever()

# 3. Synthetic Data Generation
print("🤖 Agent: Reading document and generating 3 synthetic test cases...")
try:
    with open("data/dummy_policy.txt", "r", encoding="utf-8") as f:
        doc_text = f.read()
except:
    print("❌ Error: data/dummy_policy.txt not found.")
    exit()

gen_prompt = PromptTemplate.from_template(
    "Read the text and generate 3 legal questions that can be answered using this text.\n"
    "Output ONLY a JSON list of dictionaries with the key: 'query'.\n"
    "TEXT:\n{text}"
)
dataset_json = (gen_prompt | llm_json).invoke({"text": doc_text}).content

try:
    match = re.search(r'\[.*\]', dataset_json, re.DOTALL)
    test_cases = json.loads(match.group(0))
except:
    print("❌ Failed to parse synthetic dataset.")
    exit()

# 4. LLM-as-a-Judge Evaluation Prompts
relevance_prompt = PromptTemplate.from_template(
    "Given the QUESTION and the RETRIEVED CONTEXT, does the context contain the information needed to answer the question?\n"
    "Output ONLY a JSON object with a boolean key 'is_relevant'.\n"
    "QUESTION: {question}\nCONTEXT: {context}"
)

faithfulness_prompt = PromptTemplate.from_template(
    "Given the CONTEXT and the AI's ANSWER, is the answer entirely based on the context without any outside hallucination?\n"
    "Output ONLY a JSON object with a boolean key 'is_faithful'.\n"
    "CONTEXT: {context}\nANSWER: {answer}"
)

gen_answer_prompt = PromptTemplate.from_template(
    "Answer the question using ONLY the context. If you don't know, say 'I do not know'.\n"
    "CONTEXT: {context}\nQUESTION: {question}\nANSWER:"
)

# 5. Run Evaluation
print(f"\n📊 Running RAGAS Metrics on {len(test_cases)} queries...\n")
score_relevance = 0
score_faithfulness = 0

for i, test in enumerate(test_cases):
    query = test["query"]
    print(f"Testing Query {i+1}: '{query}'")
    
    # Retrieval & Reranking
    initial_docs = retriever.invoke(query)
    pairs = [[query, doc.page_content] for doc in initial_docs]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
    best_docs = [doc for score, doc in scored_docs[:3]]
    context = "\n\n".join([d.page_content for d in best_docs])
    
    # Generate Answer
    answer = (gen_answer_prompt | llm_gen).invoke({"context": context, "question": query}).content
    
    # Judge 1: Context Relevance
    rel_json = (relevance_prompt | llm_json).invoke({"question": query, "context": context}).content
    try:
        is_relevant = json.loads(re.search(r'\{.*\}', rel_json, re.DOTALL).group(0)).get("is_relevant", False)
    except:
        is_relevant = False
    
    # Judge 2: Faithfulness
    faith_json = (faithfulness_prompt | llm_json).invoke({"context": context, "answer": answer}).content
    try:
        is_faithful = json.loads(re.search(r'\{.*\}', faith_json, re.DOTALL).group(0)).get("is_faithful", False)
    except:
        is_faithful = False
        
    if is_relevant: score_relevance += 1
    if is_faithful: score_faithfulness += 1

print("\n" + "="*50)
print("🏆 FINAL RAGAS-LITE BENCHMARK SCORES")
print("="*50)
print(f"Context Relevance (Retrieval Accuracy): {(score_relevance/len(test_cases))*100:.1f}%")
print(f"Faithfulness (Hallucination Free):      {(score_faithfulness/len(test_cases))*100:.1f}%")
print("="*50)
