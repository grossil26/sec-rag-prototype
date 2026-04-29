from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def test_hybrid_retrieval():
    # 1. Initialize our Local Embedding Model
    print("Loading Local Embedding Model...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # 2. Connect to the existing Local Qdrant Database (Dense Semantic Retriever)
    print("Connecting to Local Qdrant Database...")
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="legal_policies",
        path="./local_qdrant_db"
    )
    # Configure it to return the top 2 semantic matches
    dense_retriever = qdrant.as_retriever(search_kwargs={"k": 2})

    # 3. Create the BM25 Sparse Retriever (Exact Keyword Retriever)
    # We briefly load the chunks into memory to build the keyword index
    loader = TextLoader("data/dummy_policy.txt")
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, separators=["\n\n", "\nSection", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    
    sparse_retriever = BM25Retriever.from_documents(chunks)
    sparse_retriever.k = 2 # Configure it to return top 2 keyword matches

    # 4. Build the Hybrid Ensemble Retriever
    print("Building Hybrid Retriever (Dense + Sparse)...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.7, 0.3] # 70% weight to Semantic meaning, 30% to Exact Keywords
    )

    # 5. Let's test it with a tricky question!
    query = "Who needs to be notified if there is a PII breach, and within what timeframe?"
    print(f"\n==============================================")
    print(f"🔎 SEARCHING FOR: '{query}'")
    print(f"==============================================\n")

    # This invokes both retrievers, merges the results, and deduplicates them!
    results = ensemble_retriever.invoke(query)

    print(f"--- Top Retrieved Chunks ({len(results)} found) ---")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:\n{doc.page_content}")
        print("-" * 40)

if __name__ == "__main__":
    test_hybrid_retrieval()
