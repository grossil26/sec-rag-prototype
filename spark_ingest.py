import os
from pyspark.sql import SparkSession
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# 1. Initialize Spark Session (Local Mode for Prototype, Cluster Mode for Production)
print("🚀 Initializing PySpark Session...")
spark = SparkSession.builder \
    .appName("Legal_RAG_Distributed_Ingestion") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

def extract_pdf_text(file_path):
    """Worker function to load and extract text from a single PDF"""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # Combine all pages into one massive string for the document
        full_text = "\n".join([doc.page_content for doc in docs])
        return (file_path, full_text)
    except Exception as e:
        return (file_path, f"Error: {e}")

def main():
    pdf_dir = "data/pdfs"
    if not os.path.exists(pdf_dir):
        print(f"❌ Directory {pdf_dir} does not exist. Please create it and add PDFs.")
        return

    # 2. Gather all PDF files
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("⚠️ No PDFs found in data/pdfs. Exiting.")
        return

    print(f"📚 Found {len(pdf_files)} PDF(s) for distributed processing.")

    # 3. Distribute the PDF extraction across Spark RDDs
    # In a real cluster, this sends different PDFs to different server nodes!
    rdd = spark.sparkContext.parallelize(pdf_files)
    extracted_rdd = rdd.map(extract_pdf_text)
    
    # Collect the processed text back to the driver
    processed_docs = extracted_rdd.collect()

    # 4. Initialize Local Embedding Model
    print("🧠 Loading Embedding Model...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    # 5. Chunk and Inject into Qdrant
    total_chunks = 0
    for file_path, text in processed_docs:
        if text.startswith("Error:"):
            print(f"❌ Failed to process {file_path}: {text}")
            continue
            
        print(f"✂️ Chunking {os.path.basename(file_path)}...")
        from langchain_core.documents import Document
        chunks = splitter.split_documents([Document(page_content=text, metadata={"source": file_path})])
        
        print(f"💾 Saving {len(chunks)} chunks to Qdrant Vector Database...")
        QdrantVectorStore.from_documents(
            chunks, embeddings, path="./local_qdrant_db", collection_name="legal_policies"
        )
        total_chunks += len(chunks)

    print(f"✅ PySpark Ingestion Complete! Successfully indexed {total_chunks} legal chunks.")
    spark.stop()

if __name__ == "__main__":
    main()
