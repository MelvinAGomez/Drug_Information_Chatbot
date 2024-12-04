import os
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Directory containing JSON files
DATA_DIR = "./datasets/microlabs_usa"
PERSIST_DIR = "./db4"  # Persistent store for ChromaDB

# Batch size for processing documents
BATCH_SIZE = 50
TOP_N = 10

# Read JSON files and convert to LangChain documents
def load_top_n_json_to_documents(data_dir, top_n):
    docs = []
    # Get all JSON files and sort them alphabetically
    json_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")])[:top_n]
    
    for file_name in json_files:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, "r") as f:
            json_data = json.load(f)
            drug_name = file_name.replace(".json", "")  # Extract drug name from filename
            
            # Iterate over the keys in the JSON
            for key, value in json_data.items():
                # Combine drug name, key name, and content
                chunk_content = f"Drug Name: {drug_name}\nKey: {key}\nContent:\n{value}"
                metadata = {"source": file_name, "drug_name": drug_name, "key": key}
                docs.append(Document(page_content=chunk_content, metadata=metadata))
    return docs

# Load top N JSON files as documents
docs_list = load_top_n_json_to_documents(DATA_DIR, TOP_N)


# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800, chunk_overlap=150
)

# Split documents into smaller chunks
all_chunks = text_splitter.split_documents(docs_list)

# Batch processing: split all chunks into batches of BATCH_SIZE
def chunk_batches(chunks, batch_size):
    for i in range(0, len(chunks), batch_size):
        yield chunks[i:i + batch_size]

# Initialize Ollama embeddings
embd = OllamaEmbeddings(model="nomic-embed-text")  # Specify the embedding model

# Process each batch of 100 documents

for batch_num, batch in enumerate(chunk_batches(all_chunks, BATCH_SIZE)):
    print(f"Processing batch {batch_num + 1}")
    if os.path.exists(PERSIST_DIR):
        # Load existing database
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embd)
    else:
        # Create a new database for the first batch'ArithmeticError
        print('bleh')
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embd)

    # Add batch to the vectorstore
    vectorstore.add_documents(batch)
    vectorstore.persist()
    print(f"Batch {batch_num + 1} persisted successfully.")


