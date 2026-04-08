import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# ---- CONFIG ----
DATA_PATH = "data/documents/txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ---- LOAD FILES ----
def load_documents():
    docs = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".txt"):
            with open(os.path.join(DATA_PATH, filename), "r", encoding="utf-8") as f:
                text = f.read()
                docs.append({
                    "id": filename,
                    "text": text
                })
    return docs

# ---- SPLIT ----
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    for doc in docs:
        split_texts = splitter.split_text(doc["text"])
        for i, chunk in enumerate(split_texts):
            chunks.append({
                "id": f"{doc['id']}_{i}",
                "text": chunk,
                "source": doc["id"]
            })
    return chunks

# ---- EMBEDDING MODEL ----
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- VECTOR DB ----
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="docs")

def store_chunks(chunks):
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]

    embeddings = model.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

# ---- MAIN ----
if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    store_chunks(chunks)

    print(f" Stored {len(chunks)} chunks in vector DB")