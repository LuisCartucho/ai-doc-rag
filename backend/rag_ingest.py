"""
ingest.py – Load documents into ChromaDB
------------------------------------------
This script reads all documents from data/documents/txt/ and data/documents/pdf/,
splits them into chunks, creates embeddings with Ollama,
and stores them in a local ChromaDB vector database.

Document layout:
    data/documents/
    ├── txt/   ← plain text files (easier to work with)
    └── pdf/   ← PDF versions of the same documents (run create_pdfs.py first)

Run ONCE before starting the API (and again whenever you add new documents):
    python ingest.py

You need Ollama running with the embedding model:
    ollama pull nomic-embed-text
"""

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Subdirectories for TXT and PDF documents
BASE_PATH = os.path.join(os.path.dirname(__file__), "data/documents")
TXT_PATH  = os.path.join(BASE_PATH, "txt")
PDF_PATH  = os.path.join(BASE_PATH, "pdf")

# Where ChromaDB will store the vector index (local folder)
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# Ollama embedding model – must be pulled with: ollama pull nomic-embed-text
EMBED_MODEL = "nomic-embed-text"


def load_documents_from_dir(path: str):
    """
    Load all supported documents (.txt and .pdf) from a directory.
    Returns a flat list of LangChain Document objects.
    """
    if not os.path.isdir(path):
        return []
    documents = []
    for fname in sorted(os.listdir(path)):
        fpath = os.path.join(path, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.lower().endswith(".txt"):
            loader = TextLoader(fpath, encoding="utf-8")
            documents.extend(loader.load())
        elif fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(fpath)
            documents.extend(loader.load())
        else:
            print(f"  Skipping unsupported file: {fname}")
    return documents


def ingest():
    print("Loading TXT documents from:", TXT_PATH)
    txt_docs = load_documents_from_dir(TXT_PATH)
    print(f"  {len(txt_docs)} TXT document(s) loaded")

    print("Loading PDF documents from:", PDF_PATH)
    pdf_docs = load_documents_from_dir(PDF_PATH)
    print(f"  {len(pdf_docs)} PDF document(s) loaded")

    documents = txt_docs + pdf_docs
    print(f"Total: {len(documents)} document(s)")

    if not documents:
        print("\nNo documents found. Make sure txt/ has .txt files, and run create_pdfs.py for PDFs.")
        return

    # Split into smaller chunks so the LLM can process them
    # chunk_size = how many characters per chunk
    # chunk_overlap = how many characters to repeat between chunks (for context)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s)")

    # Create embeddings using Ollama (runs locally, no API key needed)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Store in ChromaDB (persisted to disk at CHROMA_PATH)
    print("Creating embeddings and storing in ChromaDB...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"Done! {len(chunks)} chunks stored in: {CHROMA_PATH}")
    print("You can now start the API with: uvicorn main:app --reload --port 8001")


if __name__ == "__main__":
    ingest()