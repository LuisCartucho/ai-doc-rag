"""
ai-doc-cases-no-rag – Backend API
----------------------------------
FastAPI app with 3 case endpoints. NO RAG.

Three ways to use Case 2 and Case 3:

1. Manual input   – send title + description (AI sees weak metadata only)
2. Select a file  – pick a pre-loaded TXT or PDF; full text is sent directly
                    in the prompt (no vector database, no chunking)
3. Upload a file  – same as above, but with your own file

This lets you compare: metadata-only vs full-text-in-prompt vs RAG.

Run with:
    uvicorn main:app --reload --port 8000
"""

import io
import os

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cases.case1_search import search_documents
from cases.case2_classify import classify_document
from cases.case3_metadata import extract_metadata

app = FastAPI(title="AI Doc Cases – No RAG", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCS_TXT_PATH = os.path.join(os.path.dirname(__file__), "data/documents/txt")
DOCS_PDF_PATH = os.path.join(os.path.dirname(__file__), "data/documents/pdf")


# ---------- Request models ----------

class SearchRequest(BaseModel):
    query: str

class ClassifyRequest(BaseModel):
    title: str
    description: str

class ClassifyFileRequest(BaseModel):
    filename: str  # e.g. "return-policy.txt" or "return-policy.pdf"

class MetadataRequest(BaseModel):
    title: str
    description: str

class MetadataFileRequest(BaseModel):
    filename: str


# ---------- Helpers ----------

def read_file_text(filename: str) -> str:
    """
    Read a pre-loaded document file (TXT or PDF) and return its text.
    Looks in data/documents/txt/ for .txt files and data/documents/pdf/ for .pdf files.
    """
    if filename.lower().endswith(".pdf"):
        path = os.path.join(DOCS_PDF_PATH, filename)
        from pypdf import PdfReader
        with open(path, "rb") as f:
            reader = PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        path = os.path.join(DOCS_TXT_PATH, filename)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


async def extract_text_from_upload(file: UploadFile) -> str:
    """Extract text from an uploaded PDF or TXT file."""
    content = await file.read()
    if file.filename.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return content.decode("utf-8", errors="ignore")


def filename_to_title(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    return name.replace("-", " ").replace("_", " ").title()


# ---------- Routes ----------

@app.get("/")
def root():
    return {"message": "AI Doc Cases – No RAG API is running"}


# --- Case 1: Search ---

@app.post("/api/case1/search")
async def case1_search(req: SearchRequest):
    """Search documents using metadata only."""
    return await search_documents(req.query)


# --- Case 2: Classify ---

@app.post("/api/case2/classify")
async def case2_classify(req: ClassifyRequest):
    """
    Classify using title + description only (metadata).
    Weakest signal – AI cannot read the document.
    """
    return await classify_document(req.title, req.description)


@app.post("/api/case2/classify/file")
async def case2_classify_file(req: ClassifyFileRequest):
    """
    Classify a pre-loaded document by filename.
    Reads the full TXT or PDF content and sends it directly in the prompt.
    No vector database – entire text goes into one prompt.
    """
    text = read_file_text(req.filename)
    title = filename_to_title(req.filename)
    result = await classify_document(title, text[:3000])
    result["source_file"] = req.filename
    result["text_length"] = len(text)
    result["note"] = (
        "No RAG: full document text sent directly in the prompt. "
        "No vector search or chunking."
    )
    return result


@app.post("/api/case2/classify/upload")
async def case2_classify_upload(file: UploadFile = File(...)):
    """
    Classify an uploaded PDF or TXT file.
    Full text sent directly in the prompt – no vector database.
    """
    text = await extract_text_from_upload(file)
    title = filename_to_title(file.filename)
    result = await classify_document(title, text[:3000])
    result["uploaded_file"] = file.filename
    result["text_length"] = len(text)
    result["note"] = (
        "No RAG: full document text sent directly in the prompt. "
        "No vector search or chunking."
    )
    return result


# --- Case 3: Metadata ---

@app.post("/api/case3/metadata")
async def case3_metadata(req: MetadataRequest):
    """
    Extract metadata from title + description only.
    High hallucination risk – AI cannot read the document.
    """
    return await extract_metadata(req.title, req.description)


@app.post("/api/case3/metadata/file")
async def case3_metadata_file(req: MetadataFileRequest):
    """
    Extract metadata from a pre-loaded document by filename.
    Reads the full TXT or PDF content and sends it directly in the prompt.
    No vector database – entire text goes into one prompt.
    """
    text = read_file_text(req.filename)
    title = filename_to_title(req.filename)
    result = await extract_metadata(title, text[:3000])
    result["source_file"] = req.filename
    result["text_length"] = len(text)
    result["note"] = (
        "No RAG: full document text sent directly in the prompt. "
        "No vector search or chunking."
    )
    return result


@app.post("/api/case3/metadata/upload")
async def case3_metadata_upload(file: UploadFile = File(...)):
    """
    Extract metadata from an uploaded PDF or TXT file.
    Full text sent directly in the prompt – no vector database.
    """
    text = await extract_text_from_upload(file)
    title = filename_to_title(file.filename)
    result = await extract_metadata(title, text[:3000])
    result["uploaded_file"] = file.filename
    result["text_length"] = len(text)
    result["note"] = (
        "No RAG: full document text sent directly in the prompt. "
        "No vector search or chunking."
    )
    return result
