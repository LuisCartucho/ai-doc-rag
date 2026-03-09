"""
ai-doc-cases-no-rag – Backend API
----------------------------------
FastAPI app with 3 case endpoints. NO RAG.
The AI only sees document metadata (title, tags, description) – NOT the actual content.

Run with:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our 3 case handlers
from cases.case1_search import search_documents
from cases.case2_classify import classify_document
from cases.case3_metadata import extract_metadata

app = FastAPI(title="AI Doc Cases – No RAG", version="1.0")

# Allow the frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request models ----------
# These define what data each endpoint expects

class SearchRequest(BaseModel):
    query: str  # e.g. "how do I return an item?"

class ClassifyRequest(BaseModel):
    title: str         # document title
    description: str   # short description of the document

class MetadataRequest(BaseModel):
    title: str         # document title
    description: str   # short description of the document


# ---------- Routes ----------

@app.get("/")
def root():
    return {"message": "AI Doc Cases – No RAG API is running"}


@app.post("/api/case1/search")
async def case1_search(req: SearchRequest):
    """
    Case 1: Document Search
    Takes a user query and returns the most relevant documents.
    Uses ONLY metadata (title, tags, description) – no actual document content.
    """
    result = await search_documents(req.query)
    return result


@app.post("/api/case2/classify")
async def case2_classify(req: ClassifyRequest):
    """
    Case 2: Document Classification
    Takes a document title + description and classifies it into a category.
    Uses ONLY the title and description – no actual document content.
    """
    result = await classify_document(req.title, req.description)
    return result


@app.post("/api/case3/metadata")
async def case3_metadata(req: MetadataRequest):
    """
    Case 3: Metadata Extraction
    Takes a document title + description and extracts structured metadata.
    Uses ONLY the title and description – high risk of hallucination.
    """
    result = await extract_metadata(req.title, req.description)
    return result
