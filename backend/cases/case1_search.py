"""
Case 1: Document Search – WITHOUT RAG
---------------------------------------
The AI reads document metadata (title, tags, description) from a JSON file.
It does NOT read the actual document content.

Flow:
1. Load all document metadata from documents.json
2. Send the user query + metadata to the LLM
3. LLM returns which documents are most relevant and why

Limitation: The AI can only match on weak signals (titles, tags, descriptions).
It will miss nuanced or specific questions that require reading the actual text.
"""

import json
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Load the LLM (make sure Ollama is running with llama3.1)
llm = ChatOllama(model="llama3.1", temperature=0)

# Path to our metadata file
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/documents.json")


def load_documents():
    """Load document metadata from JSON file."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# The prompt tells the AI what it knows and what to do
SEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a document search assistant.
You have access to a list of documents with their titles, tags, and short descriptions.
You do NOT have access to the actual content of the documents.

Based on the user's search query, identify which documents are most likely relevant.
Return a JSON object with this structure:
{{
  "results": [
    {{
      "id": "doc-001",
      "title": "...",
      "relevance_score": 0.9,
      "reason": "This document is relevant because..."
    }}
  ],
  "search_terms_used": ["term1", "term2"],
  "warning": "Note any limitations of this search (e.g. could not verify actual content)"
}}

Only include documents with a relevance score above 0.3.
Sort by relevance score (highest first).
"""),
    ("human", """User query: {query}

Available documents:
{documents}

Return only the JSON object, no extra text.""")
])


async def search_documents(query: str) -> dict:
    """
    Search documents using only metadata.
    Returns a list of relevant documents with relevance scores.
    """
    documents = load_documents()

    # Format document metadata as readable text for the LLM
    doc_text = ""
    for doc in documents:
        doc_text += f"""
ID: {doc['id']}
Title: {doc['title']}
Type: {doc['type']}
Tags: {', '.join(doc['tags'])}
Description: {doc['description']}
---"""

    # Build and send the prompt
    chain = SEARCH_PROMPT | llm
    response = chain.invoke({"query": query, "documents": doc_text})

    # Try to parse the JSON response
    try:
        import re
        # Extract JSON from the response (LLMs sometimes add extra text)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"error": "Could not parse response", "raw": response.content}
    except json.JSONDecodeError:
        result = {"error": "Invalid JSON from LLM", "raw": response.content}

    # Add metadata about this approach for educational purposes
    result["approach"] = "no-rag"
    result["note"] = "Search based on metadata only – actual document content was NOT read"
    return result
