# ai-doc-cases-no-rag

A working demo of AI document processing **without RAG**.

The AI only sees document metadata (title, tags, description) — it never reads the actual document content.
This repo exists to show the **limitations** of prompting alone.

> Compare this repo side-by-side with `ai-doc-cases-with-rag` to see the difference.

---

## What is this?

This repo contains 3 cases that use LangChain + Ollama to process documents:

| Case | What it does | Limitation without RAG |
|------|-------------|----------------------|
| Case 1 – Search | Finds relevant documents for a query | Can only match titles and tags, misses specific content |
| Case 2 – Classify | Classifies a document into a category | Fails on vague or ambiguous document titles |
| Case 3 – Metadata | Extracts structured metadata | Halluccinates specific dates, rules, and numbers |

---

## Stack

- **Backend**: Python + FastAPI
- **AI**: LangChain + Ollama (`llama3.1`)
- **Frontend**: Plain HTML + JavaScript (no build step)
- **Vector DB**: None (no RAG)

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

---

## Setup

### 1. Install Ollama and pull the model

```bash
# Install Ollama from https://ollama.com
ollama pull llama3.1
```

### 2. Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Start the API

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Open the frontend

Open `frontend/index.html` in your browser (just double-click the file).

---

## Project structure

```
ai-doc-cases-no-rag/
├── backend/
│   ├── main.py                  # FastAPI app, defines all routes
│   ├── requirements.txt
│   ├── cases/
│   │   ├── case1_search.py      # Document search using metadata
│   │   ├── case2_classify.py    # Document classification
│   │   └── case3_metadata.py    # Metadata extraction
│   └── data/
│       └── documents.json       # Document metadata (NO full content)
└── frontend/
    └── index.html               # Simple UI, no build step needed
```

---

## Try these examples

**Case 1 – Search:**
- "how do I return a product?" → should find Return Policy
- "employee holiday days" → should find Internal HR Policy (but might miss details)
- "what is not covered by the warranty?" → AI might guess wrong because it can't read the actual warranty text

**Case 2 – Classify:**
- "Internal HR Policy 2024" → might be misclassified as "Legal Terms" because the title is ambiguous

**Case 3 – Metadata:**
- Try the HR Policy and notice the AI inventing specific dates like "valid_from" that it cannot actually know

---

## Key takeaway

Without RAG, the AI is working blind. It makes educated guesses based on titles and tags,
but fails on anything that requires reading actual document content.

This is fine for simple cases, but breaks down for:
- Specific rules and conditions
- Exact dates and numbers
- Ambiguous or similarly-named documents
