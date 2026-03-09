"""
Case 3: Metadata Extraction – WITHOUT RAG
------------------------------------------
The AI extracts structured metadata from a document title and short description.
It does NOT read the actual document content.

Flow:
1. Receive document title + description
2. LLM tries to extract/guess metadata from those weak signals
3. Returns a structured JSON with metadata fields

Limitation: The AI will HALLUCINATE specific details like exact dates, policy numbers,
version numbers, or specific rules, because it cannot actually read the document.
This is a great example of where RAG is necessary.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json
import re

llm = ChatOllama(model="llama3.1", temperature=0)

METADATA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a metadata extraction assistant.
Your job is to extract structured metadata from a document based only on its title and description.
You do NOT have access to the full document content.

Extract and return a JSON object with these fields:
{{
  "document_type": "e.g. policy, warranty, legal, internal",
  "topics": ["list", "of", "main", "topics"],
  "language": "en or other language code",
  "valid_from": "YYYY-MM-DD or null if unknown",
  "valid_to": "YYYY-MM-DD or null if unknown",
  "department": "e.g. HR, Legal, Customer Service, or null",
  "is_confidential": true or false,
  "target_audience": "e.g. customers, employees, all",
  "key_rules": ["list of key rules or policies mentioned, if any"]
}}

If you are not sure about a field, use null. Do NOT invent specific dates or rules.
"""),
    ("human", """Document title: {title}
Description: {description}

Return only the JSON object, no extra text.""")
])


async def extract_metadata(title: str, description: str) -> dict:
    """
    Extract metadata using only title and description.
    WARNING: High hallucination risk for specific details.
    """
    chain = METADATA_PROMPT | llm
    response = chain.invoke({"title": title, "description": description})

    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"error": "Could not parse response", "raw": response.content}
    except json.JSONDecodeError:
        result = {"error": "Invalid JSON from LLM", "raw": response.content}

    result["approach"] = "no-rag"
    result["warning"] = "Metadata extracted from title/description only – specific dates and rules may be hallucinated"
    return result
