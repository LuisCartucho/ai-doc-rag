"""
Case 2: Document Classification – WITHOUT RAG
-----------------------------------------------
The AI classifies a document into a category based only on its title and description.
It does NOT read the actual document content.

Flow:
1. Receive document title + description
2. Send to LLM with a classification prompt
3. LLM guesses the category from weak signals

Limitation: When documents have vague titles or ambiguous descriptions,
the AI will make incorrect classifications. This is especially visible with
the internal HR policy (could be confused with legal or compliance docs).
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json
import re

llm = ChatOllama(model="llama3.1", temperature=0)

# These are our 5 defined categories
CATEGORIES = [
    "Product Information",
    "Delivery",
    "Returns & Complaints",
    "Legal Terms",
    "Internal Documentation",
]

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a document classification assistant.
Your job is to classify documents into exactly one of these categories:
- Product Information
- Delivery
- Returns & Complaints
- Legal Terms
- Internal Documentation

You only receive the document title and a short description.
You do NOT have access to the full document content.

Return a JSON object with this structure:
{{
  "category": "one of the 5 categories above",
  "confidence": 0.85,
  "reasoning": "Why you chose this category based on the title and description",
  "alternative_category": "second most likely category if applicable"
}}
"""),
    ("human", """Document title: {title}
Description: {description}

Return only the JSON object, no extra text.""")
])


async def classify_document(title: str, description: str) -> dict:
    """
    Classify a document using only title and description.
    Returns category with confidence score and reasoning.
    """
    chain = CLASSIFY_PROMPT | llm
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
    result["note"] = "Classification based on title and description only – full document was NOT read"
    return result
