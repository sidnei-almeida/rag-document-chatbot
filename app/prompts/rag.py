"""RAG and general conversation prompts."""

import re

from app.core.config import settings


def is_general_question(question: str) -> bool:
    """Return True when the question is conversational and does not need document context."""
    q = question.lower().strip()
    return any(re.search(rf"\b{re.escape(gq)}\b", q) for gq in settings.GENERAL_QUESTIONS)


def create_rag_prompt(question: str, context: str) -> str:
    """Build the main RAG prompt with retrieved document context."""
    return f"""You are DocMind, a document intelligence assistant.

Answer ONLY using the retrieved context from the documents below.
If the context does not contain enough evidence to answer confidently, say clearly that the document does not contain sufficient information to answer reliably.
Never invent facts, metrics, names, dates, page numbers, or conclusions.
Do not claim something is in the document if it is not present in the context.
Be clear, objective, and professional.
When possible, mention that your answer is based on the retrieved excerpts.
If the question is generic, respond helpfully but make clear when you are limited to the document.

Retrieved context:
{context}

User question:
{question}

Answer:"""


def create_general_prompt(question: str) -> str:
    """Build a prompt for conversational questions that do not require document context."""
    return f"""You are DocMind, a document intelligence assistant.

The user sent a general message that does not require document context.

User message:
{question}

Respond briefly, clearly, and professionally. Explain that you can answer questions about uploaded PDF documents once they are indexed."""
