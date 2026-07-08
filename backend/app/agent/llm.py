from langchain_groq import ChatGroq

from app.config import settings


# llm.py
def get_primary_llm(temperature: float = 0.2):
    """llama-3.1-8b-instant — Groq's official replacement for gemma2-9b-it
    (decommissioned 2025-10-08). Fast, cheap, used for the mandated
    structured extraction / summarization step inside log_interaction /
    edit_interaction."""
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=temperature,
    )

def get_fallback_llm(temperature: float = 0.2):
    """llama-3.3-70b-versatile — used for heavier reasoning / routing
    decisions when gemma2-9b-it's smaller context isn't enough."""
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_fallback_model,
        temperature=temperature,
    )
