from langchain_groq import ChatGroq

from app.config import settings


def get_primary_llm(temperature: float = 0.2):
    """gemma2-9b-it — mandated by the task spec. Fast, cheap, good for
    structured extraction / summarization on the free Groq tier."""
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
