"""
llm.py

LLM invocation helper module utilizing pydantic settings and structured logging.
"""

from functools import lru_cache
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, AIMessage

from config import settings
from logger import get_logger

logger = get_logger(__name__)


def get_fast_llm() -> ChatGroq:
    """Get fast model for general tasks."""
    return ChatGroq(
        model=settings.GROQ_FAST_MODEL,
        temperature=0,
        groq_api_key=settings.GROQ_API_KEY or None,
    )


def get_llm() -> ChatGroq:
    """Alias for backward compatibility."""
    return get_fast_llm()


@lru_cache(maxsize=500)
def _cached_sync_call(model_name: str, prompt: str, api_key: str) -> str:
    """Cached synchronous LLM call for deterministic contextualization."""
    llm = ChatGroq(model=model_name, temperature=0, groq_api_key=api_key or None)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


async def get_cached_llm_response(llm, messages, use_cache: bool = False) -> BaseMessage:
    """
    Async LLM call with optional LRU caching.
    """
    if isinstance(messages, str):
        prompt_str = messages
    else:
        prompt_str = "\n".join(
            f"{getattr(m, 'type', 'msg')}:{m.content if hasattr(m, 'content') else str(m)}"
            for m in messages
        )

    if use_cache:
        model_name = getattr(llm, "model", getattr(llm, "model_name", settings.GROQ_FAST_MODEL))
        api_key = settings.GROQ_API_KEY
        try:
            content = _cached_sync_call(model_name, prompt_str, api_key)
            return AIMessage(content=content)
        except Exception as e:
            logger.warning(f"Cache lookup failed, falling back to async LLM call: {e}")

    try:
        response = await llm.ainvoke(messages)
        return response
    except Exception as e:
        logger.error(f"LLM generation failed: {e}", exc_info=True)
        raise


async def aget_llm_response(llm, messages) -> BaseMessage:
    """Async LLM call without caching - for interactive conversations."""
    return await get_cached_llm_response(llm, messages, use_cache=False)