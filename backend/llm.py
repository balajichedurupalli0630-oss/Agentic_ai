"""
llm_simple.py

Simplified LLM module - uses functools.lru_cache instead of custom cache.
Only uses fast_llm (8b) - no smart_llm needed.
"""

import os
from dotenv import load_dotenv
from functools import lru_cache
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, AIMessage

load_dotenv()


def get_fast_llm() -> ChatGroq:
    """Get fast 8b model for all tasks"""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )


def get_llm() -> ChatGroq:
    """Alias for backward compatibility"""
    return get_fast_llm()


@lru_cache(maxsize=500)
def _cached_sync_call(model_name: str, prompt: str, api_key: str) -> str:
    """Cached synchronous LLM call"""
    llm = ChatGroq(model=model_name, temperature=0, groq_api_key=api_key)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


async def get_cached_llm_response(llm, messages, use_cache: bool = False) -> BaseMessage:
    """
    Async LLM call with optional caching.
    
    Args:
        llm: LangChain chat model
        messages: List[BaseMessage] or string
        use_cache: True for deterministic prompts (contextualization)
    """
    if isinstance(messages, str):
        prompt_str = messages
    else:
        prompt_str = "\n".join(
            f"{getattr(m, 'type', 'msg')}:{m.content if hasattr(m, 'content') else str(m)}"
            for m in messages
        )
    
    # Use LRU cache for deterministic prompts
    if use_cache:
        model_name = getattr(llm, "model", getattr(llm, "model_name", "unknown"))
        api_key = os.environ.get("GROQ_API_KEY", "")
        
        try:
            content = _cached_sync_call(model_name, prompt_str, api_key)
            return AIMessage(content=content)
        except Exception as e:
            print(f"[CACHE] Failed, falling back: {e}")
    
    # Direct async call (no cache)
    try:
        response = await llm.ainvoke(messages)
        return response
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        raise


async def aget_llm_response(llm, messages) -> BaseMessage:
    """Async LLM call without caching - for conversations"""
    return await get_cached_llm_response(llm, messages, use_cache=False)