"""
retrieval_subagent.py

Retrieval Sub-Agent specialized in Document Hybrid Search, Self-RAG, and Web Search.
"""

from logger import get_logger
from tools import rag_answer, web_search
from llm import get_fast_llm

logger = get_logger(__name__)


async def run_retrieval_subagent(query: str) -> str:
    """Execute document retrieval and search workflows."""
    logger.info(f"Retrieval Sub-Agent processing query: '{query}'")
    try:
        context = await rag_answer.ainvoke({"query": query})
        if not context or "No relevant documents found" in context:
            logger.info("Retrieval Sub-Agent triggering web search fallback...")
            web_res = await web_search.ainvoke({"query": query})
            return web_res
        return context
    except Exception as e:
        logger.error(f"Retrieval Sub-Agent error: {e}", exc_info=True)
        return f"Retrieval error: {str(e)}"
