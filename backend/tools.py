"""
tools.py

Groq-compatible RAG & Web Search tools with adaptive retrieval and Self-RAG query refinement.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from logger import get_logger
from rag_advanced import (
    classify_query_intent,
    grade_retrieved_context,
    refine_query_for_retrieval,
    format_citations_context,
)

logger = get_logger(__name__)

_retrieval_system = None


def set_retrieval_system(system):
    """Wire retrieval system singleton into tool module."""
    global _retrieval_system
    _retrieval_system = system


class RAGInput(BaseModel):
    """Input schema for the RAG answer tool."""
    query: str = Field(description="The question to search for in documents")


class WebSearchInput(BaseModel):
    """Input schema for live web search."""
    query: str = Field(description="The query for live web search")


@tool(args_schema=RAGInput)
async def rag_answer(query: str) -> str:
    """
    Search indexed documents and retrieve relevant information with source citations.
    Use this tool to answer factual questions from uploaded documents.
    """
    logger.info(f"rag_answer tool invoked with query: '{query}'")
    if _retrieval_system is None:
        logger.error("Retrieval system is not initialized.")
        return "ERROR: Retrieval system not initialized."

    try:
        # 1. Query-Adaptive Intent Classification
        route = await classify_query_intent(query)
        strategy = route.get("strategy", "precise")
        top_k = 12 if strategy == "synthesis" else 5

        # 2. Initial Retrieval
        docs = await _retrieval_system.retrieve(
            query=query,
            top_k=top_k,
            use_hybrid=True,
            use_reranking=True
        )

        formatted_context = format_citations_context(docs) if docs else ""

        # 3. Self-RAG Context Grading & CRAG Query Refinement
        is_sufficient = await grade_retrieved_context(query, formatted_context)
        if not is_sufficient:
            logger.info(f"Initial context graded insufficient for '{query}'. Initiating CRAG query refinement...")
            refined_q = await refine_query_for_retrieval(query)

            re_docs = await _retrieval_system.retrieve(
                query=refined_q,
                top_k=top_k,
                use_hybrid=True,
                use_reranking=True
            )

            if re_docs:
                docs = re_docs
                formatted_context = format_citations_context(docs)

        if not docs or not formatted_context:
            logger.warning(f"No relevant document chunks found for query: '{query}'")
            return "No relevant documents found to answer this question."

        logger.info(f"Returning formatted citations context ({len(formatted_context)} chars from {len(docs)} chunks)")
        return formatted_context

    except Exception as e:
        logger.error(f"Error in rag_answer tool: {e}", exc_info=True)
        return f"ERROR: {str(e)}"


@tool(args_schema=WebSearchInput)
async def web_search(query: str) -> str:
    """
    Search the live web for current events, breaking updates, or time-sensitive information.
    Use this tool when information is not found in documents or requires live internet data.
    """
    logger.info(f"web_search tool invoked with query: '{query}'")
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        ddg = DuckDuckGoSearchRun()
        results = ddg.run(query)
        if not results:
            return "No live web results found for this query."
        logger.info(f"Web search returned {len(results)} chars")
        return f"[Live Web Search Results]\n{results[:3000]}"
    except Exception as e:
        logger.error(f"Web search failed ({e})", exc_info=True)
        return f"Live web search failed: {str(e)}"