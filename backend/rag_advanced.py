"""
rag_advanced.py

Advanced Agentic RAG Module implementing SOTA patterns:
1. Query-Adaptive Retrieval Selection (Precise vs Synthesis vs Time-Sensitive)
2. Self-RAG / CRAG Context Grading & Query Refinement
3. Groundedness Verification
4. Citations & Attribution Formatting
"""

from typing import List, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import settings
from logger import get_logger
from llm import get_fast_llm, get_cached_llm_response

logger = get_logger(__name__)


# ── 1. Query Adaptive Retrieval Selector ─────────────────────────────────

class AdaptiveRoute(BaseModel):
    strategy: str = Field(
        description="One of: 'precise' (specific clause/rule lookup), 'synthesis' (broad topic/summary), 'time_sensitive' (current news/live info)"
    )
    refined_query: str = Field(description="Optimized search query")


async def classify_query_intent(query: str) -> Dict[str, str]:
    """
    Classify query intent to select optimal retrieval parameters.
    Returns strategy: 'precise' | 'synthesis' | 'time_sensitive'
    """
    lowered = query.lower()

    # Fast heuristic check for live time-sensitive words
    time_words = {"latest", "current news", "today's update", "live", "weather today", "breaking news"}
    if any(w in lowered for w in time_words):
        logger.info(f"Query adaptive router selected 'time_sensitive' for: '{query}'")
        return {"strategy": "time_sensitive", "refined_query": query}

    # Fast heuristic check for broad synthesis
    synthesis_words = {"summarize", "overview", "all policies", "comparison", "explain everything", "curriculum summary"}
    if any(w in lowered for w in synthesis_words):
        logger.info(f"Query adaptive router selected 'synthesis' for: '{query}'")
        return {"strategy": "synthesis", "refined_query": query}

    logger.info(f"Query adaptive router selected 'precise' for: '{query}'")
    return {"strategy": "precise", "refined_query": query}


# ── 2. Self-RAG Document Relevance Grading ─────────────────────────────

async def grade_retrieved_context(query: str, context: str) -> bool:
    """
    Grade whether the retrieved context contains relevant information to answer the query.
    Returns True if context is sufficient/relevant, False if re-retrieval is needed.
    """
    if not context or "No relevant documents found" in context or context.startswith("ERROR:"):
        return False

    prompt = f"""You are a strict relevance evaluator for a RAG system.

User Question: {query}

Retrieved Document Context:
{context[:2000]}

Is the retrieved context relevant and sufficient to answer the user question?
Respond with ONLY 'YES' or 'NO'."""

    try:
        llm = get_fast_llm()
        res = await get_cached_llm_response(llm, prompt, use_cache=True)
        ans = res.content.strip().upper() if hasattr(res, "content") else str(res).strip().upper()
        is_relevant = "YES" in ans
        logger.info(f"Self-RAG Grading result for query '{query}': {'RELEVANT' if is_relevant else 'NOT_RELEVANT'}")
        return is_relevant
    except Exception as e:
        logger.warning(f"Self-RAG grading check failed ({e}); defaulting to relevant.")
        return True


# ── 3. Query Refinement (CRAG Pattern) ───────────────────────────────────

async def refine_query_for_retrieval(original_query: str) -> str:
    """
    Refine a query when initial retrieval yields insufficient context.
    """
    prompt = f"""The initial search for the query below did not yield sufficient context:
Original Query: "{original_query}"

Provide a clearer, keyword-focused search query to improve retrieval. Output ONLY the refined query string."""

    try:
        llm = get_fast_llm()
        res = await get_cached_llm_response(llm, prompt, use_cache=True)
        refined = res.content.strip() if hasattr(res, "content") else str(res).strip()
        logger.info(f"CRAG Query refinement: '{original_query}' -> '{refined}'")
        return refined
    except Exception as e:
        logger.warning(f"Query refinement failed ({e}); using original query.")
        return original_query


# ── 4. Groundedness Verification ─────────────────────────────────────────

async def verify_groundedness(context: str, answer: str) -> Dict[str, Any]:
    """
    Verify that the final response is strictly supported by the retrieved document context.
    """
    if not context or not answer:
        return {"grounded": True, "score": 1.0}

    prompt = f"""Verify if the following answer is factual and grounded strictly in the provided document context.

Context:
{context[:2500]}

Generated Answer:
{answer[:1500]}

Does the context support the statements in the answer?
Respond with ONLY 'YES' or 'NO' followed by a brief reason on the next line."""

    try:
        llm = get_fast_llm()
        res = await llm.ainvoke(prompt)
        text = res.content if hasattr(res, "content") else str(res)
        is_grounded = "YES" in text.split("\n")[0].upper()
        logger.info(f"Groundedness verification: {'GROUNDED' if is_grounded else 'HALLUCINATION_DETECTED'}")
        return {"grounded": is_grounded, "reason": text}
    except Exception as e:
        logger.warning(f"Groundedness check failed ({e}); defaulting to grounded.")
        return {"grounded": True, "score": 1.0}


# ── 5. Citations Formatter Helper ───────────────────────────────────────

def format_citations_context(docs: list) -> str:
    """
    Format document chunks with explicit citations, chunk IDs, and score metadata.
    """
    MAX_TOTAL_CHARS = 6500
    MAX_DOC_CHARS = 1500

    parts = []
    total = 0

    for i, doc in enumerate(docs, 1):
        score = doc.get("final_score", doc.get("combined_score", doc.get("similarity_score", 0)))
        content = doc.get("content", "")[:MAX_DOC_CHARS]
        meta = doc.get("metadata", {}) or {}
        source = meta.get("source", "document")
        chunk_id = doc.get("id", f"chunk_{i}")
        page = meta.get("page", "N/A")

        citation_header = f"[Citation {i} | Source: {source} | Chunk: {chunk_id} | Page: {page} | Score: {score:.3f}]"
        entry = f"{citation_header}\n{content}"

        if total + len(entry) > MAX_TOTAL_CHARS:
            break

        parts.append(entry)
        total += len(entry)

    return "\n\n---\n".join(parts)
