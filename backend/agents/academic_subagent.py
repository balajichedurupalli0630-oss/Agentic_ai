"""
academic_subagent.py

Academic Sub-Agent specialized in Academic Period, Exam Schedule, and Syllabus queries.
"""

from logger import get_logger
from tools import rag_answer

logger = get_logger(__name__)


async def run_academic_subagent(query: str) -> str:
    """Execute academic operations queries."""
    logger.info(f"Academic Sub-Agent processing query: '{query}'")
    try:
        # Enhance query with academic operational terms
        enhanced_query = f"Academic schedule exam rules syllabus fees: {query}"
        context = await rag_answer.ainvoke({"query": enhanced_query})
        return context
    except Exception as e:
        logger.error(f"Academic Sub-Agent error: {e}", exc_info=True)
        return f"Academic query error: {str(e)}"
