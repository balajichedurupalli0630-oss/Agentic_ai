"""
coordinator.py

Coordinator Agent (Lead Orchestrator) routing user requests to specialized sub-agents:
- Retrieval Sub-Agent (Document Search & Self-RAG)
- Academic Operations Sub-Agent (Schedules, Fees, Syllabus)
- Student Service Sub-Agent (Hall Tickets, Transcripts, Leave Requests)
"""

from typing import Dict, Any
from logger import get_logger
from agents.retrieval_subagent import run_retrieval_subagent
from agents.academic_subagent import run_academic_subagent
from agents.service_subagent import run_service_subagent

logger = get_logger(__name__)


async def route_and_execute_agent(message: str, user_id: str = "default_user") -> Dict[str, Any]:
    """
    Lead Coordinator Agent function: classifies intent and delegates to sub-agents.
    """
    lowered = message.lower()

    # Intent Classification
    if any(k in lowered for k in ["hall ticket", "transcript", "leave app", "change address", "update profile"]):
        target = "service"
    elif any(k in lowered for k in ["exam", "schedule", "fee", "syllabus", "period", "semester"]):
        target = "academic"
    else:
        target = "retrieval"

    logger.info(f"Coordinator Agent routed message to '{target}' sub-agent")

    if target == "service":
        result = await run_service_subagent("general", message, user_id=user_id)
    elif target == "academic":
        result = await run_academic_subagent(message)
    else:
        result = await run_retrieval_subagent(message)

    return {
        "target_subagent": target,
        "response": result,
    }
