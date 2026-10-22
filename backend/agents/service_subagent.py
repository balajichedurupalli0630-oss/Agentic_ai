"""
service_subagent.py

Student Service Sub-Agent specialized in Student Actions (Hall Tickets, Transcripts, Leave Requests).
"""

from typing import Dict, Any
from logger import get_logger

logger = get_logger(__name__)


async def run_service_subagent(action_type: str, query: str, user_id: str = "student") -> str:
    """Execute student workflow actions."""
    logger.info(f"Student Service Sub-Agent executing action '{action_type}' for user '{user_id}'")

    action_map = {
        "hall_ticket": f"📄 **Hall Ticket Request**\nHall Ticket generated for Student `{user_id}`.\nDownload Link: [Hall_Ticket_{user_id}.pdf](#)",
        "transcript": f"📜 **Transcript Request**\nOfficial Transcript request queued for Student `{user_id}`.\nReference Ticket ID: `REQ-{user_id[:6].upper()}-2026`",
        "leave_application": f"📝 **Leave Application**\nLeave application submitted for approval to Department HOD.\nStatus: Pending Approval",
        "address_update": f"🏠 **Address Update Request**\nProfile change request logged into Student Portal.\nReference ID: `UPD-2026-88`",
    }

    lowered = query.lower()
    for key, response in action_map.items():
        if key in lowered or key.replace("_", " ") in lowered:
            return response

    return action_map.get(action_type, f"Action '{action_type}' processed for user `{user_id}`.")
