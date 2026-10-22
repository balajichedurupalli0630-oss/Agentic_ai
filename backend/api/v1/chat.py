"""
chat.py

API v1 Chat endpoints supporting blocking JSON responses and SSE streaming.
"""

import json
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from auth import verify_api_key
from config import settings
from dependencies import get_agent_graph, get_retrieval_system
from logger import get_logger
from vector_base import vectorstore

logger = get_logger(__name__)
router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: str = Field(default="default", description="Session ID")


class ChatResponse(BaseModel):
    response: str
    session_id: str


def _build_input(message: str, session_id: str) -> tuple[dict, dict]:
    """Construct input state and LangGraph thread configuration."""
    config = {"configurable": {"thread_id": session_id}}
    state = {
        "messages": [{"role": "user", "content": message}],
        "session_id": session_id,
    }
    return state, config


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    body: ChatRequest,
    _auth=Depends(verify_api_key),
):
    """Blocking chat — returns completed agent response."""
    if vectorstore.collection.count() == 0:
        return ChatResponse(
            response="No documents uploaded. Please upload documents first.",
            session_id=body.session_id,
        )

    try:
        agent_graph = get_agent_graph()
        state, config = _build_input(body.message, body.session_id)
        logger.info(f"Invoking agent for session '{body.session_id}'...")
        result = await agent_graph.ainvoke(state, config=config)
        messages = result.get("messages", [])
        response_text = (
            messages[-1].content
            if messages and hasattr(messages[-1], "content")
            else "No response generated"
        )
        return ChatResponse(response=response_text, session_id=body.session_id)
    except Exception as e:
        logger.error(f"Chat processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/stream")
async def chat_stream(
    request: Request,
    message: str,
    session_id: str = "default",
    _auth=Depends(verify_api_key),
):
    """Server-Sent Events (SSE) streaming endpoint."""
    if vectorstore.collection.count() == 0:
        async def _empty():
            yield f"data: {json.dumps({'token': 'No documents uploaded. Please upload documents first.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    async def generate():
        agent_graph = get_agent_graph()
        state, config = _build_input(message, session_id)
        try:
            async for event in agent_graph.astream_events(state, config=config, version="v2"):
                if event["event"] != "on_chat_model_stream":
                    continue
                node = event.get("metadata", {}).get("langgraph_node", "")
                if node == "tools":
                    continue
                chunk = event["data"].get("chunk")
                if not chunk:
                    continue
                content = chunk.content if hasattr(chunk, "content") else ""
                if not content or not isinstance(content, str):
                    continue
                yield f"data: {json.dumps({'token': content})}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream generation error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
