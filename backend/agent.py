"""
agent.py

LangGraph Agent Orchestrator with:
- Checkpointer (PostgresSaver with MemorySaver fallback)
- Self-RAG & Adaptive Retrieval Tools (rag_answer, web_search)
- Groundedness verification
- Conversation summarization
"""

from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, RemoveMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq

from config import settings
from logger import get_logger
from state import AgentState
from llm import aget_llm_response, get_fast_llm
from tools import rag_answer, web_search
from rag_advanced import verify_groundedness
from db.session import is_db_available

logger = get_logger(__name__)

tools = [rag_answer, web_search]
MAX_MESSAGES_BEFORE_SUMMARY = 12


def get_checkpointer():
    """Initialize PostgresSaver checkpointer if DB active; fallback to MemorySaver."""
    if settings.sync_database_url and is_db_available():
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            logger.info("Initializing PostgresSaver checkpointer...")
            saver = PostgresSaver.from_conn_string(settings.sync_database_url)
            saver.setup()
            logger.info("✓ PostgresSaver checkpointer setup verified")
            return saver
        except Exception as e:
            logger.warning(f"PostgresSaver setup fallback ({e}); using MemorySaver.")
    return MemorySaver()


llm = ChatGroq(
    model=settings.GROQ_PRIMARY_MODEL,
    temperature=0,
    groq_api_key=settings.GROQ_API_KEY or None,
)
llm_with_tools = llm.bind_tools(tools)


def _build_system_prompt() -> str:
    now = datetime.now()
    date_str = now.strftime("%d %B %Y")
    month = now.month

    if 1 <= month <= 5:
        semester_hint = "Even Semester (January – May)"
    elif 6 <= month <= 7:
        semester_hint = "between semesters (summer/registration period)"
    else:
        semester_hint = "Odd Semester (August – December)"

    return f"""You are a Grounded Document & Operations Assistant — reliable and precise.

Today's date: {date_str}
Current academic period: {semester_hint}

MANDATORY TOOL RULES:
- Call `rag_answer` for factual questions about documents, college policies, schedules, fees, or course rules.
- Call `web_search` for current live information or breaking updates not present in documents.
- Never answer factual document questions from unverified pre-trained general knowledge.

RESPONSE & CITATION GUIDELINES:
- When answering from retrieved documents, include inline citations referencing source files (e.g. "[Source: policy_doc.pdf]").
- Clearly state if information is not found in documents. Never hallucinate document text.
- Be direct, professional, and structured.
"""


tool_node = ToolNode(tools=tools)


async def agent_node(state: AgentState):
    """Main agent node — injects system prompt, conversation summary, and calls LLM."""
    messages = state["messages"].copy()

    system_content = _build_system_prompt()
    summary = state.get("summary", "")
    if summary:
        system_content += f"\n\n---\nPrior conversation summary:\n{summary}\n---"

    messages.insert(0, SystemMessage(content=system_content))
    last_msg_content = messages[-1].content if messages else ""
    logger.info(f"Agent node invoking LLM... Last message snippet: '{last_msg_content[:80]}...'")

    try:
        response = await aget_llm_response(llm_with_tools, messages)
        tool_calls = getattr(response, "tool_calls", [])
        logger.info(f"Agent node LLM call completed (tool_calls={len(tool_calls)})")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"LLM call error in agent node: {e}", exc_info=True)
        raise


async def groundedness_node(state: AgentState):
    """Verify that final generated text is grounded in retrieved tool outputs."""
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1]
    if getattr(last_msg, "tool_calls", None):
        return {}

    # Extract recent tool content context
    tool_contexts = [
        m.content for m in messages[-4:]
        if getattr(m, "type", "") == "tool" and isinstance(m.content, str)
    ]
    if not tool_contexts:
        return {}

    combined_context = "\n".join(tool_contexts)
    grounded_res = await verify_groundedness(combined_context, last_msg.content)

    if not grounded_res.get("grounded", True):
        logger.warning("Unverified hallucination detected in answer! Appending groundedness warning.")
        warning_msg = "\n\n*(Note: Some details in this answer could not be verified against indexed documents).* "
        updated_content = last_msg.content + warning_msg
        return {"messages": [AIMessage(content=updated_content)]}

    return {}


async def summarize_node(state: AgentState):
    """Summarize conversation history when message count exceeds MAX_MESSAGES_BEFORE_SUMMARY."""
    messages = state["messages"]
    if len(messages) <= MAX_MESSAGES_BEFORE_SUMMARY:
        return {}

    existing_summary = state.get("summary") or ""
    messages_to_summarize = messages[:-4]

    prior = f"Prior summary:\n{existing_summary}\n\n" if existing_summary else ""
    history_text = "\n".join(
        f"{getattr(m, 'type', 'msg').upper()}: {getattr(m, 'content', str(m))}"
        for m in messages_to_summarize
    )

    summary_prompt = (
        f"{prior}"
        f"Summarize the following conversation turns concisely. "
        f"Capture key topics, questions asked, and answers given.\n\n"
        f"{history_text}\n\n"
        f"Concise summary:"
    )

    fast_llm = get_fast_llm()
    try:
        response = await fast_llm.ainvoke(summary_prompt)
        new_summary = response.content.strip()
    except Exception as e:
        logger.error(f"Summarize node error: {e}", exc_info=True)
        return {}

    delete_ops = [RemoveMessage(id=m.id) for m in messages_to_summarize]
    logger.info(f"Summarize node trimmed {len(messages_to_summarize)} messages. Summary length: {len(new_summary)}")
    return {
        "messages": delete_ops,
        "summary": new_summary,
    }


def build_agent():
    """Build LangGraph agent state graph."""
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("groundedness", groundedness_node)
    builder.add_node("summarize", summarize_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: "groundedness"}
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("groundedness", "summarize")
    builder.add_edge("summarize", END)

    checkpointer = get_checkpointer()
    return builder.compile(checkpointer=checkpointer)