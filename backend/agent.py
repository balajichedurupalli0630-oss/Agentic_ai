"""
agent.py

LangGraph agent with:
- Conversation summarization (trims old messages when history grows)
- Summary injected into system prompt for context continuity
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, RemoveMessage
from state import AgentState
from langgraph.checkpoint.memory import MemorySaver
from llm import aget_llm_response, get_fast_llm
from tools import rag_answer
from langchain_groq import ChatGroq
import os

memory = MemorySaver()
tools = [rag_answer]

# Number of messages before we trigger summarization
MAX_MESSAGES_BEFORE_SUMMARY = 12

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.environ.get("GROQ_API_KEY"),
)
llm_with_tools = llm.bind_tools(tools)

from datetime import datetime

def _build_system_prompt() -> str:
    now = datetime.now()
    date_str = now.strftime("%d %B %Y")  # e.g. "26 March 2026"
    month = now.month

    # Determine academic semester from current month
    if 1 <= month <= 5:
        semester_hint = "Even Semester (January – May)"
    elif 6 <= month <= 7:
        semester_hint = "between semesters (summer/registration period)"
    else:
        semester_hint = "Odd Semester (August – December)"

    return f"""You are a Document Grounded Assistant — reliable and helpful.

Today's date: {date_str}
Current academic period: {semester_hint}

CRITICAL RULE — MANDATORY TOOL USE:
- You MUST always call the `rag_answer` tool for any factual questions about college operations, policies, schedules, fees, or events.
- Never answer a factual question from your pre-trained general knowledge; always retrieve context first.

IMPORTANT — time-aware query handling:
- When the user says "this ", "current ", "now", "upcoming" — use the
  current academic period above to interpret which semester they mean.
- When calling the rag_answer tool, make the query SPECIFIC. Do not pass vague terms
  like "this semester" — instead pass "Even Semester 2026 holidays" or
  "Odd Semester 2025 exam schedule" based on context.
- If retrieved results seem to be from the wrong semester, explicitly note that and
  clarify which semester the data belongs to.

Response guidelines:
- Answer directly from retrieved documents
- If sufficient information found, respond immediately
- Do NOT make multiple tool calls for the same question

You should:
- Be clear, concise, and structured
- Ask clarifying questions when needed
- Provide step-by-step explanations for complex topics
- Prioritize correctness and safety

If information is not found in documents, clearly state that.
Never hallucinate document content.

Response Style:
Clear, friendly, and professional. Simple language. Short paragraphs.
"""

STYLE_SYSTEM = _build_system_prompt()

tool_node = ToolNode(tools=tools)


async def agent_node(state: AgentState):
    """Main agent — injects conversation summary and fresh date into system prompt"""
    messages = state["messages"].copy()

    # Rebuild prompt each call so the date is always current
    system_content = _build_system_prompt()
    summary = state.get("summary", "")
    if summary:
        system_content += f"\n\n---\nConversation summary so far:\n{summary}\n---"

    messages.insert(0, SystemMessage(content=system_content))
    last_msg_content = messages[-1].content if messages else ""
    print(f"[AGENT START] Invoking LLM with tool bindings... Last message: '{last_msg_content[:80]}...'")
    
    try:
        response = await aget_llm_response(llm_with_tools, messages)
        print(f"[AGENT SUCCESS] LLM responded with: '{response.content[:80]}...' and tool_calls={getattr(response, 'tool_calls', [])}")
        return {"messages": [response]}
    except Exception as e:
        print(f"[AGENT ERROR] LLM generation failed: {e}")
        raise


async def summarize_node(state: AgentState):
    """
    Summarize conversation when it exceeds MAX_MESSAGES_BEFORE_SUMMARY.
    Uses RemoveMessage to trim old messages, preserving the last 4.
    """
    messages = state["messages"]

    if len(messages) <= MAX_MESSAGES_BEFORE_SUMMARY:
        return {}  # Nothing to do

    existing_summary = state.get("summary") or ""
    messages_to_summarize = messages[:-4]   # All except last 4
    recent_messages = messages[-4:]          # Keep these

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
        print(f"[SUMMARIZE] Failed: {e}")
        return {}

    # Delete old messages from state (LangGraph RemoveMessage pattern)
    delete_ops = [RemoveMessage(id=m.id) for m in messages_to_summarize]

    print(f"[SUMMARIZE] Trimmed {len(messages_to_summarize)} messages. New summary length: {len(new_summary)}")
    return {
        "messages": delete_ops,
        "summary": new_summary,
    }


def build_agent():
    """Build LangGraph agent with summarization"""
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("summarize", summarize_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: "summarize"}  # always pass through summarize
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("summarize", END)

    return builder.compile(checkpointer=memory)