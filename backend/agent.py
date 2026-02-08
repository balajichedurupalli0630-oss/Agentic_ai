from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from state import AgentState
from langgraph.checkpoint.memory import MemorySaver
from llm import get_llm, Ollama_llm, aget_llm_response
from tools import rag_answer

# Initialize components
memory = MemorySaver()
tools = [rag_answer]
llm = get_llm()
llm_with_tools = llm.bind_tools(tools)

STYLE_SYSTEM = """
You are an intelligent, reliable, and helpful AI assistant designed to support users with:
before answering any Query you should check documents first ..
- Answering questions
- Explaining concepts
- Generating and improving code
- Summarizing and analyzing documents
- Assisting with learning, problem-solving, and decision-making

You should:
- Be clear, concise, and structured
- Ask clarifying questions when needed
- Provide step-by-step explanations for complex topics
- Adapt responses to the user's skill level
- Prioritize correctness and safety

You have access to:
- A vector database containing user-uploaded documents
- Tools for retrieving, searching, and summarizing documents
- Programming and reasoning capabilities

If information is not found in the available documents, rely on general knowledge and clearly state that the answer is based on general knowledge.

Never hallucinate document content. If unsure, say you do not know.

If the user asks about capabilities, tools, or documents, respond using the predefined capability explanation.

"""

# Create tool node - this will ACTUALLY execute the tools
tool_node = ToolNode(tools=tools)

async def agent_node(state: AgentState):
    """
    Simple agent node: Just calls LLM with tools.
    DOES NOT intercept tool calls - lets ToolNode handle execution.
    """
    messages = state["messages"].copy()

    # Add system prompt at the beginning
    messages.insert(0, SystemMessage(content=STYLE_SYSTEM))

    # Call LLM (if it decides to use tools, ToolNode will execute them)
    response = await aget_llm_response(llm_with_tools, messages)
    
    # Just return the response - if it has tool_calls, 
    # the graph will route to ToolNode automatically
    return {"messages": [response]}

async def summary_node(state: AgentState) -> AgentState:
    """Summarize conversation if it gets too long"""
    messages = state["messages"]
    
    # Only summarize if we have many messages (optional optimization)
    if len(messages) < 10:
        return {}
    
    summary_prompt = [
        SystemMessage(
            content=(
                "Summarize the conversation so far. "
                "Preserve the important details and user's intent clearly."
            )
        ),
        *messages
    ]
    
    llm1 = Ollama_llm()
    summary_response = await aget_llm_response(llm1, summary_prompt)
    summary = summary_response.content
    
    # Keep last 4 messages + summary
    return {
        "messages": messages[-4:],
        "summary": summary
    }

def build_agent():
    # Create graph
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("summary", summary_node)
    
    # Define edges
    builder.add_edge(START, "agent")
    
    # Conditional: if tool_calls exist -> go to tools, else -> end
    builder.add_conditional_edges(
        "agent",
        tools_condition,  # Built-in: returns "tools" if tool_calls present, else END
        {
            "tools": "tools",
            END: END
        }
    )
    
    # After tools execute, return to agent to process results
    builder.add_edge("tools", "agent")
    
    # Optional: Add summarization logic if needed
    # For now, we keep it simple: agent -> END or agent -> tools -> agent -> END
    
    return builder.compile(checkpointer=memory)