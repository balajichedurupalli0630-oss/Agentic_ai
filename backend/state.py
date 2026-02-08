

from langgraph.graph.message import add_messages

from typing_extensions import TypedDict
from typing import Annotated, Optional 


class AgentState(TypedDict):
    session_id : str
    messages : Annotated[list, add_messages]
    summary : Optional[str]
    tool_used: bool
    
    
    
