"""
dependencies.py

Global dependency container for initialized application singletons.
Provides clean accessors for router endpoints.
"""

from typing import Any

_retrieval_system: Any = None
_agent_graph: Any = None


def set_app_dependencies(retrieval_sys: Any, agent_g: Any) -> None:
    """Wire application singletons into dependencies module."""
    global _retrieval_system, _agent_graph
    _retrieval_system = retrieval_sys
    _agent_graph = agent_g


def get_retrieval_system() -> Any:
    """Retrieve the global HybridRetrieval instance."""
    if _retrieval_system is None:
        raise RuntimeError("Retrieval system has not been initialized.")
    return _retrieval_system


def get_agent_graph() -> Any:
    """Retrieve the compiled LangGraph agent instance."""
    if _agent_graph is None:
        raise RuntimeError("Agent graph has not been initialized.")
    return _agent_graph
