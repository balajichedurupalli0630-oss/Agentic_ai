"""
run_cli_agent.py

Interactive Command Line Interface to run and test the Agentic AI agent.
"""

import sys
import os
import asyncio
import dotenv

dotenv.load_dotenv()

from vector_base import vectorstore
from embeddings import embedding_loader
from hybrid_search import HybridRetrieval
from tools import set_retrieval_system
from agent import build_agent

async def main():
    print("=" * 60)
    print("🤖 Agentic AI CLI - LangGraph RAG Agent")
    print("=" * 60)
    
    print("\n[INIT] Initializing Hybrid Retrieval System...")
    retrieval_system = HybridRetrieval(
        vectorstore=vectorstore,
        embedding_loader=embedding_loader
    )
    set_retrieval_system(retrieval_system)
    
    print("[INIT] Building LangGraph Agent...")
    agent_graph = build_agent()
    print("✓ Agent Ready!\n")

    session_id = "cli_session"
    config = {"configurable": {"thread_id": session_id}}

    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            
            state = {
                "messages": [{"role": "user", "content": user_input}],
                "session_id": session_id
            }

            print("\nAgent > ", end="", flush=True)
            async for event in agent_graph.astream_events(state, config=config, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    node = event.get("metadata", {}).get("langgraph_node", "")
                    if node == "tools":
                        continue
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        if isinstance(chunk.content, str):
                            print(chunk.content, end="", flush=True)
            print("\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}\n")

if __name__ == "__main__":
    asyncio.run(main())
