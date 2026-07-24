"""
tools.py - FIXED for Groq compatibility
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

_retrieval_system = None


def set_retrieval_system(system):
    """Wire retrieval system into tool"""
    global _retrieval_system
    _retrieval_system = system


# Define input schema explicitly for Groq
class RAGInput(BaseModel):
    """Input for the RAG answer tool"""
    query: str = Field(description="The question to search for in documents")


@tool(args_schema=RAGInput)
async def rag_answer(query: str) -> str:
    """
    Search documents and retrieve information to answer questions.
    Use this tool to find relevant information from uploaded documents.
    """
    print(f"[TOOL START] rag_answer tool invoked with query: '{query}'")
    if _retrieval_system is None:
        print("[TOOL ERROR] Retrieval system is not initialized.")
        return "ERROR: Retrieval system not initialized."

    try:
        # Retrieve with cross-encoder reranking
        docs = await _retrieval_system.retrieve(
            query=query,
            top_k=5,
            use_hybrid=True,
            use_reranking=True
        )

        if not docs:
            print("[TOOL WARNING] No relevant documents found from retrieval.")
            return "No relevant documents found to answer this question."

        # Format results
        MAX_TOTAL_CHARS = 6000
        MAX_DOC_CHARS = 1500
        
        docs = sorted(
            docs,
            key=lambda x: x.get("final_score", 
                               x.get("combined_score", 
                                    x.get("similarity_score", 0))),
            reverse=True
        )
        
        parts = []
        total = 0
        
        for i, doc in enumerate(docs[:5], 1):
            score = doc.get("final_score", doc.get("combined_score", 0))
            content = doc["content"][:MAX_DOC_CHARS]
            meta = doc.get("metadata", {}) or {}
            content_type = meta.get("content_type", "unknown")
            entry = f"[Doc {i} | Score: {score:.3f} | Type: {content_type}]\n{content}"
            
            if total + len(entry) > MAX_TOTAL_CHARS:
                break
            
            parts.append(entry)
            total += len(entry)

        context = "\n\n---\n".join(parts)
        print(f"[TOOL] Returning context ({len(context)} chars from {len(parts)} docs)")
        print("[TOOL] Retrieved documents detailed summary:")
        for idx, doc in enumerate(docs[:5], 1):
            meta = doc.get("metadata", {}) or {}
            c_type = meta.get("content_type", "unknown")
            score = doc.get("final_score", doc.get("combined_score", 0))
            print(f"  - Doc {idx}: score={score:.3f}, type={c_type}")
        return context

    except Exception as e:
        import traceback
        print(f"[ERROR] rag_answer: {e}")
        traceback.print_exc()
        return f"ERROR: {str(e)}"