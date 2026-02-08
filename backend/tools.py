from unittest import result
from langchain.tools import tool
from typing import List, Dict, Any
from rag import RAG
from embeddings import embedding_loader
from vector_base import vectorstore
from llm import get_llm, aget_llm_response
from langchain_core.messages import HumanMessage

# Initialize RAG system
llm = get_llm()
rag_system = RAG(vectorstore, embedding_loader, llm)

@tool
async def rag_answer(query: str) -> str:
    """Search documents and retrieve information to answer questions.
    
    Args:
        query: The question to search for in documents
    
    Returns:
        Answer based on retrieved documents
    """
    try:
    # Async retrieval
        results = await rag_system.retrieve(query, top_k=10)

        if not isinstance(results , list):
            print(f"[ERROR] Excepted list ,got {type(results)}")
            return "ERROR : Invalid Retrieved format "
        
        if not results:
            return "No relevant documents found to answer this question."
        
        # Sort by similarity score
        results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
        
        # Build context
        context_parts = []
       
        for i , doc in enumerate(results[:5] , 1):
            content = doc['content']
            metadata = doc['metadata']
            score = doc['similarity_score']
           

            context_parts.append(f" Document : {i} | Score : {score:.3f}  \n {content} | metadata : {metadata}"  )
         
        context = "\n\n--\n".join(context_parts)
        print(f"[TOOL] Returning context ({len(context)} chars)")
        return context

    except Exception as e:

        print(f"[ERROR] rag answer : {e}")
        import traceback
        traceback.print_exc()
        return f"ERROR : {str(e)}"



     











































