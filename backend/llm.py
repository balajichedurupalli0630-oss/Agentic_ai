import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )

def Ollama_llm():
    return ChatOllama(
        model="llama3",
        temperature=0
    )

# Async wrapper for the LLM
async def aget_llm_response(llm, messages):
    """Async invoke wrapper"""
    try:
        response = await llm.ainvoke(messages)
        return response 
    except Exception as e :

        print(f"[ERROR] Error in async llm call !!! : {e}")
        raise 
     








