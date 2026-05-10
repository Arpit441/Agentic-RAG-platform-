from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

class QueryDecontextualizer:
    """
    Looks at the conversation history and rewrites the user's latest query
    so that it makes sense as a standalone search query (resolving pronouns
    like 'it', 'they', 'this').
    """
    def __init__(self, model_name: str = "llama-3.1-8b-instant", temperature: float = 0.0):
        self.llm = ChatGroq(model=model_name, temperature=temperature)
        
        template = """Given a chat history and the latest user query \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.

Chat History:
{chat_history}

Latest Query: {query}

Standalone Query:"""
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["chat_history", "query"]
        )
        self.chain = self.prompt | self.llm

    def decontextualize(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Takes the raw query and the Streamlit messages array and returns a standalone query.
        """
        if not chat_history:
            return query
            
        # Format the chat history into a readable string
        formatted_history = ""
        # Only take the last 4 messages to avoid blowing up context window unnecessarily
        recent_history = chat_history[-4:] 
        
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['content']}\n"
            
        try:
            result = self.chain.invoke({
                "chat_history": formatted_history,
                "query": query
            })
            # The result is an AIMessage object
            return result.content.strip()
        except Exception as e:
            # If the LLM fails, just fall back to the original query
            print(f"[Memory] Decontextualizer failed: {e}")
            return query
