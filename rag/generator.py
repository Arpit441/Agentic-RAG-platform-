import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class Generator:
    """
    Generates answers from retrieved and compressed contexts using Groq's LLM gateway.
    Strictly enforces constraints on hallucinations and mandates citation formats.
    """
    def __init__(self, model_name: str = "llama-3.1-8b-instant", temperature: float = 0.0):
        """
        Initializes the Groq LLM chain. 
        Requires GROQ_API_KEY to be set in environment variables.
        """
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature
        )
        
        # System prompt enforcing strict RAG guidelines
        system_prompt = """You are a highly precise, enterprise-grade AI assistant.
Your task is to answer the user's question based strictly on the provided context.

CRITICAL RULES:
1. Answer ONLY using the provided context. Do not use any external knowledge.
2. You must cite your sources by referencing the chunk ID from which the information was drawn. Format your citations exactly like this: [chunk_1]. Place the citation immediately after the relevant sentence or claim.
3. If the information required to answer the query is NOT present in the provided context, you must ignore the query and reply exactly with: "Not found in provided documents"

Provided Context:
{context}
"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        
        # Build the LangChain execution chain
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generates an answer for a query based on the compressed context.
        
        Args:
            query (str): The user's question.
            context (str): The string context (which must contain chunk_id tags 
                           for the LLM to successfully cite them).
                           
        Returns:
            str: The LLM's formatted answer or the fallback failure string.
        """
        # If no context was retrieved or survived compression, fail fast
        if not context or not context.strip():
            return "Not found in provided documents"
            
        return self.chain.invoke({
            "query": query,
            "context": context
        })

    def stream_answer(self, query: str, context: str):
        """
        Streams the generated answer for real-time UI rendering.
        """
        if not context or not context.strip():
            yield "Not found in provided documents"
            return
            
        for chunk in self.chain.stream({
            "query": query,
            "context": context
        }):
            yield chunk
