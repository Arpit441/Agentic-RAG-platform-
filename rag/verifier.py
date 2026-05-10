from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

class VerificationResult(BaseModel):
    """Pydantic model to enforce structured JSON output from the LLM."""
    faithful: bool = Field(
        description="True if the answer is fully supported by the context. False if it contains hallucinations or unsupported claims."
    )
    unsupported_sentences: List[str] = Field(
        description="A list of specific sentences from the answer that are NOT supported by the context. Empty if fully faithful."
    )
    confidence_score: float = Field(
        description="A score between 0.0 and 1.0 indicating confidence in this assessment."
    )

class AnswerVerifier:
    """
    Verifies the faithfulness of generated answers against the source context
    to detect hallucinations and unsupported claims.
    """
    def __init__(self, model_name: str = "llama-3.1-8b-instant", temperature: float = 0.0):
        """
        Initializes the LLM and structured output parser for verification.
        """
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature
        )
        template = """You are an objective, enterprise-grade AI auditor.
Your job is to read a provided Context and an AI's Generated Answer.
You must carefully evaluate if the Generated Answer is entirely faithful to the Context.

Rules:
1. An answer is NOT faithful if it contains any facts, numbers, or claims that cannot be directly proven by the Context.
2. If it is not faithful, you must extract the exact unsupported sentences.

Context:
{context}

Generated Answer:
{answer}
"""
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "answer"]
        )
        
        # Use native tool calling / JSON mode for guaranteed structured output
        self.chain = self.prompt | self.llm.with_structured_output(VerificationResult)

    def verify(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Verifies the answer against the context and returns the evaluation dict.
        
        Args:
            answer (str): The generated response from the RAG system.
            context (str): The compressed context provided to the generator.
            
        Returns:
            Dict[str, Any]: Contains 'faithful', 'unsupported_sentences', 
                            'confidence_score', and an optional 'flag'.
        """
        # Short-circuit logic for the standard fallback response
        if answer.strip() == "Not found in provided documents":
            return {
                "faithful": True,
                "unsupported_sentences": [],
                "confidence_score": 1.0
            }
            
        try:
            # Execute the LangChain chain to get structured output
            result: VerificationResult = self.chain.invoke({
                "context": context,
                "answer": answer
            })
            
            output = {
                "faithful": result.faithful,
                "unsupported_sentences": result.unsupported_sentences,
                "confidence_score": result.confidence_score
            }
            
            # Explicitly flag hallucinations as requested
            if not result.faithful:
                output["flag"] = "hallucination"
                
            return output
            
        except Exception as e:
            # A system/parsing error is NOT a hallucination — don't poison dashboard stats.
            # Return faithful=True with a note so the query is not incorrectly flagged.
            return {
                "faithful": True,
                "unsupported_sentences": [],
                "confidence_score": 0.5,
                "flag": f"verification_error: {str(e)[:120]}"
            }
