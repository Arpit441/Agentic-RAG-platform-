import time
from typing import Dict, Any, List

# Importing all modular components of the Enterprise RAG System
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import Reranker
from rag.compressor import ContextCompressor
from rag.generator import Generator
from rag.verifier import AnswerVerifier
from rag.evaluator import RAGEvaluator
from rag.feedback import FeedbackManager
from rag.memory import QueryDecontextualizer

class RAGOrchestrator:
    """
    The master orchestrator that wires together the entire Enterprise RAG pipeline,
    from intent classification all the way to evaluation and feedback logging.
    """
    def __init__(self):
        print("Initializing Enterprise RAG Pipeline Components...")
        self.retriever = HybridRetriever()
        self.reranker = Reranker()
        self.compressor = ContextCompressor()
        self.generator = Generator()
        self.verifier = AnswerVerifier()
        self.evaluator = RAGEvaluator()
        self.feedback_manager = FeedbackManager()
        self.memory = QueryDecontextualizer()
        self.last_run_data = None

    def run_stream(self, query: str, chat_history: List[Dict[str, str]] = None, top_k: int = 10, ground_truth: List[str] = None):
        """
        Executes the RAG pipeline with streaming. Yields string chunks.
        Saves telemetry to self.last_run_data after the stream finishes.
        """
        start_time = time.time()
        
        # 1. Memory Decontextualization
        standalone_query = self.memory.decontextualize(query, chat_history or [])
        
        # 2 & 3. Retrieve and Rerank
        retrieved_docs = self.retriever.retrieve(standalone_query, base_top_k=top_k)
        reranked_docs = self.reranker.rerank(standalone_query, retrieved_docs, top_k=top_k)
        
        # 4. Compress
        compressed_context = self.compressor.compress(standalone_query, reranked_docs)
        
        # 5. Generate (Stream)
        answer_chunks = []
        stream = self.generator.stream_answer(standalone_query, compressed_context)
        for chunk in stream:
            answer_chunks.append(chunk)
            yield chunk
            
        final_answer = "".join(answer_chunks)
        
        # 6. Verify
        verification_result = self.verifier.verify(final_answer, compressed_context)
        
        latency_sec = time.time() - start_time
        
        mock_token_usage = {
            "prompt_tokens": len(compressed_context.split()) * 1.5,
            "completion_tokens": len(final_answer.split()) * 1.5
        }
        
        # 7. Log Metrics
        metrics = self.evaluator.evaluate(
            query=standalone_query,
            answer=final_answer,
            retrieved_chunks=retrieved_docs,
            verifier_result=verification_result,
            latency_sec=latency_sec,
            token_usage=mock_token_usage,
            ground_truth=ground_truth
        )
        
        # 8. Feedback Logging
        hallucination_flag = not verification_result.get("faithful", True)
        avg_retrieval_score = 0.0
        if reranked_docs:
            avg_retrieval_score = sum(doc.get("rerank_score", 0.0) for doc in reranked_docs) / len(reranked_docs)
            
        self.feedback_manager.log_feedback(
            query=standalone_query,
            answer=final_answer,
            user_rating=0, 
            hallucination_flag=hallucination_flag,
            retrieval_score=avg_retrieval_score
        )
        
        # Save telemetry state
        self.last_run_data = {
            "standalone_query": standalone_query,
            "retrieved_docs": retrieved_docs,
            "reranked_docs": reranked_docs,
            "compressed_context": compressed_context,
            "answer": final_answer,
            "verification_result": verification_result,
            "metrics": metrics
        }
