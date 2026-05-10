import re
from typing import Dict, Any, List

class RAGEvaluator:
    """
    Evaluates execution runs of the Enterprise RAG pipeline against
    key performance, cost, and quality metrics.
    """
    def __init__(self, cost_per_1m_input: float = 0.05, cost_per_1m_output: float = 0.08):
        """
        Initializes the evaluator with pricing estimators.
        Defaults represent rough estimates for Llama3-8b via Groq.
        """
        self.cost_per_1m_input = cost_per_1m_input
        self.cost_per_1m_output = cost_per_1m_output

    def evaluate(self, 
                 query: str, 
                 answer: str, 
                 retrieved_chunks: List[Dict[str, Any]], 
                 verifier_result: Dict[str, Any], 
                 latency_sec: float, 
                 token_usage: Dict[str, int] = None,
                 ground_truth: Any = None) -> Dict[str, Any]:
        """
        Calculates core metrics for a RAG query execution.
        
        Args:
            query (str): The user's original query.
            answer (str): The generated answer.
            retrieved_chunks (list): The chunks retrieved by the HybridRetriever.
            verifier_result (dict): The output from the AnswerVerifier.
            latency_sec (float): Total pipeline execution time in seconds.
            token_usage (dict, optional): Dictionary with 'prompt_tokens' and 'completion_tokens'.
            ground_truth (list/str, optional): The expected true doc_ids or answer.
            
        Returns:
            Dict[str, Any]: The calculated metrics.
        """
        metrics = {}
        
        # 1. Latency
        metrics["latency_sec"] = round(latency_sec, 3)
        
        # 2. Answer Length (Words)
        metrics["answer_length_words"] = len(answer.split())
        
        # Parse sentences for granular metrics
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
        total_sentences = len(sentences) if sentences else 1

        # 3. Faithfulness Rate
        # Calculated based on the verifier's flagged unsupported sentences
        if verifier_result.get("faithful", False):
            metrics["faithfulness_rate"] = 1.0
        else:
            unsupported = verifier_result.get("unsupported_sentences", [])
            # Subtract the ratio of unsupported sentences from 1.0
            penalty = len(unsupported) / total_sentences
            metrics["faithfulness_rate"] = round(max(0.0, 1.0 - penalty), 2)
            
        # 4. Citation Coverage
        # Percentage of generated sentences that include a chunk citation bracket
        cited_sentences = sum(1 for sent in sentences if re.search(r'\[chunk_.*?\]', sent))
        metrics["citation_coverage"] = round(cited_sentences / total_sentences, 2)
        
        # 5. Cost Per Query
        cost = 0.0
        if token_usage:
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            cost = (prompt_tokens * self.cost_per_1m_input / 1_000_000) + \
                   (completion_tokens * self.cost_per_1m_output / 1_000_000)
        metrics["cost_per_query_usd"] = cost
        
        # 6. Retrieval Precision@K
        # Can only be fully calculated if ground_truth is provided as a list of correct doc_ids
        metrics["precision_at_k"] = None
        if ground_truth and isinstance(ground_truth, list):
            k = len(retrieved_chunks) if retrieved_chunks else 1
            retrieved_doc_ids = [c.get("metadata", {}).get("doc_id") for c in retrieved_chunks]
            
            hits = sum(1 for doc_id in retrieved_doc_ids if doc_id in ground_truth)
            metrics["precision_at_k"] = round(hits / k, 2)
            
        return metrics
