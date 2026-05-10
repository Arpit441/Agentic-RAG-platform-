import numpy as np
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class Reranker:
    """
    Reranks a list of retrieved documents using a Cross-Encoder model.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the reranker with the specified cross-encoder model.
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks documents by scoring query-document pairs using the cross-encoder.
        
        Args:
            query (str): The search query.
            documents (List[Dict[str, Any]]): Retrieved documents containing a 'text' key.
            top_k (int): Number of top documents to return after reranking.
            
        Returns:
            List[Dict[str, Any]]: The top_k reranked documents with their new scores.
        """
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            # Safely extract text; default to empty string if missing
            text = doc.get("text", "")
            pairs.append((query, text))

        # Generate cross-encoder similarity scores
        # Note: predict() returns raw logits (can be negative)
        scores = self.model.predict(pairs)

        # Normalize scores to [0, 1] using sigmoid so the dashboard displays sensibly
        scores_normalized = 1 / (1 + np.exp(-np.array(scores)))

        # Associate scores with documents
        reranked_docs = []
        for doc, score in zip(documents, scores_normalized):
            doc_copy = dict(doc)
            # Add the normalized reranking score (0=irrelevant, 1=perfect match)
            doc_copy["rerank_score"] = float(score)
            reranked_docs.append(doc_copy)

        # Sort documents descending based on the cross-encoder score
        reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked_docs[:top_k]
