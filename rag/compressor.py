import re
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

class ContextCompressor:
    """
    Compresses retrieved contexts by filtering low-similarity chunks,
    removing redundant sentences, and enforcing a maximum token limit.
    """
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2", 
        chunk_sim_threshold: float = 0.05,
        redundancy_threshold: float = 0.90,
        max_tokens: int = 2000
    ):
        self.model = SentenceTransformer(model_name)
        self.chunk_sim_threshold = chunk_sim_threshold
        self.redundancy_threshold = redundancy_threshold
        self.max_tokens = max_tokens
        # Use tiktoken to enforce exact token limits matching common LLMs
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits text into sentences using basic regex boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    def compress(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Compresses a list of documents down to a single concise string.
        
        Args:
            query (str): The original search query.
            documents (List[Dict[str, Any]]): Retrieved documents containing 'text'.
            
        Returns:
            str: The compressed context string within the max token limit.
        """
        if not documents:
            return ""

        query_emb = self.model.encode([query])
        
        # Step 1: Remove low-similarity chunks
        valid_chunks = []
        for doc in documents:
            text = doc.get("text", "").strip()
            chunk_id = doc.get("metadata", {}).get("chunk_id", "unknown")
            if not text:
                continue
                
            chunk_emb = self.model.encode([text])
            sim = cosine_similarity(query_emb, chunk_emb)[0][0]
            
            if sim >= self.chunk_sim_threshold:
                valid_chunks.append({"text": text, "chunk_id": chunk_id})
                
        if not valid_chunks:
            return ""

        # Step 2: Extract all sentences from surviving chunks
        all_sentences = []
        for chunk in valid_chunks:
            sentences = self._split_into_sentences(chunk["text"])
            for s in sentences:
                all_sentences.append(f"{s} [{chunk['chunk_id']}]")
            
        if not all_sentences:
            return ""

        # Step 3: Embed sentences to evaluate relevance and redundancy
        sent_embs = self.model.encode(all_sentences)
        query_sims = cosine_similarity(query_emb, sent_embs)[0]
        
        # Sort sentences by their relevance to the query (highest first)
        scored_sentences = sorted(
            zip(all_sentences, sent_embs, query_sims), 
            key=lambda x: x[2], 
            reverse=True
        )

        final_sentences = []
        accepted_embs = []
        current_tokens = 0

        # Step 4: Remove redundant sentences & enforce token limit
        for sent, emb, q_sim in scored_sentences:
            sent_tokens = len(self.tokenizer.encode(sent))
            
            # Stop if we hit the token limit
            if current_tokens + sent_tokens > self.max_tokens:
                continue

            # Check for redundancy against already selected high-relevance sentences
            is_redundant = False
            if accepted_embs:
                emb_matrix = np.array([emb])
                accepted_matrix = np.array(accepted_embs)
                redundancy_sims = cosine_similarity(emb_matrix, accepted_matrix)[0]
                
                # If sentence is too similar to any already accepted sentence, skip it
                if np.any(redundancy_sims >= self.redundancy_threshold):
                    is_redundant = True
                    
            if not is_redundant:
                final_sentences.append(sent)
                accepted_embs.append(emb)
                current_tokens += sent_tokens

        # Note: final_sentences are currently ordered by relevance, which might
        # alter the chronological flow of the text, but provides the most
        # information-dense context for the LLM.
        return " ".join(final_sentences)
