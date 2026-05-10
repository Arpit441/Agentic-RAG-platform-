import json
from pathlib import Path
from typing import List, Dict, Any

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class IntentResult(BaseModel):
    """Pydantic model to enforce structured output for intent classification."""
    intent: str = Field(
        description="The intent of the query. Must be exactly one of: 'factual', 'analytical', 'summarization', 'multi-hop'"
    )

class IntentClassifier:
    """
    Classifies queries to intelligently adjust retrieval depth and breadth.
    """
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.llm = ChatGroq(model=model_name, temperature=0.0)
        self.parser = PydanticOutputParser(pydantic_object=IntentResult)
        
        template = """Classify the following user query into exactly one of these categories:
- factual: Simple fact retrieval (e.g., what, who, when)
- analytical: Requires comparing, contrasting, or deeper analysis
- summarization: Asking for a broad summary or overview of a topic
- multi-hop: Requires connecting multiple pieces of disparate information to answer

Query: {query}

{format_instructions}
"""
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chain = self.prompt | self.llm | self.parser

    def classify(self, query: str) -> str:
        try:
            result = self.chain.invoke({"query": query})
            intent = result.intent.lower()
            if intent not in ["factual", "analytical", "summarization", "multi-hop"]:
                return "factual"  # Safe fallback
            return intent
        except Exception:
            # Fallback in case of API failure or parsing error
            return "factual"


class HybridRetriever:
    """
    A hybrid retriever combining dense FAISS and sparse BM25 retrieval,
    enhanced with an LLM intent classifier to dynamically adjust retrieval depth.
    """
    def __init__(
        self,
        storage_dir: str = "storage",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.classifier = IntentClassifier()
        
        storage_path = Path(storage_dir)
        vector_store_path = storage_path / "vector_store"
        bm25_corpus_path = storage_path / "bm25_corpus.json"
        
        # 1. Initialize Dense Store
        if not (vector_store_path / "index.faiss").exists():
            # In a real environment, we'd raise FileNotFoundError. 
            # We initialize to None so the class can be imported if needed.
            self.vector_store = None
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            self.vector_store = FAISS.load_local(
                folder_path=str(vector_store_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True 
            )
        
        # 2. Initialize Sparse Store
        if not bm25_corpus_path.exists():
            self.bm25_data = []
            self.bm25 = None
        else:
            with open(bm25_corpus_path, "r", encoding="utf-8") as f:
                self.bm25_data = json.load(f)
            tokenized_corpus = [doc["text"].lower().split() for doc in self.bm25_data]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def _normalize(self, scores: List[float], invert: bool = False) -> List[float]:
        if not scores:
            return []
        min_val, max_val = min(scores), max(scores)
        if max_val == min_val:
            return [1.0] * len(scores)
            
        normalized = [(s - min_val) / (max_val - min_val) for s in scores]
        if invert:
            normalized = [1.0 - s for s in normalized]
        return normalized

    def retrieve(self, query: str, base_top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves top chunks using hybrid search, intelligently adjusting 
        the top_k and search pool size based on the classified query intent.
        """
        # 1. Classify Intent and Adjust Parameters
        intent = self.classifier.classify(query)
        print(f"[Retriever] Query classified as: {intent.upper()}")
        
        if intent == "factual":
            top_k = max(5, base_top_k)          # Facts don't need huge context
            pool_multiplier = 2                 # Shallow search pool
        elif intent == "analytical":
            top_k = max(15, base_top_k + 5)     # Analysis needs more chunks to compare
            pool_multiplier = 3                 # Deeper search pool
        elif intent == "summarization":
            top_k = max(20, base_top_k + 10)    # Summaries need maximum breadth
            pool_multiplier = 2
        elif intent == "multi-hop":
            top_k = max(15, base_top_k + 5)     # Multi-hop needs distinct pieces
            pool_multiplier = 4                 # Deepest search pool to catch hidden links
        else:
            top_k = base_top_k
            pool_multiplier = 2
            
        pool_size = max(top_k * pool_multiplier, 20)
        print(f"[Retriever] Adjusted parameters - Top K: {top_k}, Search Depth: {pool_size}")

        if not self.vector_store or not self.bm25:
            print("Indexes not found. Please run ingestion.")
            return []

        # 2. Sparse Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_sparse_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:pool_size]
        
        # 3. Dense Search (FAISS)
        dense_results = self.vector_store.similarity_search_with_score(query, k=pool_size)
        
        # 4. Score Aggregation
        sparse_scores_dict = {}
        for idx in top_sparse_indices:
            chunk_id = self.bm25_data[idx]["chunk_id"]
            sparse_scores_dict[chunk_id] = bm25_scores[idx]
            
        dense_scores_dict = {}
        chunk_map = {}
        
        for doc, distance in dense_results:
            chunk_id = doc.metadata.get("chunk_id")
            dense_scores_dict[chunk_id] = distance
            chunk_map[chunk_id] = {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            
        for idx in top_sparse_indices:
            chunk_id = self.bm25_data[idx]["chunk_id"]
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = {
                    "text": self.bm25_data[idx]["text"],
                    "metadata": {
                        "chunk_id": chunk_id,
                        "doc_id": self.bm25_data[idx].get("doc_id")
                    }
                }
                
        all_chunk_ids = list(set(sparse_scores_dict.keys()) | set(dense_scores_dict.keys()))
        
        raw_sparse = [sparse_scores_dict.get(cid, 0.0) for cid in all_chunk_ids]
        max_dense = max(dense_scores_dict.values()) if dense_scores_dict else 1.0
        raw_dense = [dense_scores_dict.get(cid, max_dense) for cid in all_chunk_ids]
        
        norm_sparse = self._normalize(raw_sparse, invert=False)
        norm_dense = self._normalize(raw_dense, invert=True) 
        
        final_results = []
        for i, chunk_id in enumerate(all_chunk_ids):
            combined_score = (self.dense_weight * norm_dense[i]) + (self.sparse_weight * norm_sparse[i])
            final_results.append({
                "text": chunk_map[chunk_id]["text"],
                "score": combined_score,
                "metadata": chunk_map[chunk_id]["metadata"]
            })
            
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:top_k]
