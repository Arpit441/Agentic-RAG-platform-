import sqlite3
import os
from pathlib import Path
from typing import List, Dict, Any

class FeedbackManager:
    """
    Manages telemetry, user feedback, and RAG evaluation metrics using a local SQLite database.
    Provides analytical methods to automatically recommend system adjustments.
    """
    def __init__(self, db_path: str = "storage/feedback.db"):
        self.db_path = db_path
        
        # Ensure the directory exists before connecting to SQLite
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database schema if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    user_rating INTEGER,
                    hallucination_flag BOOLEAN,
                    retrieval_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def log_feedback(self, query: str, answer: str, user_rating: int, hallucination_flag: bool, retrieval_score: float):
        """Logs a single RAG execution cycle into the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO feedback (query, answer, user_rating, hallucination_flag, retrieval_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (query, answer, user_rating, hallucination_flag, retrieval_score))
            conn.commit()

    def get_low_performing_queries(self, rating_threshold: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves a list of queries that performed poorly, either via explicit user
        rating or internal hallucination flagging.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Fetch records with a low star rating or where a hallucination was detected
            cursor.execute('''
                SELECT id, query, answer, user_rating, hallucination_flag, retrieval_score, timestamp 
                FROM feedback 
                WHERE user_rating <= ? OR hallucination_flag = 1
                ORDER BY timestamp DESC
            ''', (rating_threshold,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def adjust_chunk_size_recommendation(self) -> str:
        """
        Analyzes historical feedback to automatically recommend adjustments 
        to the chunking strategy.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) as total FROM feedback')
            total = cursor.fetchone()['total']
            
            # Need a baseline of queries to make a statistical recommendation
            if total < 5:
                return "Not enough data to make a recommendation. (Need at least 5 records)"
                
            cursor.execute('''
                SELECT 
                    AVG(retrieval_score) as avg_retrieval,
                    SUM(CASE WHEN hallucination_flag = 1 THEN 1 ELSE 0 END)*1.0 / COUNT(*) as hallucination_rate,
                    AVG(user_rating) as avg_rating
                FROM feedback
            ''')
            stats = cursor.fetchone()
            
            avg_retrieval = stats['avg_retrieval'] or 0.0
            hallucination_rate = stats['hallucination_rate'] or 0.0
            avg_rating = stats['avg_rating'] or 0.0
            
            # Diagnostic heuristics
            if hallucination_rate > 0.3 and avg_retrieval < 0.5:
                return "Recommendation: DECREASE chunk size. The retriever is struggling to find precise matches, leading to noisy context and high hallucination rates."
                
            elif avg_retrieval < 0.4 and hallucination_rate <= 0.15:
                return "Recommendation: INCREASE chunk size or overlap. Retrieval precision is low, meaning semantic concepts might be getting split across chunk boundaries."
                
            elif hallucination_rate > 0.2 and avg_retrieval > 0.7:
                return "Recommendation: INCREASE compressor strictness. The retriever is finding relevant chunks, but the LLM is confused by secondary noise. Filter aggressively before generation."
                
            elif avg_rating > 3.5 and hallucination_rate < 0.1:
                return "Recommendation: System is HEALTHY. Current chunk sizes and overlaps are performing optimally."
                
            return "Recommendation: Monitor trends. Data is inconclusive for a definitive chunk size adjustment."

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """
        Retrieves aggregate statistics for the Admin Dashboard.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(latency) as avg_latency,
                    SUM(CASE WHEN hallucination_flag = 1 THEN 1 ELSE 0 END)*100.0 / COUNT(*) as hallucination_percentage,
                    AVG(retrieval_score) as avg_retrieval_score
                FROM (
                    SELECT 
                        id, hallucination_flag, retrieval_score,
                        -- We do not have latency in the db currently, so we just return 0 for it
                        0 as latency
                    FROM feedback
                )
            ''')
            row = cursor.fetchone()
            
            if not row or row['total_queries'] == 0:
                return {
                    "total_queries": 0,
                    "avg_latency": 0.0,
                    "hallucination_percentage": 0.0,
                    "avg_retrieval_score": 0.0
                }
                
            return {
                "total_queries": row['total_queries'],
                "avg_latency": round(row['avg_latency'] or 0.0, 2),
                "hallucination_percentage": round(row['hallucination_percentage'] or 0.0, 2),
                "avg_retrieval_score": round(row['avg_retrieval_score'] or 0.0, 2)
            }

    def clear_all_feedback(self):
        """Wipes all feedback records — useful for resetting stale/corrupt test data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM feedback")
            conn.commit()
