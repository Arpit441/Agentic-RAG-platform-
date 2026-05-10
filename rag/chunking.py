import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SmartChunker:
    """
    A production-grade text chunker that uses a recursive strategy to split documents 
    into chunks while preserving sentence boundaries and managing token limits.
    """

    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 50):
        """
        Initialize the SmartChunker.
        
        Args:
            max_tokens (int): Maximum number of tokens per chunk.
            overlap_tokens (int): Number of tokens to overlap between chunks.
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be less than max_tokens")
            
        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self._count_tokens = lambda text: len(self.encoding.encode(text))
        except ImportError:
            logger.warning("tiktoken not found. Falling back to approximate word-based token counting.")
            # Rough approximation: 1 word ~ 1.3 tokens
            self._count_tokens = lambda text: int(len(text.split()) * 1.3)

    def _split_text(self, text: str, separators: List[str]) -> List[Dict[str, Any]]:
        """
        Recursively split text into smaller segments using a list of separators,
        keeping track of the exact start_index in the original text.
        """
        if not separators:
            return [{"text": text, "start_index": 0}]
            
        separator = separators[0]
        splits = []
        
        if separator == "":
            # Character level split fallback
            splits = [{"text": c, "start_index": i} for i, c in enumerate(text)]
            
        elif separator.startswith("regex:"):
            # Regex based split (e.g., sentence boundaries)
            pattern = separator.replace("regex:", "")
            last_end = 0
            for match in re.finditer(pattern, text):
                split_text = text[last_end:match.start()]
                if split_text:
                    splits.append({"text": split_text, "start_index": last_end})
                    
                # Append the separator whitespace to the previous split or as a new one
                if splits:
                    splits[-1]["text"] += match.group(0)
                else:
                    splits.append({"text": match.group(0), "start_index": last_end})
                last_end = match.end()
            
            if last_end < len(text):
                splits.append({"text": text[last_end:], "start_index": last_end})
                
        else:
            # String based split
            parts = text.split(separator)
            last_end = 0
            for i, part in enumerate(parts):
                segment = part + separator if i < len(parts) - 1 else part
                if segment:
                    splits.append({"text": segment, "start_index": last_end})
                last_end += len(segment)

        # Recursively process splits if they are too large
        final_splits = []
        for split in splits:
            if not split["text"]:
                continue
                
            split_tokens = self._count_tokens(split["text"])
            if split_tokens > self.max_tokens and len(separators) > 1:
                # Recurse down to the next separator
                sub_splits = self._split_text(split["text"], separators[1:])
                for sub_split in sub_splits:
                    final_splits.append({
                        "text": sub_split["text"],
                        "start_index": split["start_index"] + sub_split["start_index"]
                    })
            else:
                final_splits.append(split)
                
        return final_splits

    def chunk_document(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Chunk a document into smaller pieces based on token limits and overlap.
        
        Args:
            text (str): The document text to chunk.
            doc_id (str): The unique identifier for the document.
            
        Returns:
            List[Dict[str, Any]]: A list of chunk dictionaries containing metadata:
                - doc_id: Original document ID
                - chunk_id: Unique ID for this chunk
                - text: The chunk text
                - start_index: Start index of the chunk in the original text
        """
        if not text:
            return []
            
        # Hierarchical separators: Paragraphs -> Newlines -> Sentences -> Words -> Chars
        separators = ["\n\n", "\n", r"regex:(?<=[.!?])\s+", " ", ""]
        
        granular_splits = self._split_text(text, separators)
        
        chunks = []
        current_chunk_text = ""
        current_chunk_start = 0
        current_chunk_tokens = 0
        split_buffer = []
        chunk_counter = 0
        
        for split_obj in granular_splits:
            split_text = split_obj["text"]
            split_start = split_obj["start_index"]
            split_tokens = self._count_tokens(split_text)
            
            # If adding this split exceeds max_tokens, finalize the current chunk
            if current_chunk_tokens + split_tokens > self.max_tokens and current_chunk_text:
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{chunk_counter}",
                    "text": current_chunk_text.strip(),
                    "start_index": current_chunk_start
                })
                chunk_counter += 1
                
                # Calculate overlap for the new chunk
                overlap_text = ""
                overlap_tokens = 0
                overlap_start = split_start
                
                for buf_split in reversed(split_buffer):
                    buf_tokens = self._count_tokens(buf_split["text"])
                    if overlap_tokens + buf_tokens <= self.overlap_tokens:
                        overlap_text = buf_split["text"] + overlap_text
                        overlap_tokens += buf_tokens
                        overlap_start = buf_split["start_index"]
                    else:
                        break
                        
                current_chunk_text = overlap_text + split_text
                current_chunk_start = overlap_start
                current_chunk_tokens = overlap_tokens + split_tokens
                
                # Maintain the buffer starting from the overlap point
                split_buffer = [b for b in split_buffer if b["start_index"] >= overlap_start]
                split_buffer.append(split_obj)
            else:
                # Add to the current chunk
                if not current_chunk_text:
                    current_chunk_start = split_start
                current_chunk_text += split_text
                current_chunk_tokens += split_tokens
                split_buffer.append(split_obj)
                
        # Add the final chunk if any text remains
        if current_chunk_text.strip():
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{chunk_counter}",
                "text": current_chunk_text.strip(),
                "start_index": current_chunk_start
            })
            
        return chunks
