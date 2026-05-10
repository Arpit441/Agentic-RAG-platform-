import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .chunking import SmartChunker

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Pipeline for ingesting text documents, chunking them, creating dense embeddings,
    and storing them in a FAISS vector index along with a raw chunk store for BM25.
    """
    def __init__(
        self, 
        embedding_model_name: str = "all-MiniLM-L6-v2", 
        storage_dir: str = "storage",
        chunk_max_tokens: int = 500,
        chunk_overlap_tokens: int = 50
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunker = SmartChunker(max_tokens=chunk_max_tokens, overlap_tokens=chunk_overlap_tokens)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
    def _read_text_files(self, folder_path: str) -> Dict[str, str]:
        """Read all .txt files recursively from a given folder."""
        documents = {}
        path = Path(folder_path)
        for file_path in path.rglob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    doc_id = str(file_path.relative_to(path))
                    documents[doc_id] = f.read()
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
        return documents

    def _read_pdf_files(self, folder_path: str) -> Dict[str, str]:
        """Read all .pdf files recursively using PyMuPDF (embedded text only, no OCR)."""
        documents = {}
        path = Path(folder_path)
        try:
            import fitz  # pymupdf
        except ImportError:
            logger.warning("pymupdf not installed. Skipping PDF files. Run: pip install pymupdf")
            return documents

        for file_path in path.rglob("*.pdf"):
            try:
                doc = fitz.open(str(file_path))
                full_text = ""
                for page in doc:
                    full_text += page.get_text()
                doc.close()
                if full_text.strip():
                    doc_id = str(file_path.relative_to(path))
                    documents[doc_id] = full_text
                    logger.info(f"Extracted {len(full_text)} chars from PDF: {file_path.name}")
                else:
                    logger.warning(f"No text extracted from {file_path.name} — may be a scanned PDF.")
            except Exception as e:
                logger.error(f"Error reading PDF {file_path}: {e}")
        return documents


    def ingest_documents(self, folder_path: str):
        # Read both .txt and .pdf files
        documents = self._read_text_files(folder_path)
        pdf_docs = self._read_pdf_files(folder_path)
        documents.update(pdf_docs)  # merge together

        if not documents:
            print("No .txt or .pdf files found in the folder.")
            return
            
        all_chunks = []
        for doc_id, text in documents.items():
            chunks = self.chunker.chunk_document(text, doc_id)
            all_chunks.extend(chunks)
            
        if not all_chunks:
            return
            
        # 1. Prepare for LangChain FAISS
        texts_for_vectorstore = [chunk["text"] for chunk in all_chunks]
        metadatas_for_vectorstore = [{
            "chunk_id": chunk["chunk_id"], 
            "doc_id": chunk["doc_id"], 
            "start_index": chunk["start_index"]
        } for chunk in all_chunks]
        
        # 2. Build & Save FAISS Index
        vector_store_path = self.storage_dir / "vector_store"
        vector_store = FAISS.from_texts(
            texts=texts_for_vectorstore,
            embedding=self.embeddings,
            metadatas=metadatas_for_vectorstore
        )
        vector_store.save_local(str(vector_store_path))
        
        # 3. Save Raw Chunks for BM25 (Required by HybridRetriever)
        bm25_corpus_path = self.storage_dir / "bm25_corpus.json"
        with open(bm25_corpus_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2)
            
        # Optional raw chunk dump
        chunks_path = self.storage_dir / "raw_chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2)

        print(f"Ingested {len(all_chunks)} chunks successfully.")
