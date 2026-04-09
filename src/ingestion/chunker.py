"""
Agentic Adaptive RAG - Text Chunker

Splits documents into overlapping chunks for optimal retrieval.
"""

import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import Config

logger = logging.getLogger(__name__)


class TextChunker:
    """Splits documents into overlapping chunks for vector store ingestion."""
    
    def __init__(
        self,
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP,
    ):
        """Initialize the text chunker.
        
        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    
    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks, preserving metadata.
        
        Args:
            documents: List of documents to chunk.
            
        Returns:
            List of chunked documents with original metadata preserved.
        """
        if not documents:
            return []
        
        chunks = self.splitter.split_documents(documents)
        
        logger.info(
            f"📦 Chunked {len(documents)} documents → {len(chunks)} chunks "
            f"(chunk_size={self.splitter._chunk_size}, overlap={self.splitter._chunk_overlap})"
        )
        
        return chunks
