"""
Agentic Adaptive RAG - Vector Store Manager

Manages ChromaDB for document storage and semantic similarity search.
Uses HuggingFace sentence-transformers for local embeddings.
"""

import logging
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.config import Config

logger = logging.getLogger(__name__)

# Module-level singleton for embeddings (avoid re-loading model)
_embeddings = None
_vectorstore = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create the embeddings model (singleton)."""
    global _embeddings
    if _embeddings is None:
        logger.info(f"📥 Loading embedding model: {Config.EMBEDDING_MODEL}...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("✅ Embedding model loaded")
    return _embeddings


class VectorStoreManager:
    """Manages ChromaDB vector store operations."""
    
    def __init__(self):
        """Initialize with persistent ChromaDB storage."""
        global _vectorstore
        if _vectorstore is None:
            _vectorstore = Chroma(
                collection_name=Config.CHROMA_COLLECTION_NAME,
                embedding_function=_get_embeddings(),
                persist_directory=Config.CHROMA_PERSIST_DIR,
            )
        self.vectorstore = _vectorstore
    
    def add_documents(self, documents: list[Document]) -> int:
        """Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects.
            
        Returns:
            Number of documents added.
        """
        if not documents:
            logger.warning("No documents to add")
            return 0
        
        self.vectorstore.add_documents(documents)
        logger.info(f"📥 Added {len(documents)} documents to vector store")
        return len(documents)
    
    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """Search for similar documents.
        
        Args:
            query: The search query.
            k: Number of results to return.
            
        Returns:
            List of most similar documents.
        """
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"🔍 Found {len(results)} similar documents for: '{query[:60]}...'")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the vector store collection."""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {
                "collection_name": Config.CHROMA_COLLECTION_NAME,
                "document_count": count,
                "persist_directory": Config.CHROMA_PERSIST_DIR,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
