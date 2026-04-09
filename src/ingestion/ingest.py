"""
Agentic Adaptive RAG - Ingestion Pipeline

Orchestrates the full ingestion pipeline:
1. Fetch papers from arXiv
2. Chunk documents
3. Embed and store in ChromaDB
"""

import logging
from src.ingestion.arxiv_loader import ArxivLoader
from src.ingestion.chunker import TextChunker
from src.retrieval.vector_store import VectorStoreManager
from src.config import Config

logger = logging.getLogger(__name__)


def run_ingestion(
    queries: list[str] | None = None,
    max_results_per_query: int | None = None,
) -> dict:
    """Run the full ingestion pipeline.
    
    Args:
        queries: Custom search queries (defaults to Config.ARXIV_QUERIES).
        max_results_per_query: Max papers per query (defaults to Config).
        
    Returns:
        Statistics about the ingestion run.
    """
    queries = queries or Config.ARXIV_QUERIES
    max_results = max_results_per_query or Config.ARXIV_MAX_RESULTS_PER_QUERY
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Ingestion Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Fetch papers
    logger.info("\n📡 Step 1: Fetching papers from arXiv...")
    loader = ArxivLoader()
    documents = loader.fetch_multiple_queries(queries, max_results)
    
    if not documents:
        logger.error("❌ No documents fetched. Aborting ingestion.")
        return {"status": "failed", "reason": "No documents fetched"}
    
    # Step 2: Chunk documents
    logger.info("\n📦 Step 2: Chunking documents...")
    chunker = TextChunker()
    chunks = chunker.chunk_documents(documents)
    
    # Step 3: Store in vector database
    logger.info("\n💾 Step 3: Storing in ChromaDB...")
    vs_manager = VectorStoreManager()
    num_stored = vs_manager.add_documents(chunks)
    
    # Get final stats
    stats = vs_manager.get_collection_stats()
    
    result = {
        "status": "success",
        "papers_fetched": len(documents),
        "chunks_created": len(chunks),
        "chunks_stored": num_stored,
        "collection_stats": stats,
        "queries_used": queries,
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Ingestion Pipeline Complete!")
    logger.info(f"   📄 Papers fetched: {len(documents)}")
    logger.info(f"   📦 Chunks created: {len(chunks)}")
    logger.info(f"   💾 Total in store: {stats.get('document_count', 'Unknown')}")
    logger.info("=" * 60)
    
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ingestion()
