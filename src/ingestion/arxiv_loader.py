"""
Agentic Adaptive RAG - ArXiv Paper Loader

Downloads and processes research papers from arXiv for the knowledge base.
Focuses on particle physics + machine learning papers relevant to CERN.
"""

import logging
import arxiv
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ArxivLoader:
    """Downloads and processes arXiv papers."""
    
    def __init__(self):
        """Initialize the arXiv loader."""
        self.client = arxiv.Client()
    
    def fetch_papers(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[Document]:
        """Fetch papers from arXiv and convert to LangChain Documents.
        
        Uses paper abstracts and metadata as document content.
        This is faster and more reliable than full PDF download.
        
        Args:
            query: arXiv search query.
            max_results: Maximum papers to fetch per query.
            
        Returns:
            List of LangChain Document objects.
        """
        documents = []
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            
            results = list(self.client.results(search))
            
            for paper in results:
                # Build rich content from paper metadata + abstract
                content = self._format_paper(paper)
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": f"arXiv:{paper.entry_id}",
                        "title": paper.title,
                        "authors": ", ".join([a.name for a in paper.authors[:5]]),
                        "published": str(paper.published.date()) if paper.published else "Unknown",
                        "categories": ", ".join(paper.categories),
                        "arxiv_url": paper.entry_id,
                        "query": query,
                    },
                )
                documents.append(doc)
                logger.info(f"  📄 Fetched: {paper.title[:80]}...")
            
            logger.info(f"✅ Fetched {len(documents)} papers for query: '{query[:50]}...'")
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch papers for '{query}': {e}")
        
        return documents
    
    def _format_paper(self, paper) -> str:
        """Format a paper into a rich text representation."""
        authors = ", ".join([a.name for a in paper.authors[:10]])
        categories = ", ".join(paper.categories)
        published = str(paper.published.date()) if paper.published else "Unknown"
        
        content = f"""Title: {paper.title}

Authors: {authors}
Published: {published}
Categories: {categories}
ArXiv ID: {paper.entry_id}

Abstract:
{paper.summary}

---
This paper ({paper.title}) was published in the categories: {categories}.
Key topics covered include the research described in the abstract above.
"""
        return content
    
    def fetch_multiple_queries(
        self,
        queries: list[str],
        max_results_per_query: int = 5,
    ) -> list[Document]:
        """Fetch papers for multiple search queries.
        
        Args:
            queries: List of arXiv search queries.
            max_results_per_query: Max papers per query.
            
        Returns:
            Combined list of all fetched documents (deduplicated).
        """
        all_docs = []
        seen_ids = set()
        
        for i, query in enumerate(queries, 1):
            logger.info(f"\n📡 [{i}/{len(queries)}] Searching: '{query}'")
            docs = self.fetch_papers(query, max_results_per_query)
            
            for doc in docs:
                arxiv_id = doc.metadata.get("arxiv_url", "")
                if arxiv_id not in seen_ids:
                    seen_ids.add(arxiv_id)
                    all_docs.append(doc)
        
        logger.info(f"\n🎯 Total unique papers fetched: {len(all_docs)}")
        return all_docs
