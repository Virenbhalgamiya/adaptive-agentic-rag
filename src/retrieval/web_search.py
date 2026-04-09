"""
Agentic Adaptive RAG - Web Search Manager

Provides web search fallback using DuckDuckGo when the local
knowledge base doesn't contain relevant information.
"""

import logging
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


class WebSearchManager:
    """Manages web search operations via DuckDuckGo."""
    
    def __init__(self):
        """Initialize the web search manager."""
        self.ddgs = DDGS()
    
    def search(self, query: str, max_results: int = 5) -> list[str]:
        """Search the web and return formatted results.
        
        Args:
            query: The search query.
            max_results: Maximum number of results to return.
            
        Returns:
            List of formatted search result strings.
        """
        try:
            results = list(self.ddgs.text(query, max_results=max_results))
            
            formatted = []
            for r in results:
                title = r.get("title", "No title")
                body = r.get("body", "No description")
                href = r.get("href", "")
                formatted.append(f"Title: {title}\nURL: {href}\nContent: {body}")
            
            logger.info(f"🌐 Web search returned {len(formatted)} results for: '{query[:60]}...'")
            return formatted
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return [f"Web search failed: {str(e)}"]
