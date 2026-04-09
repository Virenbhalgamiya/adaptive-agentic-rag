"""
Agentic Adaptive RAG - Configuration Module

Centralizes all configuration settings loaded from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # OpenRouter Settings
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemma-3-27b-it:free")

    # Embedding Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ChromaDB Settings
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    CHROMA_COLLECTION_NAME: str = "arxiv_physics_ml"

    # Retrieval Settings
    TOP_K_RESULTS: int = 5
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Agent Settings
    MAX_RETRIES: int = 2

    # ArXiv Settings
    ARXIV_QUERIES: list[str] = [
        "machine learning particle physics CERN",
        "anomaly detection large hadron collider",
        "federated learning scientific computing",
        "graph neural networks particle reconstruction",
        "deep learning accelerator diagnostics",
    ]
    ARXIV_MAX_RESULTS_PER_QUERY: int = 5

    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required. Set it in .env file.")
        return True
