"""
Agentic Adaptive RAG - FastAPI Application

Provides REST API endpoints for the agentic RAG system.
Includes query, ingestion, and stats endpoints.
"""

import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.agent.graph import run_agent
from src.ingestion.ingest import run_ingestion
from src.retrieval.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic Adaptive RAG",
    description="An intelligent RAG system with adaptive retrieval strategies for particle physics & ML research",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ──
class QueryRequest(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    query: str
    answer: str
    query_type: str
    route_reasoning: str
    is_grounded: bool
    relevance_score: float
    grading_reasoning: str
    steps_taken: list[str]
    retry_count: int
    processing_time_seconds: float

class IngestResponse(BaseModel):
    status: str
    papers_fetched: int
    chunks_created: int


# ── API Endpoints ──
@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a query through the agentic adaptive RAG pipeline."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    start_time = time.time()
    
    try:
        result = run_agent(request.query)
        processing_time = time.time() - start_time
        
        return QueryResponse(
            query=request.query,
            answer=result.get("generation", "No answer generated"),
            query_type=result.get("query_type", "unknown"),
            route_reasoning=result.get("route_reasoning", ""),
            is_grounded=result.get("is_grounded", False),
            relevance_score=result.get("relevance_score", 0.0),
            grading_reasoning=result.get("grading_reasoning", ""),
            steps_taken=result.get("steps_taken", []),
            retry_count=result.get("retry_count", 0),
            processing_time_seconds=round(processing_time, 2),
        )
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_endpoint():
    """Trigger the document ingestion pipeline."""
    try:
        result = run_ingestion()
        return IngestResponse(
            status=result.get("status", "unknown"),
            papers_fetched=result.get("papers_fetched", 0),
            chunks_created=result.get("chunks_created", 0),
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def stats_endpoint():
    """Get vector store statistics."""
    try:
        vs_manager = VectorStoreManager()
        return vs_manager.get_collection_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "agentic-adaptive-rag"}


# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the frontend application."""
    return FileResponse("frontend/index.html")
