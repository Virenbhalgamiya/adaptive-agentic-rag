"""
Agentic Adaptive RAG - LangGraph Workflow

Constructs the agent graph with conditional routing and self-correction loops.

Architecture:
    START → classify_query → [simple | complex | web_search]
    
    simple  → retrieve      → generate → grade_answer → [accept | retry]
    complex → decompose      → multi_retrieve → generate → grade_answer → [accept | retry]  
    web     → web_search     → generate → grade_answer → [accept | END]
    
    retry   → transform_query → classify_query (loop back)
    accept  → END
"""

import logging
from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    classify_query,
    retrieve,
    decompose_query,
    multi_retrieve,
    web_search,
    generate,
    grade_answer,
    transform_query,
)
from src.config import Config

logger = logging.getLogger(__name__)


def _route_after_classification(state: AgentState) -> str:
    """Route to the appropriate retrieval strategy based on query classification."""
    query_type = state.get("query_type", "simple")
    logger.info(f"🔀 Routing to: {query_type}")
    
    if query_type == "complex":
        return "decompose_query"
    elif query_type == "web_search":
        return "web_search"
    else:
        return "retrieve"


def _route_after_grading(state: AgentState) -> str:
    """Decide whether to accept the answer or retry with a transformed query."""
    is_grounded = state.get("is_grounded", True)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", Config.MAX_RETRIES)
    
    if is_grounded:
        logger.info("✅ Answer accepted — grounded and relevant")
        return END
    elif retry_count < max_retries:
        logger.info(f"🔄 Answer not grounded — retrying ({retry_count + 1}/{max_retries})")
        return "transform_query"
    else:
        logger.info("⚠️ Max retries reached — returning best available answer")
        return END


def build_graph() -> StateGraph:
    """Build and compile the agentic adaptive RAG graph.
    
    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    workflow = StateGraph(AgentState)
    
    # ── Add nodes ──
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("decompose_query", decompose_query)
    workflow.add_node("multi_retrieve", multi_retrieve)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_answer", grade_answer)
    workflow.add_node("transform_query", transform_query)
    
    # ── Set entry point ──
    workflow.set_entry_point("classify_query")
    
    # ── Conditional routing after classification ──
    workflow.add_conditional_edges(
        "classify_query",
        _route_after_classification,
        {
            "retrieve": "retrieve",
            "decompose_query": "decompose_query",
            "web_search": "web_search",
        },
    )
    
    # ── Simple path: retrieve → generate ──
    workflow.add_edge("retrieve", "generate")
    
    # ── Complex path: decompose → multi_retrieve → generate ──
    workflow.add_edge("decompose_query", "multi_retrieve")
    workflow.add_edge("multi_retrieve", "generate")
    
    # ── Web search path: web_search → generate ──
    workflow.add_edge("web_search", "generate")
    
    # ── All generation paths lead to grading ──
    workflow.add_edge("generate", "grade_answer")
    
    # ── Conditional routing after grading (accept or retry) ──
    workflow.add_conditional_edges(
        "grade_answer",
        _route_after_grading,
        {
            END: END,
            "transform_query": "transform_query",
        },
    )
    
    # ── Retry loop: transform → re-classify ──
    workflow.add_edge("transform_query", "classify_query")
    
    # ── Compile ──
    compiled = workflow.compile()
    logger.info("🏗️ Agent graph compiled successfully")
    
    return compiled


def run_agent(query: str) -> dict:
    """Run the agentic adaptive RAG pipeline on a query.
    
    Args:
        query: The user's natural language question.
        
    Returns:
        The final agent state containing the answer, steps, and metadata.
    """
    Config.validate()
    
    graph = build_graph()
    
    initial_state = {
        "query": query,
        "query_type": "",
        "sub_queries": [],
        "retrieved_docs": [],
        "web_results": [],
        "context": "",
        "generation": "",
        "is_grounded": False,
        "relevance_score": 0.0,
        "retry_count": 0,
        "max_retries": Config.MAX_RETRIES,
        "transformed_query": "",
        "route_reasoning": "",
        "grading_reasoning": "",
        "steps_taken": [],
    }
    
    logger.info(f"\n{'='*60}\n🚀 Running agent for query: '{query}'\n{'='*60}")
    
    result = graph.invoke(initial_state)
    
    logger.info(f"\n{'='*60}\n🏁 Agent finished — {len(result.get('steps_taken', []))} steps taken\n{'='*60}")
    
    return result
