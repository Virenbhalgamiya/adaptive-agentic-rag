"""
Agentic Adaptive RAG - Agent State Definition

Defines the state schema used by the LangGraph agent.
"""

from typing import TypedDict, Annotated
from langchain_core.documents import Document


class AgentState(TypedDict):
    """State that flows through the agentic RAG graph.
    
    Attributes:
        query: The original user query.
        query_type: Classification result - 'simple', 'complex', or 'web_search'.
        sub_queries: Decomposed sub-questions for complex queries.
        retrieved_docs: Documents retrieved from the vector store.
        web_results: Results from web search.
        context: Formatted context string for generation.
        generation: The generated answer.
        is_grounded: Whether the answer passed the grading check.
        relevance_score: Score from the answer grader (0.0-1.0).
        retry_count: Number of retry attempts made.
        max_retries: Maximum allowed retries.
        transformed_query: Reformulated query for retries.
        route_reasoning: Reasoning behind the routing decision.
        grading_reasoning: Reasoning from the answer grader.
        steps_taken: Log of steps taken by the agent for explainability.
    """
    query: str
    query_type: str
    sub_queries: list[str]
    retrieved_docs: list[Document]
    web_results: list[str]
    context: str
    generation: str
    is_grounded: bool
    relevance_score: float
    retry_count: int
    max_retries: int
    transformed_query: str
    route_reasoning: str
    grading_reasoning: str
    steps_taken: list[str]
