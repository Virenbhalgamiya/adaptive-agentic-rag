"""
Agentic Adaptive RAG - Agent Node Functions

Each function represents a node in the LangGraph state machine.
Nodes process the agent state and return updates.
"""

import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.agent.state import AgentState
from src.agent.prompts import (
    QUERY_ROUTER_PROMPT,
    QUERY_DECOMPOSITION_PROMPT,
    GENERATION_PROMPT,
    ANSWER_GRADER_PROMPT,
    QUERY_TRANSFORM_PROMPT,
    WEB_SEARCH_SYNTHESIS_PROMPT,
)
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.web_search import WebSearchManager
from src.config import Config

logger = logging.getLogger(__name__)


def _get_llm() -> ChatOpenAI:
    """Create an OpenRouter-backed LLM instance."""
    return ChatOpenAI(
        model=Config.OPENROUTER_MODEL,
        openai_api_key=Config.OPENROUTER_API_KEY,
        openai_api_base=Config.OPENROUTER_BASE_URL,
        temperature=0.1,
        max_tokens=2048,
    )


def _safe_parse_json(text: str) -> dict:
    """Safely parse JSON from LLM output, handling markdown code blocks."""
    text = text.strip()
    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning(f"Failed to parse JSON from LLM output: {text[:200]}")
        return {}


# ─────────────────────────────────────────────
# Node: Classify Query
# ─────────────────────────────────────────────
def classify_query(state: AgentState) -> dict:
    """Classify the query as simple, complex, or requiring web search."""
    logger.info("🔍 Classifying query...")
    
    llm = _get_llm()
    query = state.get("transformed_query") or state["query"]
    
    prompt = QUERY_ROUTER_PROMPT.format(query=query)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    result = _safe_parse_json(response.content)
    query_type = result.get("query_type", "simple")
    reasoning = result.get("reasoning", "No reasoning provided")
    
    # Validate query_type
    if query_type not in ["simple", "complex", "web_search"]:
        query_type = "simple"
    
    logger.info(f"📋 Query classified as: {query_type} — {reasoning}")
    
    return {
        "query_type": query_type,
        "route_reasoning": reasoning,
        "steps_taken": state.get("steps_taken", []) + [
            f"[Router] Classified as '{query_type}': {reasoning}"
        ],
    }


# ─────────────────────────────────────────────
# Node: Simple Retrieval
# ─────────────────────────────────────────────
def retrieve(state: AgentState) -> dict:
    """Retrieve relevant documents from the vector store."""
    logger.info("📚 Retrieving documents...")
    
    query = state.get("transformed_query") or state["query"]
    vs_manager = VectorStoreManager()
    docs = vs_manager.similarity_search(query, k=Config.TOP_K_RESULTS)
    
    # Format context from retrieved documents
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", "Untitled")
        context_parts.append(f"[Document {i}] Title: {title}\nSource: {source}\n{doc.page_content}\n")
    
    context = "\n---\n".join(context_parts)
    
    logger.info(f"📄 Retrieved {len(docs)} documents")
    
    return {
        "retrieved_docs": docs,
        "context": context,
        "steps_taken": state.get("steps_taken", []) + [
            f"[Retrieval] Retrieved {len(docs)} documents for: '{query[:80]}...'"
        ],
    }


# ─────────────────────────────────────────────
# Node: Decompose Complex Query
# ─────────────────────────────────────────────
def decompose_query(state: AgentState) -> dict:
    """Break a complex query into sub-questions."""
    logger.info("🧩 Decomposing complex query...")
    
    llm = _get_llm()
    query = state.get("transformed_query") or state["query"]
    
    prompt = QUERY_DECOMPOSITION_PROMPT.format(query=query)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    result = _safe_parse_json(response.content)
    sub_queries = result.get("sub_queries", [query])
    
    if not sub_queries:
        sub_queries = [query]
    
    logger.info(f"🔀 Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
    
    return {
        "sub_queries": sub_queries,
        "steps_taken": state.get("steps_taken", []) + [
            f"[Decomposition] Split into {len(sub_queries)} sub-queries: {sub_queries}"
        ],
    }


# ─────────────────────────────────────────────
# Node: Multi-hop Retrieval (for complex queries)
# ─────────────────────────────────────────────
def multi_retrieve(state: AgentState) -> dict:
    """Retrieve documents for each sub-query and merge results."""
    logger.info("📚 Multi-hop retrieval...")
    
    vs_manager = VectorStoreManager()
    all_docs = []
    seen_contents = set()
    
    for sq in state["sub_queries"]:
        docs = vs_manager.similarity_search(sq, k=3)
        for doc in docs:
            # Deduplicate
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_docs.append(doc)
    
    # Format context
    context_parts = []
    for i, doc in enumerate(all_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", "Untitled")
        context_parts.append(f"[Document {i}] Title: {title}\nSource: {source}\n{doc.page_content}\n")
    
    context = "\n---\n".join(context_parts)
    
    logger.info(f"📄 Multi-retrieval gathered {len(all_docs)} unique documents")
    
    return {
        "retrieved_docs": all_docs,
        "context": context,
        "steps_taken": state.get("steps_taken", []) + [
            f"[Multi-Retrieval] Gathered {len(all_docs)} unique docs across {len(state['sub_queries'])} sub-queries"
        ],
    }


# ─────────────────────────────────────────────
# Node: Web Search
# ─────────────────────────────────────────────
def web_search(state: AgentState) -> dict:
    """Search the web when local knowledge base is insufficient."""
    logger.info("🌐 Performing web search...")
    
    query = state.get("transformed_query") or state["query"]
    ws_manager = WebSearchManager()
    results = ws_manager.search(query, max_results=5)
    
    # Format web results as context
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(f"[Web Result {i}] {result}")
    
    context = "\n---\n".join(context_parts)
    
    logger.info(f"🌐 Found {len(results)} web results")
    
    return {
        "web_results": results,
        "context": context,
        "steps_taken": state.get("steps_taken", []) + [
            f"[Web Search] Found {len(results)} results for: '{query[:80]}...'"
        ],
    }


# ─────────────────────────────────────────────
# Node: Generate Answer
# ─────────────────────────────────────────────
def generate(state: AgentState) -> dict:
    """Generate an answer using the retrieved context."""
    logger.info("✍️ Generating answer...")
    
    llm = _get_llm()
    query = state["query"]
    context = state.get("context", "No context available.")
    
    # Use web search prompt if we came from web search
    if state.get("query_type") == "web_search":
        prompt = WEB_SEARCH_SYNTHESIS_PROMPT.format(web_results=context, query=query)
    else:
        prompt = GENERATION_PROMPT.format(context=context, query=query)
    
    response = llm.invoke([HumanMessage(content=prompt)])
    generation = response.content
    
    logger.info(f"✅ Answer generated ({len(generation)} chars)")
    
    return {
        "generation": generation,
        "steps_taken": state.get("steps_taken", []) + [
            f"[Generation] Generated answer ({len(generation)} chars)"
        ],
    }


# ─────────────────────────────────────────────
# Node: Grade Answer
# ─────────────────────────────────────────────
def grade_answer(state: AgentState) -> dict:
    """Grade the answer for groundedness and relevance."""
    logger.info("🔎 Grading answer...")
    
    # Skip grading for web search results (no ground truth context)
    if state.get("query_type") == "web_search":
        logger.info("⏭️ Skipping grading for web search results")
        return {
            "is_grounded": True,
            "relevance_score": 0.8,
            "grading_reasoning": "Web search results - grading skipped",
            "steps_taken": state.get("steps_taken", []) + [
                "[Grader] Skipped for web search results (no ground truth)"
            ],
        }
    
    llm = _get_llm()
    prompt = ANSWER_GRADER_PROMPT.format(
        context=state.get("context", ""),
        query=state["query"],
        answer=state["generation"],
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    result = _safe_parse_json(response.content)
    
    is_grounded = result.get("is_grounded", True)
    relevance_score = float(result.get("relevance_score", 0.7))
    reasoning = result.get("reasoning", "No reasoning provided")
    
    logger.info(f"📊 Grading: grounded={is_grounded}, relevance={relevance_score:.2f} — {reasoning}")
    
    return {
        "is_grounded": is_grounded,
        "relevance_score": relevance_score,
        "grading_reasoning": reasoning,
        "retry_count": state.get("retry_count", 0),
        "steps_taken": state.get("steps_taken", []) + [
            f"[Grader] Grounded: {is_grounded}, Relevance: {relevance_score:.2f} — {reasoning}"
        ],
    }


# ─────────────────────────────────────────────
# Node: Transform Query (for retries)
# ─────────────────────────────────────────────
def transform_query(state: AgentState) -> dict:
    """Reformulate the query when the previous attempt was unsatisfactory."""
    logger.info("🔄 Transforming query for retry...")
    
    llm = _get_llm()
    query = state.get("transformed_query") or state["query"]
    
    prompt = QUERY_TRANSFORM_PROMPT.format(query=query)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    result = _safe_parse_json(response.content)
    transformed = result.get("transformed_query", query)
    strategy = result.get("strategy", "Unknown transformation")
    
    new_retry_count = state.get("retry_count", 0) + 1
    logger.info(f"🔄 Retry {new_retry_count}: '{query}' → '{transformed}' ({strategy})")
    
    return {
        "transformed_query": transformed,
        "retry_count": new_retry_count,
        "steps_taken": state.get("steps_taken", []) + [
            f"[Transform] Retry {new_retry_count}: '{transformed}' ({strategy})"
        ],
    }
