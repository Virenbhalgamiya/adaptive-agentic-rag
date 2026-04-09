"""
Agentic Adaptive RAG - Demo Runner & Results Generator

Runs the agent on curated demo queries, captures results, and generates
metrics/visualizations for the README.
"""

import json
import time
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent.graph import run_agent
from src.ingestion.ingest import run_ingestion
from src.retrieval.vector_store import VectorStoreManager
from src.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Demo Queries (designed to trigger different strategies) ──
DEMO_QUERIES = [
    {
        "query": "What is the Higgs boson and how was it discovered at CERN?",
        "expected_type": "simple",
        "description": "Simple factual query — single retrieval pass",
    },
    {
        "query": "Compare the effectiveness of Graph Neural Networks versus Convolutional Neural Networks for particle jet classification, and discuss their computational trade-offs in high-energy physics experiments.",
        "expected_type": "complex",
        "description": "Complex analytical query — decomposition + multi-hop retrieval",
    },
    {
        "query": "How has federated learning been applied to privacy-preserving data analysis in particle physics experiments across different institutions?",
        "expected_type": "complex",
        "description": "Complex domain query — multi-faceted retrieval",
    },
    {
        "query": "What anomaly detection techniques are used in the Large Hadron Collider for identifying rare particle decay events?",
        "expected_type": "simple",
        "description": "Domain-specific factual query — targeted retrieval",
    },
    {
        "query": "What are the latest developments in quantum computing applications at CERN in 2025?",
        "expected_type": "web_search",
        "description": "Recency-dependent query — web search fallback",
    },
]


def run_demo() -> dict:
    """Run the full demo: ingest papers, run queries, collect results."""
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # ── Step 1: Check if ingestion is needed ──
    vs_manager = VectorStoreManager()
    stats = vs_manager.get_collection_stats()
    doc_count = stats.get("document_count", 0)
    
    if doc_count == 0:
        logger.info("\n📡 No documents in store. Running ingestion first...\n")
        ingestion_result = run_ingestion()
        logger.info(f"\n✅ Ingestion complete: {ingestion_result}\n")
    else:
        logger.info(f"\n📚 Found {doc_count} documents in store. Skipping ingestion.\n")
        ingestion_result = {"status": "skipped", "existing_documents": doc_count}
    
    # ── Step 2: Run demo queries ──
    all_results = []
    total_start = time.time()
    
    for i, demo in enumerate(DEMO_QUERIES, 1):
        query = demo["query"]
        logger.info(f"\n{'='*70}")
        logger.info(f"🧪 Demo Query {i}/{len(DEMO_QUERIES)}: {demo['description']}")
        logger.info(f"   Query: {query}")
        logger.info(f"   Expected route: {demo['expected_type']}")
        logger.info(f"{'='*70}\n")
        
        start = time.time()
        
        try:
            result = run_agent(query)
            elapsed = time.time() - start
            
            query_result = {
                "query_number": i,
                "query": query,
                "description": demo["description"],
                "expected_type": demo["expected_type"],
                "actual_type": result.get("query_type", "unknown"),
                "route_correct": result.get("query_type") == demo["expected_type"],
                "answer": result.get("generation", "No answer generated"),
                "is_grounded": result.get("is_grounded", False),
                "relevance_score": result.get("relevance_score", 0.0),
                "retry_count": result.get("retry_count", 0),
                "steps_taken": result.get("steps_taken", []),
                "num_steps": len(result.get("steps_taken", [])),
                "route_reasoning": result.get("route_reasoning", ""),
                "grading_reasoning": result.get("grading_reasoning", ""),
                "processing_time_seconds": round(elapsed, 2),
            }
            
            all_results.append(query_result)
            
            logger.info(f"\n📊 Result: type={query_result['actual_type']}, "
                       f"grounded={query_result['is_grounded']}, "
                       f"relevance={query_result['relevance_score']:.2f}, "
                       f"time={elapsed:.2f}s\n")
            
        except Exception as e:
            logger.error(f"❌ Query {i} failed: {e}")
            all_results.append({
                "query_number": i,
                "query": query,
                "description": demo["description"],
                "error": str(e),
                "processing_time_seconds": round(time.time() - start, 2),
            })
    
    total_elapsed = time.time() - total_start
    
    # ── Step 3: Compute aggregate metrics ──
    successful = [r for r in all_results if "error" not in r]
    
    metrics = {
        "total_queries": len(DEMO_QUERIES),
        "successful_queries": len(successful),
        "failed_queries": len(all_results) - len(successful),
        "routing_accuracy": (
            sum(1 for r in successful if r.get("route_correct", False)) / len(successful) * 100
            if successful else 0
        ),
        "avg_relevance_score": (
            sum(r.get("relevance_score", 0) for r in successful) / len(successful)
            if successful else 0
        ),
        "grounding_rate": (
            sum(1 for r in successful if r.get("is_grounded", False)) / len(successful) * 100
            if successful else 0
        ),
        "avg_processing_time": (
            sum(r.get("processing_time_seconds", 0) for r in successful) / len(successful)
            if successful else 0
        ),
        "avg_steps_per_query": (
            sum(r.get("num_steps", 0) for r in successful) / len(successful)
            if successful else 0
        ),
        "total_time_seconds": round(total_elapsed, 2),
    }
    
    # ── Step 4: Save results ──
    full_results = {
        "metadata": {
            "model": Config.OPENROUTER_MODEL,
            "embedding_model": Config.EMBEDDING_MODEL,
            "max_retries": Config.MAX_RETRIES,
            "top_k": Config.TOP_K_RESULTS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "ingestion": ingestion_result,
        "metrics": metrics,
        "queries": all_results,
    }
    
    # Save JSON results
    results_path = os.path.join(results_dir, "demo_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n💾 Results saved to: {results_path}")
    
    # Generate markdown table for README
    _generate_results_markdown(metrics, all_results, results_dir)
    
    # Print summary
    _print_summary(metrics, all_results)
    
    return full_results


def _generate_results_markdown(metrics: dict, results: list, results_dir: str):
    """Generate a markdown file with results formatted for README."""
    
    md = """# 📊 Evaluation Results

## Aggregate Metrics

| Metric | Value |
|--------|-------|
| Total Queries | {total_queries} |
| Successful Queries | {successful_queries} |
| Routing Accuracy | {routing_accuracy:.1f}% |
| Average Relevance Score | {avg_relevance_score:.2f} |
| Grounding Rate | {grounding_rate:.1f}% |
| Avg Processing Time | {avg_processing_time:.2f}s |
| Avg Steps per Query | {avg_steps_per_query:.1f} |

## Per-Query Results

| # | Strategy | Grounded | Relevance | Retries | Time | Route Match |
|---|----------|----------|-----------|---------|------|-------------|
""".format(**metrics)
    
    for r in results:
        if "error" in r:
            md += f"| {r['query_number']} | ❌ Error | - | - | - | {r['processing_time_seconds']:.1f}s | - |\n"
        else:
            grounded = "✅" if r.get("is_grounded") else "❌"
            route_match = "✅" if r.get("route_correct") else "❌"
            md += (
                f"| {r['query_number']} | {r.get('actual_type', 'unknown')} | "
                f"{grounded} | {r.get('relevance_score', 0):.2f} | "
                f"{r.get('retry_count', 0)} | {r['processing_time_seconds']:.1f}s | {route_match} |\n"
            )
    
    md += "\n## Query Details\n\n"
    
    for r in results:
        if "error" in r:
            continue
        md += f"### Query {r['query_number']}: {r['description']}\n\n"
        md += f"**Query:** {r['query']}\n\n"
        md += f"**Strategy:** `{r.get('actual_type', 'unknown')}` (expected: `{r.get('expected_type', 'unknown')}`)\n\n"
        md += f"**Answer:**\n> {r.get('answer', 'N/A')[:500]}{'...' if len(r.get('answer', '')) > 500 else ''}\n\n"
        
        steps = r.get("steps_taken", [])
        if steps:
            md += "**Agent Steps:**\n"
            for step in steps:
                md += f"1. `{step}`\n"
            md += "\n"
        
        md += "---\n\n"
    
    md_path = os.path.join(results_dir, "evaluation_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    
    logger.info(f"📝 Markdown results saved to: {md_path}")


def _print_summary(metrics: dict, results: list):
    """Print a formatted summary to the console."""
    
    print("\n" + "=" * 70)
    print("🏁 DEMO COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"  📊 Routing Accuracy:    {metrics['routing_accuracy']:.1f}%")
    print(f"  📊 Avg Relevance:       {metrics['avg_relevance_score']:.2f}")
    print(f"  📊 Grounding Rate:      {metrics['grounding_rate']:.1f}%")
    print(f"  ⏱️  Avg Time/Query:      {metrics['avg_processing_time']:.2f}s")
    print(f"  🔄 Avg Steps/Query:     {metrics['avg_steps_per_query']:.1f}")
    print(f"  ⏱️  Total Time:          {metrics['total_time_seconds']:.2f}s")
    print("=" * 70)
    
    print("\n📋 Per-Query Breakdown:")
    for r in results:
        status = "✅" if r.get("is_grounded") else ("❌ Error" if "error" in r else "⚠️")
        print(f"  {r['query_number']}. [{status}] {r.get('actual_type', 'error'):12s} | {r['query'][:60]}...")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_demo()
