"""
Agentic Adaptive RAG - Prompt Templates

All LLM prompts used by the agent for routing, generation, grading, and transformation.
"""

# ─────────────────────────────────────────────
# Query Router Prompt
# ─────────────────────────────────────────────
QUERY_ROUTER_PROMPT = """You are an expert query classifier for a Retrieval-Augmented Generation system 
specialized in particle physics and machine learning research.

Analyze the user's query and classify it into ONE of these categories:

1. **simple** — A straightforward factual question that can be answered with a single retrieval pass.
   Examples: "What is the Higgs boson?", "Explain gradient descent"

2. **complex** — A multi-faceted question requiring decomposition into sub-questions and multi-hop retrieval.
   Examples: "Compare GNNs vs CNNs for jet classification and discuss computational trade-offs",
   "How has federated learning been applied to privacy-preserving particle physics research?"

3. **web_search** — A question about very recent events, real-time data, or topics unlikely to be in the 
   knowledge base of arXiv papers on particle physics and ML.
   Examples: "What are the latest LHC results from 2025?", "Current job openings at CERN"

Respond with ONLY valid JSON in this exact format:
{{"query_type": "<simple|complex|web_search>", "reasoning": "<brief explanation>"}}

User Query: {query}"""


# ─────────────────────────────────────────────
# Query Decomposition Prompt (for complex queries)
# ─────────────────────────────────────────────
QUERY_DECOMPOSITION_PROMPT = """You are an expert at breaking down complex research questions into 
simpler sub-questions that can be individually answered.

Given the following complex query about particle physics and machine learning, decompose it into 
2-4 focused sub-questions that together would fully answer the original query.

Respond with ONLY valid JSON in this exact format:
{{"sub_queries": ["<sub_question_1>", "<sub_question_2>", "<sub_question_3>"]}}

Complex Query: {query}"""


# ─────────────────────────────────────────────
# Answer Generation Prompt
# ─────────────────────────────────────────────
GENERATION_PROMPT = """You are an expert research assistant specializing in particle physics and 
machine learning. Answer the user's question using ONLY the provided context.

Rules:
- Base your answer strictly on the provided context
- If the context doesn't contain enough information, say so clearly
- Cite sources by referencing paper titles or document sections
- Be concise but thorough
- Use technical language appropriate for a research audience

Context:
{context}

User Question: {query}

Answer:"""


# ─────────────────────────────────────────────
# Answer Grading Prompt (Hallucination Check)
# ─────────────────────────────────────────────
ANSWER_GRADER_PROMPT = """You are a rigorous fact-checker for a research Q&A system.

Evaluate whether the generated answer is properly grounded in the provided context.
Check for:
1. **Factual grounding** — Every claim in the answer should be supported by the context
2. **Relevance** — The answer actually addresses the user's question
3. **Completeness** — The answer doesn't miss critical information from the context

Respond with ONLY valid JSON in this exact format:
{{"is_grounded": <true|false>, "relevance_score": <0.0-1.0>, "reasoning": "<brief explanation>"}}

Context:
{context}

User Question: {query}

Generated Answer: {answer}"""


# ─────────────────────────────────────────────
# Query Transformation Prompt (for retries)
# ─────────────────────────────────────────────
QUERY_TRANSFORM_PROMPT = """You are an expert at reformulating search queries to improve retrieval results.

The previous query did not return satisfactory results. Reformulate it to be more specific, 
use alternative terminology, or approach the question from a different angle.

Original Query: {query}
Previous Answer Quality: The answer was not well-grounded in the retrieved documents.

Respond with ONLY valid JSON:
{{"transformed_query": "<improved_query>", "strategy": "<what you changed and why>"}}"""


# ─────────────────────────────────────────────
# Web Search Synthesis Prompt
# ─────────────────────────────────────────────
WEB_SEARCH_SYNTHESIS_PROMPT = """You are an expert research assistant. Synthesize the following 
web search results into a comprehensive answer to the user's question.

Rules:
- Combine information from multiple sources when relevant
- Clearly indicate what comes from web search results
- Be accurate and do not add information not present in the results

Web Search Results:
{web_results}

User Question: {query}

Answer:"""
