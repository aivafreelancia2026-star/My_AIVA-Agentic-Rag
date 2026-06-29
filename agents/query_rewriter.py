"""
agents/query_rewriter.py

Query Rewriter / Sub-query Generator for Agentic RAG.

Takes an original query + a context evaluation result and produces:
  - A rewritten (improved) query
  - Optional sub-queries targeting specific missing topics
  - A search strategy hint

Rule-based by default (no LLM needed). If an LLM is provided it can
optionally be used to produce richer rewrites.
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ── Token helpers ──────────────────────────────────────────────────────────────

_STOPWORDS = {
    'the', 'and', 'for', 'are', 'was', 'were', 'but', 'not', 'have', 'has',
    'with', 'this', 'that', 'they', 'from', 'will', 'what', 'which', 'who',
    'whom', 'how', 'when', 'where', 'why', 'can', 'could', 'would', 'should',
    'about', 'its', 'into', 'than', 'then', 'there', 'their', 'some', 'any',
    'all', 'been', 'being', 'did', 'does', 'doing', 'had', 'him', 'his', 'her',
    'she', 'our', 'out', 'own', 'same', 'too', 'very', 'just', 'also',
    'may', 'each', 'most', 'more', 'such', 'even', 'here', 'only', 'both'
}

# Semantic synonym expansions for common query terms
_SYNONYMS: Dict[str, List[str]] = {
    "explain":      ["describe", "overview", "what is", "define"],
    "difference":   ["compare", "vs", "versus", "contrast"],
    "list":         ["show all", "enumerate", "what are the"],
    "how":          ["steps to", "process for", "procedure"],
    "who":          ["person responsible", "team", "department"],
    "when":         ["date", "timeline", "schedule"],
    "where":        ["location", "section", "part"],
    "summary":      ["overview", "key points", "main ideas"],
    "table":        ["data", "rows", "columns", "records"],
    "chart":        ["graph", "visualization", "trend", "data"],
    "image":        ["screenshot", "figure", "diagram", "picture"],
    "popup":        ["dialog", "modal", "window", "overlay"],
    "screen":       ["page", "view", "interface"],
    "button":       ["click", "action", "control"],
    "tab":          ["navigation", "section", "panel"],
}

# Search strategy detection patterns
_STRATEGY_PATTERNS = {
    "semantic":     [r'\b(what|explain|describe|how|why|overview|tell me)\b'],
    "keyword":      [r'\b(find|search|locate|where is|show me)\b'],
    "page-specific":[r'\bpage\s+\d+\b', r'\bon page\b', r'\bin page\b'],
    "summary":      [r'\b(summarize|summary|overview|main points|key points)\b'],
    "table":        [r'\b(table|list|who|staff|employees|names|directory)\b'],
    "ui":           [r'\b(popup|dialog|screen|button|tab|modal|window|click|interface)\b'],
    "chart":        [r'\b(chart|graph|trend|data|visualization|statistics)\b'],
}


def _content_words(text: str) -> List[str]:
    """Return meaningful words from text."""
    words = re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split()
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


def _detect_strategy(query: str) -> str:
    """Pick the best search strategy label for this query."""
    q = query.lower()
    for strategy, patterns in _STRATEGY_PATTERNS.items():
        if any(re.search(p, q) for p in patterns):
            return strategy
    return "semantic"


def _expand_with_synonyms(query: str) -> str:
    """Add synonyms for key terms to broaden the search."""
    words = query.lower().split()
    expansions = []
    for word in words:
        for base, synonyms in _SYNONYMS.items():
            if word == base or word in synonyms:
                expansions.extend(synonyms[:2])
    if expansions:
        unique_exp = list(dict.fromkeys(e for e in expansions if e not in query.lower()))[:3]
        return f"{query} {' '.join(unique_exp)}"
    return query


# ── Main rewriter ──────────────────────────────────────────────────────────────

def rewrite_query(
    original_query: str,
    context_manager=None,
    evaluation: Optional[Dict[str, Any]] = None,
    iteration: int = 0,
    llm=None,
) -> Dict[str, Any]:
    """
    Generate a better query when context evaluation says retrieval was weak.

    Args:
        original_query:  The user's original question.
        context_manager: Conversation context (used for entity memory).
        evaluation:      Result from context_evaluator.evaluate_context_quality().
        iteration:       Which retry pass this is (0-based within rewrite calls).
        llm:             Optional LLM for richer rewrites (not used by default).

    Returns:
        {
            "rewritten_query": str,
            "sub_queries": list[str],
            "search_strategy": str,
            "reasoning": str,
        }
    """
    missing_info: List[str] = []
    recommended: List[str] = []
    coverage_score: float = 1.0

    if evaluation:
        missing_info    = evaluation.get("missing_information", [])
        recommended     = evaluation.get("recommended_next_queries", [])
        coverage_score  = evaluation.get("coverage_score", 1.0)

    strategy = _detect_strategy(original_query)

    # ── Strategy A: use evaluator's recommended next queries ─────────────────
    if recommended and iteration == 0:
        primary = recommended[0]
        sub_queries = recommended[1:3]
        reasoning = "Using evaluator's recommended next query."

    # ── Strategy B: focus on missing terms ───────────────────────────────────
    elif missing_info and iteration <= 1:
        key_missing = missing_info[:3]
        content = _content_words(original_query)
        anchor = content[:2] if content else []
        primary = " ".join(anchor + key_missing)
        sub_queries = [
            " ".join(key_missing),                          # pure missing terms
            _expand_with_synonyms(original_query),          # synonym-expanded original
        ]
        reasoning = f"Focusing on missing terms: {', '.join(key_missing)}."

    # ── Strategy C: semantic expansion ───────────────────────────────────────
    elif coverage_score < 0.3:
        primary = _expand_with_synonyms(original_query)
        content = _content_words(original_query)
        sub_queries = [
            " ".join(content[:4]) if len(content) >= 4 else original_query,
            f"overview of {' '.join(content[:3])}" if content else original_query,
        ]
        reasoning = "Low coverage — expanding query with synonyms."

    # ── Strategy D: generic fallback — shorten to core nouns ─────────────────
    else:
        content = _content_words(original_query)
        primary = " ".join(content[:5]) if content else original_query
        sub_queries = [
            f"information about {primary}",
            _expand_with_synonyms(primary),
        ]
        reasoning = "Falling back to core content words."

    # Pull entity context from conversation manager if available
    entity_hints: List[str] = []
    if context_manager and hasattr(context_manager, 'entity_memory'):
        for _, edata in list(context_manager.entity_memory.items())[:3]:
            val = edata.get('value', '')
            if val and val.lower() not in primary.lower():
                entity_hints.append(val)
    if entity_hints:
        primary = f"{primary} {' '.join(entity_hints[:2])}"
        reasoning += f" Added entity context: {', '.join(entity_hints[:2])}."

    # Sanitise
    primary = primary.strip()
    if not primary:
        primary = original_query

    sub_queries = [s.strip() for s in sub_queries if s.strip() and s.strip().lower() != primary.lower()]

    result = {
        "rewritten_query":  primary,
        "sub_queries":      sub_queries[:3],
        "search_strategy":  strategy,
        "reasoning":        reasoning,
    }

    logger.debug(
        "[QueryRewriter] iter=%d  strategy=%s  rewritten=%r  sub=%s",
        iteration, strategy, primary, sub_queries
    )
    return result


def generate_sub_queries(
    original_query: str,
    evaluation: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Convenience wrapper: return just the sub-query list from rewrite_query().
    Useful when calling code only needs the alternative queries.
    """
    result = rewrite_query(original_query, evaluation=evaluation)
    sub = result.get("sub_queries", [])
    if result.get("rewritten_query", "").lower() != original_query.lower():
        sub = [result["rewritten_query"]] + sub
    return sub[:4]
