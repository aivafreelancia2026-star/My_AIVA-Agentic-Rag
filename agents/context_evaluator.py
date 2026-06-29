"""
agents/context_evaluator.py

Context Quality Evaluator for Agentic RAG.

Analyses retrieved chunks and decides whether they are sufficient
to generate a good answer, or whether another retrieval pass is needed.

Rule-based by default (zero extra dependencies). If an LLM is supplied
it can optionally produce a richer evaluation.
"""

import re
import hashlib
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Stopwords ──────────────────────────────────────────────────────────────────
_STOPWORDS = {
    'the', 'and', 'for', 'are', 'was', 'were', 'but', 'not', 'have', 'has',
    'with', 'this', 'that', 'they', 'from', 'will', 'what', 'which', 'who',
    'whom', 'how', 'when', 'where', 'why', 'can', 'could', 'would', 'should',
    'about', 'its', 'into', 'than', 'then', 'there', 'their', 'some', 'any',
    'all', 'been', 'being', 'did', 'does', 'doing', 'had', 'him', 'his', 'her',
    'she', 'him', 'our', 'out', 'own', 'same', 'too', 'very', 'just', 'also',
    'may', 'each', 'most', 'more', 'such', 'even', 'here', 'only', 'both'
}


def _tokenize(text: str) -> set:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return set(t for t in text.split() if len(t) > 2)


def _meaningful_tokens(text: str) -> set:
    return _tokenize(text) - _STOPWORDS


def _chunk_text(chunk) -> str:
    """Extract plain text from a LangChain Document or a dict chunk."""
    if isinstance(chunk, dict):
        return chunk.get('text', '') or chunk.get('page_content', '')
    return getattr(chunk, 'page_content', '') or ''


def _chunk_relevance(chunk) -> float:
    """Extract stored relevance score (0–1) from metadata if available."""
    if isinstance(chunk, dict):
        return float(chunk.get('relevance_score', 0.0) or chunk.get('score', 0.0))
    meta = getattr(chunk, 'metadata', {}) or {}
    return float(meta.get('relevance_score', 0.0))


# ── Main evaluator ─────────────────────────────────────────────────────────────

def evaluate_context_quality(
    query: str,
    chunks: List[Any],
    iteration: int = 0,
    min_confidence: float = 0.65,
    llm=None,
) -> Dict[str, Any]:
    """
    Evaluate whether `chunks` contain enough information to answer `query`.

    Args:
        query:          The original user question.
        chunks:         LangChain Documents (or dicts with 'text' key).
        iteration:      Current retrieval iteration (0-based).
        min_confidence: Threshold to declare context "sufficient".
        llm:            Optional LLM for richer evaluation (not used by default).

    Returns:
        {
            "sufficient": bool,
            "confidence": float,
            "reason": str,
            "missing_information": list[str],
            "recommended_next_queries": list[str],
            "should_retry": bool,
            "chunk_count": int,
            "coverage_score": float,
            "duplicate_ratio": float,
        }
    """
    # ── Empty guard ────────────────────────────────────────────────────────────
    if not chunks:
        return {
            "sufficient": False,
            "confidence": 0.0,
            "reason": "No documents retrieved — knowledge base may not contain relevant content.",
            "missing_information": [query],
            "recommended_next_queries": [_broaden_query(query)],
            "should_retry": iteration < 2,
            "chunk_count": 0,
            "coverage_score": 0.0,
            "duplicate_ratio": 0.0,
        }

    # ── 1. Deduplicate ──────────────────────────────────────────────────────────
    seen: set = set()
    unique: List[Any] = []
    for c in chunks:
        text = _chunk_text(c)
        fp = hashlib.md5(text[:200].encode('utf-8', errors='ignore')).hexdigest()
        if fp not in seen:
            seen.add(fp)
            unique.append(c)

    duplicate_ratio = 1.0 - (len(unique) / max(len(chunks), 1))

    # ── 2. Keyword / token coverage ─────────────────────────────────────────────
    query_tokens = _meaningful_tokens(query)
    combined_text = " ".join(_chunk_text(c) for c in unique[:12])
    context_tokens = _meaningful_tokens(combined_text)

    if query_tokens:
        covered = query_tokens & context_tokens
        missing_tokens = query_tokens - context_tokens
        coverage_score = len(covered) / len(query_tokens)
    else:
        covered = set()
        missing_tokens = set()
        coverage_score = 0.5

    # ── 3. Content depth (average chunk length) ──────────────────────────────────
    avg_len = sum(len(_chunk_text(c)) for c in unique) / max(len(unique), 1)
    depth_score = min(1.0, avg_len / 400.0)   # 400 chars ≈ "decent depth"

    # ── 4. Quantity score ───────────────────────────────────────────────────────
    quantity_score = min(1.0, len(unique) / 5.0)   # ≥5 unique chunks = full score

    # ── 5. Stored relevance from metadata ───────────────────────────────────────
    relevance_scores = [_chunk_relevance(c) for c in unique]
    avg_relevance = sum(relevance_scores) / max(len(relevance_scores), 1)
    # Treat 0.0 (unset) as neutral 0.5 to avoid unfair penalty
    adjusted_relevance = avg_relevance if avg_relevance > 0.05 else 0.5

    # ── 6. Generic / too-broad detection ────────────────────────────────────────
    specificity_penalty = 0.0
    if coverage_score < 0.25 and len(unique) >= 3:
        # We got results but they don't even cover a quarter of the query tokens
        specificity_penalty = 0.15

    # ── 7. Composite confidence ──────────────────────────────────────────────────
    confidence = (
        coverage_score      * 0.40 +
        depth_score         * 0.20 +
        quantity_score      * 0.20 +
        adjusted_relevance  * 0.20
    ) - specificity_penalty

    # Penalise heavy duplication
    if duplicate_ratio > 0.6:
        confidence *= (1.0 - duplicate_ratio * 0.5)

    confidence = max(0.0, min(1.0, confidence))

    # ── 8. Determine issues & sufficiency ───────────────────────────────────────
    issues = []
    if coverage_score < 0.35:
        issues.append(f"low keyword coverage ({coverage_score:.0%})")
    if len(unique) < 2:
        issues.append("very few unique chunks retrieved")
    if avg_len < 80:
        issues.append("chunks are very short — low information density")
    if duplicate_ratio > 0.55:
        issues.append(f"high duplication ({duplicate_ratio:.0%})")
    if specificity_penalty > 0:
        issues.append("retrieved context appears too generic for this query")

    sufficient = confidence >= min_confidence

    reason_parts = [f"Confidence: {confidence:.2f}."]
    if sufficient:
        reason_parts.append("Context appears sufficient to answer the query.")
    else:
        reason_parts.append("Context is insufficient.")
        if issues:
            reason_parts.append("Issues: " + "; ".join(issues) + ".")

    # ── 9. Recommendations ───────────────────────────────────────────────────────
    missing_info = sorted(missing_tokens)[:6]
    next_queries = _suggest_queries(query, missing_tokens) if not sufficient else []

    should_retry = not sufficient and iteration < 2

    result = {
        "sufficient": sufficient,
        "confidence": round(confidence, 3),
        "reason": " ".join(reason_parts),
        "missing_information": missing_info,
        "recommended_next_queries": next_queries,
        "should_retry": should_retry,
        "chunk_count": len(unique),
        "coverage_score": round(coverage_score, 3),
        "duplicate_ratio": round(duplicate_ratio, 3),
    }

    logger.debug(
        "[ContextEval] iter=%d  sufficient=%s  confidence=%.2f  coverage=%.2f  chunks=%d",
        iteration, result["sufficient"], result["confidence"],
        result["coverage_score"], result["chunk_count"]
    )
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _broaden_query(query: str) -> str:
    """Strip question words to get a broader search form."""
    q = re.sub(
        r'^(what is|who is|how does|when did|where is|why does|which|tell me about|explain|describe)\s+',
        '', query.lower()
    ).strip()
    q = re.sub(r'[?!.]$', '', q).strip()
    return q if q else query


def _suggest_queries(query: str, missing_tokens: set) -> List[str]:
    """Produce up to 3 alternative queries targeting missing information."""
    suggestions = []

    # 1. Focus on missing terms
    if missing_tokens:
        key_missing = sorted(missing_tokens)[:3]
        suggestions.append(" ".join(key_missing))

    # 2. Shorten query to its first 4 content words
    words = [w for w in query.split() if w.lower() not in _STOPWORDS]
    if len(words) > 3:
        suggestions.append(" ".join(words[:4]))

    # 3. Add "overview" framing
    if not any(w in query.lower() for w in ['overview', 'summary', 'explain', 'describe']):
        short = " ".join(words[:3]) if words else query
        suggestions.append(f"overview of {short}")

    # Deduplicate while preserving order
    seen_s: set = set()
    out = []
    for s in suggestions:
        s = s.strip()
        if s and s.lower() != query.lower() and s not in seen_s:
            seen_s.add(s)
            out.append(s)

    return out[:3]
