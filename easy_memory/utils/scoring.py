from __future__ import annotations
import math
from typing import Any, Dict, List, Optional


def get_bm25_params(query: str, *, lemmatized: Optional[str] = None) -> tuple:
    if lemmatized is None:
        from easy_memory.utils.lemmatization import lemmatize_for_bm25
        lemmatized = lemmatize_for_bm25(query)
    num_terms = len(lemmatized.split()) if lemmatized else 1
    if num_terms <= 3:
        return 5.0, 0.7
    elif num_terms <= 6:
        return 7.0, 0.6
    elif num_terms <= 9:
        return 9.0, 0.5
    elif num_terms <= 15:
        return 10.0, 0.5
    else:
        return 12.0, 0.5


def normalize_bm25(raw_score: float, midpoint: float, steepness: float) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))


ENTITY_BOOST_WEIGHT = 0.5


def score_and_rank(
        semantic_results: List[Dict[str, Any]],
        bm25_scores: Dict[str, float],
        entity_boosts: Dict[str, float],
        threshold: float,
        top_k: int,
) -> List[Dict[str, Any]]:
    has_bm25 = bool(bm25_scores)
    has_entity = bool(entity_boosts)
    max_possible = 1.0
    if has_bm25:
        max_possible += 1.0
    if has_entity:
        max_possible += ENTITY_BOOST_WEIGHT

    scored = []
    for result in semantic_results:
        mem_id = result.get("id")
        if mem_id is None:
            continue
        semantic_score = result.get("score", 0.0)
        if semantic_score < threshold:
            continue
        mem_id_str = str(mem_id)
        bm25_score = bm25_scores.get(mem_id_str, 0.0)
        entity_boost = entity_boosts.get(mem_id_str, 0.0)
        raw_combined = semantic_score + bm25_score + entity_boost
        combined = min(raw_combined / max_possible, 1.0)
        scored.append({"id": mem_id_str, "score": combined, "payload": result.get("payload")})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
