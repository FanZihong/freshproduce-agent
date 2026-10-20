"""RRF 融合测试：公式、互补性收益、可复现、权重。"""

from __future__ import annotations

import pytest

from agent.rag.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion, rrf_score
from agent.rag.hybrid_search import SearchHit


def _hit(chunk_id: str, rank: int, retriever: str = "bm25", score: float = 1.0) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        doc_id="doc",
        doc_title="文档",
        section="小节",
        text=f"文本-{chunk_id}",
        source="示例来源",
        retriever=retriever,
        rank=rank,
        score=score,
    )


def test_rrf_score_formula():
    assert rrf_score(1, k=DEFAULT_RRF_K) == pytest.approx(1 / 61)
    assert rrf_score(3, k=10) == pytest.approx(1 / 13)


def test_fusion_sums_contributions_from_both_paths():
    bm25 = [_hit("a", 1, "bm25"), _hit("b", 2, "bm25")]
    vector = [_hit("a", 1, "vector", score=0.9)]
    fused = reciprocal_rank_fusion([("bm25", bm25), ("vector", vector)], k=60)

    assert fused[0].chunk_id == "a"
    assert fused[0].fused_score == pytest.approx(2 / 61, rel=1e-6)
    assert fused[0].ranks == {"bm25": 1, "vector": 1}
    assert fused[0].scores["vector"] == pytest.approx(0.9)


def test_consistency_beats_single_path_rank_one():
    """两路都排第 2 的候选，分数应高于只被一路排第 1 的候选。"""
    bm25 = [_hit("x", 1, "bm25"), _hit("y", 2, "bm25")]
    vector = [_hit("y", 2, "vector")]
    fused = reciprocal_rank_fusion([("bm25", bm25), ("vector", vector)], k=60)
    assert fused[0].chunk_id == "y"
    assert fused[0].fused_score == pytest.approx(2 / 62, rel=1e-6)


def test_fusion_is_deterministic():
    bm25 = [_hit("b", 1, "bm25"), _hit("a", 2, "bm25")]
    vector = [_hit("a", 1, "vector"), _hit("b", 2, "vector")]
    first = [hit.chunk_id for hit in reciprocal_rank_fusion([("bm25", bm25), ("vector", vector)])]
    second = [hit.chunk_id for hit in reciprocal_rank_fusion([("bm25", bm25), ("vector", vector)])]
    assert first == second


def test_weights_shift_priority():
    bm25 = [_hit("b", 1, "bm25")]
    vector = [_hit("v", 1, "vector")]
    fused = reciprocal_rank_fusion([("bm25", bm25), ("vector", vector)], weights={"bm25": 3.0})
    assert fused[0].chunk_id == "b"
