"""评测指标测试。"""

from __future__ import annotations

from eval.metrics import (
    evaluate,
    first_relevant_rank,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_at_k():
    retrieved = ["a", "b", "c"]
    assert hit_at_k(retrieved, ["c"], 3) == 1.0
    assert hit_at_k(retrieved, ["d"], 3) == 0.0
    assert hit_at_k(retrieved, ["c"], 2) == 0.0


def test_recall_at_k_with_multiple_relevant():
    retrieved = ["a", "b", "c"]
    assert recall_at_k(retrieved, ["a", "c"], 2) == 0.5
    assert recall_at_k(retrieved, ["a", "c"], 3) == 1.0
    assert recall_at_k(retrieved, [], 3) == 0.0


def test_precision_at_k():
    assert precision_at_k(["a", "x", "b"], ["a", "b"], 3) == 2 / 3
    assert precision_at_k([], ["a"], 3) == 0.0


def test_first_relevant_rank_and_mrr():
    assert first_relevant_rank(["x", "a", "b"], ["a"]) == 2
    assert first_relevant_rank(["x", "y"], ["a"]) is None
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_evaluate_aggregates_rows():
    rows = [
        {"retrieved": ["a", "b"], "relevant": ["a"]},
        {"retrieved": ["b", "c"], "relevant": ["c"]},
    ]
    metrics = evaluate(rows, ks=[1, 2])
    assert metrics["queries"] == 2
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@2"] == 1.0
    assert metrics["mrr"] == round((1.0 + 0.5) / 2, 4)
