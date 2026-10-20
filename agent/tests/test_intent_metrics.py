"""意图评测指标测试：macro-F1 / 逐类 PRF / 混淆矩阵 / 多标签 / P95 / 成本。

这些断言在"装了 sklearn"和"没装、走兜底实现"两种环境下都成立，
所以它们同时在守护两条实现路径的口径一致性。
"""

from __future__ import annotations

from eval.metrics import (
    accuracy,
    confusion_matrix,
    cost_per_1k,
    estimate_cost,
    exact_match_ratio,
    latency_stats,
    macro_f1,
    multilabel_f1,
    p95_latency,
    per_class_prf,
    token_stats,
)


def test_macro_f1_default_only_counts_labels_in_gold():
    gold = ["a", "a", "b"]
    pred = ["a", "b", "b"]
    # a 与 b 各 0.6667，平均 0.6667
    assert macro_f1(gold, pred) == 0.6667


def test_macro_f1_with_explicit_labels_penalizes_missing_class():
    """传入完整类别表时，一次都没预测到的类按 0 分计入，避免"漏一类"被忽略。"""
    assert macro_f1(["a", "a", "b"], ["a", "b", "b"], ["a", "b", "c"]) == 0.4444


def test_macro_f1_zero_when_no_gold_labels():
    assert macro_f1([], []) == 0.0


def test_per_class_prf_reports_zero_support_label():
    report = per_class_prf(["a"], ["a"], ["a", "b"])
    assert report["a"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}
    assert report["b"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}


def test_accuracy_counts_only_primary_label():
    assert accuracy(["a", "a", "b"], ["a", "b", "b"]) == 0.6667
    assert accuracy([], []) == 0.0


def test_confusion_matrix_rows_are_gold_columns_are_pred():
    matrix = confusion_matrix(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
    assert matrix["a"]["a"] == 1
    assert matrix["a"]["b"] == 1
    assert matrix["b"]["b"] == 1
    assert matrix["b"]["a"] == 0


def test_exact_match_ratio_requires_identical_label_set():
    assert exact_match_ratio([["a"], ["a", "b"]], [["a"], ["b"]]) == 0.5
    assert exact_match_ratio([["a", "b"]], [["b", "a"]]) == 1.0


def test_multilabel_f1_macro_and_micro():
    gold = [["a"], ["a", "b"]]
    pred = [["a"], ["b"]]
    scores = multilabel_f1(gold, pred, ["a", "b"])
    # a: tp=1 fn=1 -> 0.6667；b: tp=1 -> 1.0；micro: tp=2 fp=0 fn=1 -> 0.8
    assert scores["macro"] == 0.8333
    assert scores["micro"] == 0.8


def test_p95_latency_uses_nearest_rank():
    # 20 条里只有 1 条慢请求：P95 应该是 10ms，而不是被极值拉高
    assert p95_latency([10.0] * 19 + [100.0]) == 10.0
    assert p95_latency(list(range(1, 101))) == 95.0
    assert p95_latency([]) == 0.0


def test_latency_stats_reports_distribution():
    stats = latency_stats([10.0] * 19 + [100.0])
    assert stats["count"] == 20
    assert stats["p50_ms"] == 10.0
    assert stats["p95_ms"] == 10.0
    assert stats["max_ms"] == 100.0
    assert latency_stats([])["count"] == 0


def test_estimate_cost_and_cost_per_1k():
    # 300 输入 token + 100 输出 token，按 ¥2 / ¥8 每百万 token 计价
    single = estimate_cost(300, 100, 2.0, 8.0)
    assert single == 0.0014
    assert cost_per_1k([single] * 10) == 1.4
    assert cost_per_1k([]) == 0.0


def test_token_stats_averages():
    stats = token_stats([1000, 1200], [20, 40])
    assert stats["calls"] == 2
    assert stats["avg_prompt_tokens"] == 1100.0
    assert stats["avg_completion_tokens"] == 30.0
    assert stats["total_tokens"] == 2260
