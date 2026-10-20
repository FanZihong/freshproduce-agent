"""评测指标：检索（hit@k / recall@k / precision@k / MRR）、
分类（macro-F1 / 逐类 PRF / 混淆矩阵）、多标签（完全一致率）、
性能与成本（P95 时延 / 千次调用成本）。

分层约定：
- 检索指标与多标签指标是纯函数，无外部依赖；
- 分类指标优先走 scikit-learn（F1 等口径与业界一致，不自己造），
  装了 sklearn 用 sklearn，没装才退回等价的纯 Python 实现；
- 时延与成本指标只做聚合，不做任何假设，价格由调用方传入。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)

# scikit-learn 是可选依赖：装了就用它的口径（推荐），没装则走等价兜底实现。
try:  # pragma: no cover - 取决于环境
    from sklearn.metrics import (  # type: ignore
        confusion_matrix as _sk_confusion_matrix,
    )
    from sklearn.metrics import f1_score as _sk_f1_score
    from sklearn.metrics import (
        precision_recall_fscore_support as _sk_prf_support,
    )

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于环境
    SKLEARN_AVAILABLE = False


def _top_k(retrieved: Sequence[str], k: int) -> list[str]:
    return list(retrieved[:k])


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """top-k 里是否至少命中一条相关 chunk（1.0 / 0.0）。"""
    relevant_set = set(relevant)
    return 1.0 if any(chunk_id in relevant_set for chunk_id in _top_k(retrieved, k)) else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """top-k 覆盖了多少比例的相关 chunk。"""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    found = len(relevant_set & set(_top_k(retrieved, k)))
    return found / len(relevant_set)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """top-k 里有多少比例是真相关（k 大于返回条数时按实际条数算）。"""
    top = _top_k(retrieved, k)
    if not top:
        return 0.0
    relevant_set = set(relevant)
    return len([chunk_id for chunk_id in top if chunk_id in relevant_set]) / len(top)


def first_relevant_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> int | None:
    """第一条相关结果的排名（从 1 开始），没有命中返回 None。"""
    relevant_set = set(relevant)
    for index, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_set:
            return index
    return None


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """MRR 的单条贡献：1/排名，没有命中记 0。"""
    rank = first_relevant_rank(retrieved, relevant)
    return 1.0 / rank if rank else 0.0


def evaluate(rows: Sequence[dict], ks: Sequence[int] = DEFAULT_KS) -> dict:
    """把逐条结果聚合成指标。

    rows 每项形如 {"retrieved": [...], "relevant": [...]}。
    """
    if not rows:
        return {}

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"hit@{k}"] = round(
            sum(hit_at_k(row["retrieved"], row["relevant"], k) for row in rows) / len(rows), 4
        )
        metrics[f"recall@{k}"] = round(
            sum(recall_at_k(row["retrieved"], row["relevant"], k) for row in rows) / len(rows), 4
        )
        metrics[f"precision@{k}"] = round(
            sum(precision_at_k(row["retrieved"], row["relevant"], k) for row in rows) / len(rows), 4
        )
    metrics["mrr"] = round(
        sum(reciprocal_rank(row["retrieved"], row["relevant"]) for row in rows) / len(rows), 4
    )
    metrics["queries"] = len(rows)
    return metrics


def format_metrics(metrics: dict, keys: Sequence[str] | None = None) -> str:
    """把指标字典格式化成一行，便于打印与写报告。"""
    ordered = keys or [key for key in metrics if key != "queries"]
    return "  ".join(f"{key}={metrics.get(key, 0):.4f}" for key in ordered)


def label_agreement(left: Sequence[str], right: Sequence[str]) -> float:
    """两次标注的完全一致率。"""
    if len(left) != len(right):
        raise ValueError("两份标注的样本数不一致")
    if not left:
        return 0.0
    return sum(1 for a, b in zip(left, right) if a == b) / len(left)


def cohens_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    """Cohen's kappa：衡量标注一致性（扣除随机一致的部分）。"""
    if len(left) != len(right):
        raise ValueError("两份标注的样本数不一致")
    total = len(left)
    if total == 0:
        return 0.0

    observed = sum(1 for a, b in zip(left, right) if a == b) / total
    labels = set(left) | set(right)
    expected = sum(
        (left.count(label) / total) * (right.count(label) / total) for label in labels
    )
    if expected >= 1.0:
        return 1.0
    return round((observed - expected) / (1 - expected), 4)


def per_class_prf(
    gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None
) -> dict[str, dict]:
    """逐类的 precision / recall / f1 / support。

    labels 显式传入时要按传入顺序逐类统计（**模型完全没预测过的类别也会出现，
    f1 = 0**），这对"某类被整体漏掉"的场景比默认并集更诚实。
    """
    label_list = list(labels) if labels else sorted(set(gold) | set(pred))
    if not label_list:
        return {}

    if SKLEARN_AVAILABLE:  # pragma: no cover - 取决于环境
        precision, recall, f1, support = _sk_prf_support(
            gold, pred, labels=label_list, zero_division=0
        )
        return {
            label: {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
            for index, label in enumerate(label_list)
        }

    report: dict[str, dict] = {}
    for label in label_list:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for g in gold if g == label),
        }
    return report


def macro_f1(
    gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None
) -> float:
    """逐类 F1 的宏平均（每类等权，长尾类不会被高频类稀释）。

    默认只统计真值里出现过的类别；传入 labels 时按传入集合统计，
    **没预测过或没出现过的类别也计入分母（f1 = 0）**——评测时建议传
    taxonomy 的 10 个类别，这样"某一类被整体漏掉"会直接反映到分数上。
    """
    label_list = list(labels) if labels else sorted(set(gold))
    if not label_list:
        return 0.0
    if SKLEARN_AVAILABLE:  # pragma: no cover - 取决于环境
        score = _sk_f1_score(gold, pred, labels=label_list, average="macro", zero_division=0)
        return round(float(score), 4)
    scores: list[float] = []
    for label in label_list:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return round(sum(scores) / len(scores), 4)


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    """整体准确率（只看主标签）。"""
    if not gold:
        return 0.0
    return round(sum(1 for g, p in zip(gold, pred) if g == p) / len(gold), 4)


def confusion_matrix(
    gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None
) -> dict[str, dict[str, int]]:
    """混淆矩阵：matrix[gold][pred] = 条数。"""
    label_list = list(labels) if labels else sorted(set(gold) | set(pred))
    if SKLEARN_AVAILABLE and label_list:  # pragma: no cover - 取决于环境
        raw = _sk_confusion_matrix(gold, pred, labels=label_list)
        return {
            gold_label: {pred_label: int(raw[i][j]) for j, pred_label in enumerate(label_list)}
            for i, gold_label in enumerate(label_list)
        }

    matrix = {label: {other: 0 for other in label_list} for label in label_list}
    for g, p in zip(gold, pred):
        if g in matrix and p in matrix[g]:
            matrix[g][p] += 1
    return matrix


def format_prf_table(report: dict[str, dict], order: Sequence[str] | None = None) -> list[str]:
    """把逐类 PRF 渲染成 markdown 表格行。"""
    keys = list(order) if order else list(report)
    lines = ["| 类别 | precision | recall | f1 | support |", "| --- | --- | --- | --- | --- |"]
    for label in keys:
        item = report.get(label)
        if not item:
            continue
        lines.append(
            f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {item['support']} |"
        )
    return lines


def format_confusion_matrix(
    matrix: dict[str, dict[str, int]], labels: Sequence[str] | None = None
) -> list[str]:
    """把混淆矩阵渲染成 markdown 表格行（行=真值，列=预测）。"""
    keys = list(labels) if labels else list(matrix)
    if not keys:
        return []
    header = "| 真值 \\ 预测 | " + " | ".join(keys) + " |"
    divider = "| --- | " + " | ".join("---" for _ in keys) + " |"
    lines = [header, divider]
    for gold_label in keys:
        row = matrix.get(gold_label, {})
        lines.append(
            f"| {gold_label} | " + " | ".join(str(row.get(key, 0)) for key in keys) + " |"
        )
    return lines


def exact_match_ratio(gold_sets: Sequence[Iterable[str]], pred_sets: Sequence[Iterable[str]]) -> float:
    """多标签完全一致率：预测的标签集合与真值集合完全相同才算对。"""
    if len(gold_sets) != len(pred_sets):
        raise ValueError("真值与预测的样本数不一致")
    if not gold_sets:
        return 0.0
    hits = sum(1 for g, p in zip(gold_sets, pred_sets) if set(g) == set(p))
    return round(hits / len(gold_sets), 4)


def multilabel_f1(
    gold_sets: Sequence[Iterable[str]],
    pred_sets: Sequence[Iterable[str]],
    labels: Sequence[str],
) -> dict[str, float]:
    """多标签 F1：把每个标签二值化后按 macro / micro 汇总。

    macro 把每个标签等权（小类别重要），micro 把每条样本×标签等权（大类别主导）。
    """
    if len(gold_sets) != len(pred_sets):
        raise ValueError("真值与预测的样本数不一致")
    label_list = list(labels)
    if not gold_sets or not label_list:
        return {"macro": 0.0, "micro": 0.0}

    gold_matrix = [[1 if label in set(row) else 0 for label in label_list] for row in gold_sets]
    pred_matrix = [[1 if label in set(row) else 0 for label in label_list] for row in pred_sets]

    if SKLEARN_AVAILABLE:  # pragma: no cover - 取决于环境
        return {
            "macro": round(
                float(_sk_f1_score(gold_matrix, pred_matrix, average="macro", zero_division=0)), 4
            ),
            "micro": round(
                float(_sk_f1_score(gold_matrix, pred_matrix, average="micro", zero_division=0)), 4
            ),
        }

    def _binary_scores(index: int) -> tuple[int, int, int]:
        tp = fp = fn = 0
        for gold_row, pred_row in zip(gold_matrix, pred_matrix):
            if gold_row[index] and pred_row[index]:
                tp += 1
            elif pred_row[index]:
                fp += 1
            elif gold_row[index]:
                fn += 1
        return tp, fp, fn

    macro_scores: list[float] = []
    total_tp = total_fp = total_fn = 0
    for index in range(len(label_list)):
        tp, fp, fn = _binary_scores(index)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        denominator = 2 * tp + fp + fn
        macro_scores.append(2 * tp / denominator if denominator else 0.0)
    micro_denominator = 2 * total_tp + total_fp + total_fn
    return {
        "macro": round(sum(macro_scores) / len(macro_scores), 4),
        "micro": round(2 * total_tp / micro_denominator, 4) if micro_denominator else 0.0,
    }


def p95_latency(latencies_ms: Sequence[float]) -> float:
    """P95 时延（毫秒）：95% 的请求都在这个值以内。

    用最近秩（nearest-rank）口径，不做插值：排序后取第 ceil(0.95 * n) 个值。
    这样 20 条样本里恰好是"第 19 条"，比插值更好解释。
    """
    values = sorted(float(value) for value in latencies_ms)
    if not values:
        return 0.0
    rank = max(1, math.ceil(0.95 * len(values)))
    return round(values[rank - 1], 1)


def latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    """时延分布：均值 / P50 / P90 / P95 / 最大值 / 样本数。"""
    values = sorted(float(value) for value in latencies_ms)
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}

    def _rank(percentile: float) -> float:
        rank = max(1, math.ceil(percentile * len(values)))
        return round(values[rank - 1], 1)

    return {
        "count": len(values),
        "mean_ms": round(sum(values) / len(values), 1),
        "p50_ms": _rank(0.50),
        "p90_ms": _rank(0.90),
        "p95_ms": _rank(0.95),
        "max_ms": round(values[-1], 1),
    }


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    price_in_per_m: float,
    price_out_per_m: float,
) -> float:
    """单次调用成本：输入/输出 token 分别按"元每百万 token"计价。"""
    return round(
        (prompt_tokens * price_in_per_m + completion_tokens * price_out_per_m) / 1_000_000, 8
    )


def cost_per_1k(costs: Sequence[float]) -> float:
    """千次调用成本：按每次平均成本 × 1000 估算。"""
    if not costs:
        return 0.0
    return round(sum(costs) / len(costs) * 1000, 4)


def token_stats(prompt_tokens: Sequence[int], completion_tokens: Sequence[int]) -> dict[str, float]:
    """token 用量统计（平均值与总量），用于核对成本口径。"""
    if not prompt_tokens and not completion_tokens:
        return {"calls": 0, "avg_prompt_tokens": 0.0, "avg_completion_tokens": 0.0, "total_tokens": 0}
    calls = len(prompt_tokens) or len(completion_tokens)
    total = sum(prompt_tokens) + sum(completion_tokens)
    return {
        "calls": calls,
        "avg_prompt_tokens": round(sum(prompt_tokens) / calls, 1) if calls else 0.0,
        "avg_completion_tokens": round(sum(completion_tokens) / calls, 1) if calls else 0.0,
        "total_tokens": total,
    }
