"""本地评测的行级缓存测试：中断后能接着算，不重复推理。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.eval_intent import (
    flush_cache,
    partition_rows_by_cache,
    row_cache_key,
    stamp_cache_meta,
    write_json_summary,
)


def make_rows() -> list[dict]:
    return [
        {"id": "1", "text": "怎么开发票", "context": None, "labels": ["invoice"]},
        {"id": "2", "text": "那它呢", "context": {"last_question": "草莓怎么保存"}, "labels": ["quality_qa"]},
        {"id": "3", "text": "草莓坏了", "context": None, "labels": ["bad_fruit_claim"]},
    ]


def test_row_cache_key_depends_on_text_and_context():
    assert row_cache_key("那它呢", None) == row_cache_key("那它呢", None)
    assert row_cache_key("那它呢", None) != row_cache_key("那它呢", {"last_question": "草莓怎么保存"})
    assert row_cache_key("那它呢", {"a": 1, "b": 2}) == row_cache_key("那它呢", {"b": 2, "a": 1})


def test_row_cache_key_is_scoped_by_adapter():
    """回归测试：换模型重跑时不能复用上一个模型的预测。

    之前缓存键只有文本+上下文，v2 的结果会被 v1 的评测静默当成自己的结果。
    """
    v1 = row_cache_key("怎么开发票", None, namespace="data/intent/models/v1")
    v2 = row_cache_key("怎么开发票", None, namespace="data/intent/models/v2")
    assert v1 != v2
    assert row_cache_key("怎么开发票", None, "data/intent/models/v1") == v1


def test_partition_respects_namespace():
    rows = make_rows()
    key_v2 = row_cache_key("怎么开发票", None, "data/intent/models/v2")
    cache = {key_v2: {"labels": ["invoice"], "confidence": 0.9, "latency_ms": 5.0}}

    # 用 v1 的 namespace 去查：不该命中 v2 的缓存
    pending_v1, hits_v1 = partition_rows_by_cache(rows, cache, "data/intent/models/v1")
    assert hits_v1 == {}
    assert len(pending_v1) == 3

    # 用 v2 的 namespace 去查：命中
    pending_v2, hits_v2 = partition_rows_by_cache(rows, cache, "data/intent/models/v2")
    assert list(hits_v2) == [0]
    assert len(pending_v2) == 2


def test_partition_splits_cached_and_pending():
    rows = make_rows()
    cached_key = row_cache_key("怎么开发票", None)
    cache = {cached_key: {"labels": ["invoice"], "confidence": 0.9, "latency_ms": 12.0}}

    pending, hits = partition_rows_by_cache(rows, cache)

    assert [index for index, _ in pending] == [1, 2]
    assert list(hits) == [0]
    assert hits[0]["labels"] == ["invoice"]


def test_partition_with_empty_cache_keeps_everything_pending():
    pending, hits = partition_rows_by_cache(make_rows(), {})
    assert len(pending) == 3
    assert hits == {}


def test_flush_cache_writes_and_can_be_reloaded(tmp_path):
    path = tmp_path / "local_eval_cache.json"
    cache = {"abc": {"labels": ["invoice"], "confidence": 0.5, "latency_ms": 3.0}}
    flush_cache(path, cache, enabled=True)
    assert json.loads(path.read_text(encoding="utf-8")) == cache

    # 禁用时不应写盘
    other = tmp_path / "disabled.json"
    flush_cache(other, cache, enabled=False)
    assert not other.exists()


def make_metrics(backend: str, split: str, local_dir: str | None = None) -> dict:
    return {
        "backend": backend,
        "split": split,
        "strategy": {"rule": "rule-based-v1", "fewshot": "fewshot-llm", "local": "local-lora"}[backend],
        "model": "deepseek-chat" if backend == "fewshot" else None,
        "local_dir": local_dir,
        "queries": 10,
        "macro_f1": 0.5,
        "accuracy": 0.5,
        "exact_match": 0.5,
        "multilabel": {"macro": 0.5, "micro": 0.5},
        "latency": {"p95_ms": 100.0},
        "cost_avg": 0.0,
        "cost_per_1k": 0.0,
        "errors": 0,
        "cached": 0,
        "per_class": {},
    }


def test_write_json_summary_merges_instead_of_overwriting(tmp_path):
    """分几次跑（test 三方 + dev 逐行）时，后一次不该冲掉前一次的 rule / few-shot 行。"""
    json_path = tmp_path / "intent_baseline.json"
    args = argparse.Namespace(json=json_path, shots=1)
    config = argparse.Namespace(price_in_per_m=2.0, price_out_per_m=8.0)
    dataset_dir = Path("dataset_v1_balanced")

    write_json_summary(
        args,
        dataset_dir,
        {"test": [{}] * 701},
        [
            ("rule（test）", make_metrics("rule", "test")),
            ("local（test）", make_metrics("local", "test", "/m/v1")),
        ],
        config,
        "deepseek-chat",
        None,
    )
    write_json_summary(
        args,
        dataset_dir,
        {"dev": [{}] * 704},
        [("local（dev）", make_metrics("local", "dev", "/m/v1"))],
        config,
        "deepseek-chat",
        None,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    signatures = {(row["backend"], row["split"]) for row in payload["backends"]}
    assert signatures == {("rule", "test"), ("local", "test"), ("local", "dev")}
    assert payload["split_sizes"] == {"test": 701, "dev": 704}


def test_write_json_summary_replaces_same_signature_run(tmp_path):
    """同一条（后端+集合+adapter）重跑时应当替换，而不是留两行。"""
    json_path = tmp_path / "intent_baseline.json"
    args = argparse.Namespace(json=json_path, shots=1)
    config = argparse.Namespace(price_in_per_m=2.0, price_out_per_m=8.0)

    for f1 in (0.5, 0.8):
        metrics = make_metrics("local", "test", "/m/v1")
        metrics["macro_f1"] = f1
        write_json_summary(
            args, Path("dataset"), {"test": [{}] * 5}, [("local", metrics)], config, "m", None
        )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["backends"]) == 1
    assert payload["backends"][0]["macro_f1"] == 0.8


def test_write_json_summary_writes_meta_next_to_json(tmp_path):
    """回归测试：sidecar 元信息必须跟着 --json 走。

    早先写死成 eval/reports/ 下的真实路径，于是单元测试传入临时 --json 时，
    把生产用的 local_eval_meta.json 覆盖成了测试里的占位值（/m/v1），
    导致之后的温度校准一条缓存都命中不了。
    """
    json_path = tmp_path / "out" / "intent_baseline.json"
    args = argparse.Namespace(json=json_path, shots=1)
    config = argparse.Namespace(price_in_per_m=2.0, price_out_per_m=8.0)

    write_json_summary(
        args,
        Path("dataset"),
        {"test": [{}] * 3},
        [("local", make_metrics("local", "test", "data/intent/models/v1"))],
        config,
        "m",
        {"a": 1, "b": 2, "_meta": {"adapter_dir": "x"}},
    )

    meta_path = json_path.with_name("local_eval_meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["adapter_dir"] == "data/intent/models/v1"
    assert meta["cache_entries"] == 2      # 不带下划线的才算条目


def test_stamp_cache_meta_records_namespace_and_entry_count():
    cache = {"hash-a": {}, "hash-b": {}}
    stamp_cache_meta(cache, "data/intent/models/v1")
    assert cache["_meta"]["adapter_dir"] == "data/intent/models/v1"
    assert cache["_meta"]["entries"] == 2
    # 再盖一次也不会把 _meta 自己算成条目
    stamp_cache_meta(cache, "data/intent/models/v2")
    assert cache["_meta"]["entries"] == 2
    assert cache["_meta"]["adapter_dir"] == "data/intent/models/v2"
