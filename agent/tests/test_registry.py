"""版本清单测试：从真实产物扫描出可对比的版本（含校准与域外指标挂载）。"""

from __future__ import annotations

import json
from pathlib import Path

from agent.intent.registry import scan_versions, summary_payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def make_workspace(tmp_path: Path) -> tuple[Path, Path]:
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    v1 = models / "v1"
    v1.mkdir(parents=True)
    write_json(
        v1 / "train_config.json",
        {
            "created_at": "2026-10-13 12:07:00",
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "prompt_detail": "minimal",
            "max_len": 384,
            "train_samples": 2614,
        },
    )
    write_json(v1 / "train_summary.json", {"recovered": True, "max_len": 384, "train_samples": 2614})
    (v1 / "adapter_config.json").write_text("{}", encoding="utf-8")

    # 第二个版本：只有 checkpoint，没有 adapter -> 训练中
    (models / "v2" / "checkpoints" / "checkpoint-100").mkdir(parents=True)

    write_json(
        reports / "intent_baseline.json",
        {
            "created_at": "2026-10-13 15:04:00",
            "dataset": "dataset_v1_balanced",
            "split_sizes": {"test": 701},
            "backends": [
                {"backend": "rule", "split": "test", "strategy": "rule-based-v1", "macro_f1": 0.2013,
                 "accuracy": 0.3923, "latency": {"p95_ms": 0.0}, "cost_per_1k": 0.0, "queries": 701},
                {"backend": "fewshot", "split": "test", "strategy": "fewshot-llm", "model": "deepseek-chat",
                 "macro_f1": 0.849, "accuracy": 0.8459, "latency": {"p95_ms": 0.0},
                 "cost_per_1k": 4.2121, "queries": 701},
                {"backend": "local", "split": "test", "strategy": "local-lora", "local_dir": str(v1),
                 "macro_f1": 0.7366, "accuracy": 0.7375, "latency": {"p95_ms": 1017.0},
                 "cost_per_1k": 0.0, "queries": 701, "errors": 2},
            ],
        },
    )
    write_json(
        reports / "intent_calibration.json",
        {
            "adapter_dir": str(v1),
            "temperature": 1.82,
            "ece_before": 0.21,
            "ece_after": 0.07,
            "fit_split": "dev",
            "eval_split": "test",
            "samples": 701,
            "config": {"accept_threshold": 0.62, "handoff_threshold": 0.37},
            "thresholds": {"refusal_rate": 0.083, "refused_error_rate": 0.71,
                           "accepted_error_rate": 0.12, "ratio": 5.9},
        },
    )
    write_json(
        reports / "intent_ood.json",
        {
            "backend": "local-lora",
            "with_embedder": True,
            "model_only": {"ood_recall": 0.62, "false_kill_rate": 0.01},
            "with_signals": {"ood_recall": 0.94, "false_kill_rate": 0.03},
        },
    )
    return models, reports


def test_scan_versions_reads_real_artifacts(tmp_path):
    models, reports = make_workspace(tmp_path)
    versions = scan_versions(models, reports, active_dir=str(models / "v1"))
    by_version = {item["version"]: item for item in versions}

    assert set(by_version) == {"v1", "v2", "rule-baseline", "fewshot-baseline"}
    v1 = by_version["v1"]
    assert v1["status"] == "ready"
    assert v1["active"] is True
    assert v1["f1"] == 0.7366
    assert v1["latencyMs"] == 1017
    assert v1["promptDetail"] == "minimal"
    assert v1["maxLen"] == 384
    assert "max_len 384" in v1["note"]
    assert "checkpoint 还原" in v1["note"]


def test_training_version_has_no_metrics(tmp_path):
    models, reports = make_workspace(tmp_path)
    versions = {item["version"]: item for item in scan_versions(models, reports)}
    v2 = versions["v2"]
    assert v2["status"] == "training"
    assert v2["f1"] == 0.0
    assert v2["latencyMs"] == 0


def test_baselines_are_included(tmp_path):
    models, reports = make_workspace(tmp_path)
    versions = {item["version"]: item for item in scan_versions(models, reports)}
    assert versions["fewshot-baseline"]["f1"] == 0.849
    assert versions["fewshot-baseline"]["costPerK"] == 4.2121
    assert versions["rule-baseline"]["f1"] == 0.2013


def test_calibration_attaches_only_to_matching_adapter(tmp_path):
    models, reports = make_workspace(tmp_path)
    versions = {item["version"]: item for item in scan_versions(models, reports)}
    assert versions["v1"]["calibration"]["temperature"] == 1.82
    assert versions["v1"]["calibration"]["refusalRate"] == 0.083
    assert versions["v2"].get("calibration") is None


def test_ood_metrics_attach_to_local_versions(tmp_path):
    models, reports = make_workspace(tmp_path)
    versions = {item["version"]: item for item in scan_versions(models, reports)}
    assert versions["v1"]["ood"]["oodRecall"] == 0.94
    assert versions["v1"]["ood"]["falseKillRate"] == 0.03


def test_empty_workspace_returns_empty_list(tmp_path):
    versions = scan_versions(tmp_path / "none", tmp_path / "none-reports")
    assert versions == []


def test_summary_payload_includes_dataset(tmp_path):
    models, reports = make_workspace(tmp_path)
    payload = summary_payload(models, reports)
    assert payload["dataset"]["name"] == "dataset_v1_balanced"
    assert payload["dataset"]["splits"] == {"test": 701}
    assert len(payload["versions"]) == 4
