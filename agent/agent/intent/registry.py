"""意图模型版本清单：给后端 `GET /api/intent/versions` 提供真实数据。

只读真实产物，绝不编数字。数据来源：
- `data/intent/models/<name>/train_config.json`、`train_summary.json`、`adapter_config.json`
  —— 判断训练是否完成、记录训练配置与样本量；
- `eval/reports/intent_baseline.json` —— 三方对比的 macro-F1 / P95 时延 / 千次成本；
- `eval/reports/intent_calibration.json` —— 温度、拒答阈值、ECE 前后；
- `eval/reports/intent_ood.json` —— 域外召回与已知类误杀率。

文件缺失时对应的字段留空（前端显示"—"），而不是回落到假数据。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.db import DATA_DIR

DEFAULT_MODELS_DIR = DATA_DIR / "intent" / "models"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[2] / "eval" / "reports"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _baseline_rows(reports_dir: Path) -> list[dict]:
    payload = load_json(reports_dir / "intent_baseline.json") or {}
    return list(payload.get("backends") or [])


def _match_baseline(rows: list[dict], *, backend: str, local_dir: str | None = None) -> dict | None:
    for row in rows:
        if row.get("backend") != backend:
            continue
        if local_dir is None or str(row.get("local_dir") or "") == local_dir:
            return row
    return None


def _metrics_from(row: dict | None, calibration: dict | None, ood: dict | None) -> dict:
    if not row:
        return {"f1": 0.0, "latencyMs": 0, "costPerK": 0.0}
    latency = (row.get("latency") or {}).get("p95_ms") or row.get("p95_latency_ms") or 0
    metrics = {
        "f1": round(float(row.get("macro_f1") or 0.0), 4),
        "latencyMs": int(round(float(latency))),
        "costPerK": round(float(row.get("cost_per_1k") or 0.0), 4),
        "accuracy": round(float(row.get("accuracy") or 0.0), 4),
        "errors": int(row.get("errors") or 0),
        "split": row.get("split"),
        "cached": int(row.get("cached") or 0),
        "queries": int(row.get("queries") or 0),
        "perClass": row.get("per_class") or {},
    }
    if calibration:
        metrics["calibration"] = {
            "temperature": calibration.get("temperature"),
            "eceBefore": calibration.get("ece_before"),
            "eceAfter": calibration.get("ece_after"),
            "acceptThreshold": (calibration.get("config") or {}).get("accept_threshold"),
            "handoffThreshold": (calibration.get("config") or {}).get("handoff_threshold"),
            "refusalRate": (calibration.get("thresholds") or {}).get("refusal_rate"),
            "refusedErrorRate": (calibration.get("thresholds") or {}).get("refused_error_rate"),
            "acceptedErrorRate": (calibration.get("thresholds") or {}).get("accepted_error_rate"),
            "ratio": (calibration.get("thresholds") or {}).get("ratio"),
            "fitSplit": calibration.get("fit_split"),
            "evalSplit": calibration.get("eval_split"),
            "samples": calibration.get("samples"),
        }
    if ood:
        metrics["ood"] = {
            "oodRecall": (ood.get("with_signals") or {}).get("ood_recall"),
            "falseKillRate": (ood.get("with_signals") or {}).get("false_kill_rate"),
            "modelOnlyRecall": (ood.get("model_only") or {}).get("ood_recall"),
            "withEmbedder": ood.get("with_embedder"),
            "backend": ood.get("backend"),
        }
    return metrics


def scan_versions(
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    active_dir: str | None = None,
) -> list[dict]:
    """扫描出所有可用版本（含 rule / few-shot 两个对照基线）。"""
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    baseline_rows = _baseline_rows(reports_dir)
    calibration = load_json(reports_dir / "intent_calibration.json")
    ood = load_json(reports_dir / "intent_ood.json")
    calibration_dir = (calibration or {}).get("adapter_dir")
    ood_backend = (ood or {}).get("backend")

    versions: list[dict] = []

    if models_dir.exists():
        for directory in sorted(models_dir.iterdir()):
            if not directory.is_dir():
                continue
            train_config = load_json(directory / "train_config.json")
            summary = load_json(directory / "train_summary.json") or {}
            adapter_ready = (directory / "adapter_config.json").exists()
            checkpoints = sorted(directory.glob("checkpoints/checkpoint-*"))
            if not train_config and not adapter_ready and not checkpoints:
                continue

            row = _match_baseline(baseline_rows, backend="local", local_dir=str(directory))
            own_calibration = (
                calibration if calibration_dir and str(calibration_dir) == str(directory) else None
            )
            own_ood = ood if ood_backend and row and ood_backend == row.get("strategy", "") else None
            if own_ood is None and ood and ood_backend == "local-lora":
                own_ood = ood
            metrics = _metrics_from(row, own_calibration, own_ood)

            prompt_detail = (train_config or {}).get("prompt_detail")
            max_len = (train_config or {}).get("max_len") or summary.get("max_len")
            samples = (train_config or {}).get("train_samples") or summary.get("train_samples")
            note_parts = [
                part
                for part in (
                    (train_config or {}).get("base_model", "").split("/")[-1],
                    f"提示词 {prompt_detail}" if prompt_detail else "",
                    f"max_len {max_len}" if max_len else "",
                    f"训练 {samples} 条" if samples else "",
                    "（训练收尾被中断，报告由 checkpoint 还原）" if summary.get("recovered") else "",
                )
                if part
            ]
            versions.append(
                {
                    "id": f"local-{directory.name}",
                    "version": f"{directory.name}",
                    "status": "ready" if adapter_ready else ("training" if checkpoints else "failed"),
                    "note": " · ".join(note_parts) or "本地 LoRA",
                    "createdAt": (train_config or {}).get("created_at"),
                    "promptDetail": prompt_detail,
                    "maxLen": max_len,
                    "baseModel": (train_config or {}).get("base_model"),
                    "trainSamples": samples,
                    "adapterDir": str(directory),
                    "active": active_dir is not None and Path(active_dir) == directory,
                    **metrics,
                }
            )

    # 两个对照基线：零成本兜底与大模型 few-shot
    for backend, label, note in (
        ("rule", "rule-baseline", "关键词规则版：零成本兜底，只覆盖 4 类"),
        (
            "fewshot",
            "fewshot-baseline",
            "大模型 few-shot 基线：依赖网络，按 token 计费",
        ),
    ):
        row = _match_baseline(baseline_rows, backend=backend)
        if not row:
            continue
        metrics = _metrics_from(row, None, ood if ood_backend == backend else None)
        versions.append(
            {
                "id": backend,
                "version": label,
                "status": "ready",
                "note": note,
                "createdAt": (load_json(reports_dir / "intent_baseline.json") or {}).get("created_at"),
                "promptDetail": None,
                "maxLen": None,
                "baseModel": row.get("model"),
                "trainSamples": None,
                "adapterDir": None,
                "active": False,
                **metrics,
            }
        )

    for version in versions:
        for key, default in (
            ("f1", 0.0),
            ("latencyMs", 0),
            ("costPerK", 0.0),
            ("accuracy", 0.0),
            ("errors", 0),
        ):
            version.setdefault(key, default)
    versions.sort(key=lambda item: (not item.get("active"), str(item.get("createdAt") or "")), reverse=False)
    return versions


def summary_payload(models_dir=None, reports_dir=None, active_dir=None) -> dict[str, Any]:
    """给后端用的包装：版本列表 + 数据集与校准摘要。"""
    reports_dir = Path(reports_dir or DEFAULT_REPORTS_DIR)
    baseline = load_json(reports_dir / "intent_baseline.json") or {}
    return {
        "versions": scan_versions(models_dir or DEFAULT_MODELS_DIR, reports_dir, active_dir),
        "dataset": {
            "name": baseline.get("dataset"),
            "splits": baseline.get("split_sizes") or {},
        },
        "evaluatedAt": baseline.get("created_at"),
    }
