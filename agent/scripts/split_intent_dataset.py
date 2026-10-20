"""把校准后的标注数据划分成 train / dev / test。

两条纪律：
1. 按类别分层，保证三个集合的类别分布一致；
2. 同一个 session 的样本不跨集合（防止"那它呢"这类上下文泄漏）。

用法（在 agent/ 目录下执行）：
    python scripts/split_intent_dataset.py
    python scripts/split_intent_dataset.py --input data/intent/raw/calibrated_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DATA_DIR  # noqa: E402
from agent.intent.taxonomy import load_taxonomy  # noqa: E402

RAW_DIR = DATA_DIR / "intent" / "raw"
DATASET_DIR = DATA_DIR / "intent" / "dataset_v1"
DEFAULT_INPUT = RAW_DIR / "calibrated_v1.jsonl"
DEFAULT_FALLBACK = RAW_DIR / "prelabel_v1.jsonl"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "intent_dataset_v1.md"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_key(row: dict) -> str:
    """同一个会话的样本归为一组，保证不跨集合。"""
    return str(row.get("session_id") or f"solo::{row['id']}")


def primary_label(row: dict) -> str:
    labels = row.get("labels") or []
    return labels[0] if labels else "unknown"


def split_dataset(
    rows: list[dict],
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[dict]]:
    """按主标签分层、按会话分组划分。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    by_label: dict[str, list[list[dict]]] = defaultdict(list)
    for members in groups.values():
        by_label[primary_label(members[0])].append(members)

    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}

    for _label, label_groups in sorted(by_label.items()):
        rng.shuffle(label_groups)
        total = sum(len(group) for group in label_groups)
        train_target = total * ratios[0]
        dev_target = total * (ratios[0] + ratios[1])
        counts = {"train": 0, "dev": 0, "test": 0}
        for group in label_groups:
            if counts["train"] < train_target:
                split = "train"
            elif counts["train"] + counts["dev"] < dev_target:
                split = "dev"
            else:
                split = "test"
            splits[split].extend(group)
            counts[split] += len(group)
    return splits


def build_stats(splits: dict[str, list[dict]], min_per_class: int) -> dict:
    taxonomy = load_taxonomy()
    stats: dict = {"total": sum(len(rows) for rows in splits.values()), "splits": {}}
    train_counts = Counter(primary_label(row) for row in splits["train"])

    for name, rows in splits.items():
        per_label = Counter(primary_label(row) for row in rows)
        stats["splits"][name] = {
            "total": len(rows),
            "perLabel": {spec.key: per_label.get(spec.key, 0) for spec in taxonomy.labels},
            "multiLabel": sum(1 for row in rows if len(row.get("labels") or []) > 1),
            "chitchat": per_label.get("chitchat", 0),
        }

    stats["warnings"] = [
        f"{spec.key} 训练样本仅 {train_counts.get(spec.key, 0)} 条（< {min_per_class}），需要定向补样"
        for spec in taxonomy.labels
        if train_counts.get(spec.key, 0) < min_per_class
    ]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="划分意图数据集")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-class", type=int, default=50)
    args = parser.parse_args()

    source = args.input if args.input.exists() else DEFAULT_FALLBACK
    rows = [row for row in load_jsonl(source) if row.get("labels")]
    if not rows:
        print(f"没有可用标注数据（已尝试 {args.input} 与 {DEFAULT_FALLBACK}）")
        return
    if source == DEFAULT_FALLBACK:
        print("[warn] 未找到校准后的数据，先用预标注结果划分（完成人工校准后请重跑）")

    splits = split_dataset(rows, seed=args.seed)
    stats = build_stats(splits, args.min_per_class)

    seen: dict[str, str] = {}
    conflicts = 0
    for name, split_rows in splits.items():
        for row in split_rows:
            key = group_key(row)
            if key in seen and seen[key] != name:
                conflicts += 1
            seen[key] = name
    if conflicts:
        print(f"[warn] 检测到 {conflicts} 条会话跨集合，请检查划分逻辑")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        path = args.out_dir / f"{name}.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in split_rows) + "\n",
            encoding="utf-8",
        )
    (args.out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"数据来源：{source.name}，共 {stats['total']} 条")
    print(
        f"划分：train {stats['splits']['train']['total']} / "
        f"dev {stats['splits']['dev']['total']} / test {stats['splits']['test']['total']}"
    )
    print(
        f"多标签（train）：{stats['splits']['train']['multiLabel']} 条；"
        f"域外 chitchat（train）：{stats['splits']['train']['chitchat']} 条"
    )

    taxonomy = load_taxonomy()
    print(f"\n{'类别':<18}{'train':<8}{'dev':<8}{'test':<8}")
    for spec in taxonomy.labels:
        print(
            f"{spec.key:<18}{stats['splits']['train']['perLabel'][spec.key]:<8}"
            f"{stats['splits']['dev']['perLabel'][spec.key]:<8}"
            f"{stats['splits']['test']['perLabel'][spec.key]:<8}"
        )

    if stats["warnings"]:
        print("\n长尾告警：")
        for warning in stats["warnings"]:
            print(f"  - {warning}")

    lines = [
        "# 意图数据集 v1 划分报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据来源：{source.name}（{stats['total']} 条）",
        "- 划分比例：7 : 1.5 : 1.5（按主标签分层、按会话分组）",
        "",
        "| 类别 | train | dev | test |",
        "| --- | --- | --- | --- |",
    ]
    for spec in taxonomy.labels:
        lines.append(
            f"| {spec.key} | {stats['splits']['train']['perLabel'][spec.key]} | "
            f"{stats['splits']['dev']['perLabel'][spec.key]} | "
            f"{stats['splits']['test']['perLabel'][spec.key]} |"
        )
    lines += [
        "",
        f"- 多标签样本（train）：{stats['splits']['train']['multiLabel']} 条",
        f"- 域外样本（train）：{stats['splits']['train']['chitchat']} 条",
    ]
    if stats["warnings"]:
        lines += ["", "## 长尾告警", ""] + [f"- {warning}" for warning in stats["warnings"]]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n数据集目录：{args.out_dir}")
    print(f"报告：{args.report}")


if __name__ == "__main__":
    main()
