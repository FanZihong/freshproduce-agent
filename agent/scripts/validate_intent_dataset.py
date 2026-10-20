"""校验预标注结果：schema、标签白名单、分布统计，并导出待人工校准清单。

用法（在 agent/ 目录下执行）：
    python scripts/validate_intent_dataset.py
    python scripts/validate_intent_dataset.py --min-confidence 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DATA_DIR  # noqa: E402
from agent.intent.taxonomy import load_taxonomy  # noqa: E402

RAW_DIR = DATA_DIR / "intent" / "raw"
REVIEW_DIR = DATA_DIR / "intent" / "review"
DEFAULT_INPUT = RAW_DIR / "prelabel_v1.jsonl"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "intent_prelabel_report.md"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_rows(rows: list[dict], valid_labels: set[str]) -> tuple[list[str], list[dict]]:
    """返回 (错误列表, 合法行列表)。"""
    problems: list[str] = []
    valid_rows: list[dict] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()

    for index, row in enumerate(rows, start=1):
        row_id = row.get("id", f"row-{index}")
        text = str(row.get("text") or "").strip()
        labels = row.get("labels") or []

        if not text:
            problems.append(f"{row_id}: 缺少 text")
            continue
        if row_id in seen_ids:
            problems.append(f"{row_id}: id 重复")
            continue
        seen_ids.add(row_id)
        if text in seen_text:
            problems.append(f"{row_id}: text 与前面样本重复")
            continue
        seen_text.add(text)

        if not isinstance(labels, list) or not labels:
            problems.append(f"{row_id}: labels 为空")
            continue
        unknown = [label for label in labels if label not in valid_labels]
        if unknown:
            problems.append(f"{row_id}: 出现体系外的标签 {unknown}")
            continue
        if len(labels) > 2:
            problems.append(f"{row_id}: 标签数超过 2 个 {labels}")
            continue
        confidence = row.get("confidence")
        if confidence is not None and not 0.0 <= float(confidence) <= 1.0:
            problems.append(f"{row_id}: confidence 越界 {confidence}")
            continue
        valid_rows.append(row)
    return problems, valid_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="校验意图预标注结果")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--review-out", type=Path, default=REVIEW_DIR / "low_confidence_v1.jsonl")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="低于该值进入人工校准清单")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if not rows:
        print(f"没有找到预标注结果：{args.input}（先跑 scripts/label_intent.py）")
        return

    taxonomy = load_taxonomy()
    valid_labels = set(taxonomy.keys)
    problems, valid_rows = validate_rows(rows, valid_labels)

    label_counts = Counter(label for row in valid_rows for label in row["labels"])
    primary_counts = Counter(row["labels"][0] for row in valid_rows)
    multi = [row for row in valid_rows if len(row["labels"]) > 1]
    confidences = [float(row.get("confidence") or 0.0) for row in valid_rows]
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    low_confidence = [row for row in valid_rows if float(row.get("confidence") or 0.0) < args.min_confidence]
    domain_out = [row for row in valid_rows if row["labels"] == ["chitchat"]]

    covered = set(label_counts)
    missing = sorted(valid_labels - covered)

    print(f"预标注文件：{args.input}")
    print(f"总条数：{len(rows)}；校验通过 {len(valid_rows)} 条，问题 {len(problems)} 条")
    print(f"平均置信度：{average_confidence:.3f}；低于 {args.min_confidence} 的 {len(low_confidence)} 条")
    print(f"多标签样本：{len(multi)} 条（{len(multi) / len(valid_rows):.1%}）")
    print(f"域外/闲聊样本：{len(domain_out)} 条（{len(domain_out) / len(valid_rows):.1%}）")
    print("\n标签分布（按首要标签）：")
    for spec in taxonomy.labels:
        count = primary_counts.get(spec.key, 0)
        print(f"  {spec.key:<18}{spec.name:<10}{count}")
    if missing:
        print(f"\n[warn] 以下类别没有样本：{missing}")
    if problems:
        print(f"\n前 10 条问题：")
        for problem in problems[:10]:
            print(f"  - {problem}")

    args.review_out.parent.mkdir(parents=True, exist_ok=True)
    args.review_out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in low_confidence) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# 意图预标注报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 输入文件：{args.input.name}",
        f"- 总条数：{len(rows)}（校验通过 {len(valid_rows)}，问题 {len(problems)}）",
        f"- 平均置信度：{average_confidence:.3f}",
        f"- 多标签比例：{len(multi) / len(valid_rows):.1%}；域外比例：{len(domain_out) / len(valid_rows):.1%}",
        "",
        "| 标签 | 名称 | 首要标签数 | 全部出现次数 |",
        "| --- | --- | --- | --- |",
    ]
    for spec in taxonomy.labels:
        lines.append(
            f"| {spec.key} | {spec.name} | {primary_counts.get(spec.key, 0)} | "
            f"{label_counts.get(spec.key, 0)} |"
        )
    if missing:
        lines += ["", f"> 缺少样本的类别：{missing}"]
    if problems:
        lines += ["", "## 校验问题（前 20 条）", ""]
        lines += [f"- {problem}" for problem in problems[:20]]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入：{args.report}")
    print(f"待人工校准清单：{args.review_out}（{len(low_confidence)} 条）")


if __name__ == "__main__":
    main()
