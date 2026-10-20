"""生成人工校准清单：低置信度全量 + 高置信度抽样。

用法（在 agent/ 目录下执行）：
    python scripts/prepare_calibration.py
    python scripts/prepare_calibration.py --high-rate 0.2 --low-threshold 0.6
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DATA_DIR  # noqa: E402

RAW_DIR = DATA_DIR / "intent" / "raw"
REVIEW_DIR = DATA_DIR / "intent" / "review"
DEFAULT_INPUT = RAW_DIR / "prelabel_v1.jsonl"
DEFAULT_OUT = REVIEW_DIR / "calibration_v1.jsonl"
DEFAULT_CSV = REVIEW_DIR / "calibration_v1.csv"

CSV_FIELDS = (
    "id",
    "text",
    "last_question",
    "product",
    "labels",
    "confidence",
    "reason",
    "reviewed_labels",
    "note",
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成人工校准清单")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--low-threshold", type=float, default=0.6, help="低于该置信度全部复核")
    parser.add_argument("--high-rate", type=float, default=0.2, help="高置信度抽样比例")
    parser.add_argument(
        "--max-items",
        type=int,
        default=400,
        help="校准清单上限（取置信度最低的 N 条；0 表示不限）。建议第一批 300–400 条",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [row for row in load_jsonl(args.input) if row.get("labels")]
    if not rows:
        print(f"没有可用的预标注结果：{args.input}")
        return

    low = [row for row in rows if float(row.get("confidence") or 0) < args.low_threshold]
    high = [row for row in rows if float(row.get("confidence") or 0) >= args.low_threshold]
    random.seed(args.seed)
    sampled_high = random.sample(high, max(1, int(len(high) * args.high_rate))) if high else []

    ordered = sorted(low + sampled_high, key=lambda item: float(item.get("confidence") or 0))
    if args.max_items and len(ordered) > args.max_items:
        ordered = ordered[: args.max_items]

    review_rows = []
    for row in ordered:
        review_rows.append(
            {
                "id": row["id"],
                "text": row["text"],
                "context": row.get("context"),
                "labels": row["labels"],
                "confidence": float(row.get("confidence") or 0),
                "reason": row.get("reason", ""),
                "source": row.get("source", ""),
                "reviewed_labels": [],
                "note": "",
            }
        )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in review_rows) + "\n",
        encoding="utf-8",
    )

    # 同时给一份 CSV，方便用 Excel 编辑（utf-8-sig 保证 Excel 不乱码）
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in review_rows:
            context = row.get("context") or {}
            writer.writerow(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "last_question": context.get("last_question", ""),
                    "product": context.get("product", ""),
                    "labels": "|".join(row["labels"]),
                    "confidence": f"{row['confidence']:.2f}",
                    "reason": row.get("reason", ""),
                    "reviewed_labels": "",
                    "note": "",
                }
            )

    print(f"预标注总数：{len(rows)}")
    print(f"低置信度（<{args.low_threshold}）：{len(low)} 条，全部进入校准清单")
    print(f"高置信度抽样（{args.high_rate:.0%}）：{len(sampled_high)} 条（共 {len(high)} 条）")
    print(
        f"候补合计 {len(low) + len(sampled_high)} 条，"
        f"按上限截取后输出 {len(review_rows)} 条"
        f"（约需 {len(review_rows) / 60:.1f} 小时，按每分钟 1 条估）"
    )
    print("建议：先做第一批 300–400 条算出采纳率，再决定要不要扩大校准范围。")
    print(f"\nJSONL：{args.out_jsonl}")
    print(f"CSV（推荐用 Excel 打开）：{args.out_csv}")
    print('\n填写说明：在 reviewed_labels 列填正确标签（多个用 | 分隔，如 "bad_fruit_claim|urge_logistics"）；')
    print("标签与原来一致就照抄原值，不一致就改成正确的；不确定的写进 note 列。")


if __name__ == "__main__":
    main()
