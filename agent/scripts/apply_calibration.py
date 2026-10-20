"""把人工校准结果合并回全量预标注，并统计「预标注采纳率」。

用法（在 agent/ 目录下执行）：
    python scripts/apply_calibration.py
    python scripts/apply_calibration.py --review data/intent/review/calibration_v1.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DATA_DIR  # noqa: E402
from agent.intent.taxonomy import load_taxonomy  # noqa: E402

RAW_DIR = DATA_DIR / "intent" / "raw"
REVIEW_DIR = DATA_DIR / "intent" / "review"
DEFAULT_PRELABEL = RAW_DIR / "prelabel_v1.jsonl"
DEFAULT_REVIEW_JSONL = REVIEW_DIR / "calibration_v1.jsonl"
DEFAULT_REVIEW_CSV = REVIEW_DIR / "calibration_v1.csv"
DEFAULT_OUTPUT = RAW_DIR / "calibrated_v1.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_review(path: Path) -> list[dict]:
    """支持两种校准文件：JSONL（原格式）与 CSV（Excel 填写后直接使用）。"""
    if not path.exists():
        return []
    if path.suffix.lower() == ".csv":
        rows: list[dict] = []
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                labels = [
                    part.strip()
                    for part in re.split(r"[|，,]", raw.get("reviewed_labels") or "")
                    if part.strip()
                ]
                rows.append(
                    {
                        "id": raw.get("id", ""),
                        "reviewed_labels": labels,
                        "note": raw.get("note", ""),
                    }
                )
        return rows
    return load_jsonl(path)


def resolve_review_path(explicit: Path | None) -> Path | None:
    """校准文件优先用 CSV（Excel 里填的那份），其次才是 JSONL。"""
    if explicit is not None:
        return explicit if explicit.exists() else None
    if DEFAULT_REVIEW_CSV.exists():
        return DEFAULT_REVIEW_CSV
    if DEFAULT_REVIEW_JSONL.exists():
        return DEFAULT_REVIEW_JSONL
    return None


def merge_calibration(prelabel_rows: list[dict], review_rows: list[dict]) -> tuple[list[dict], dict]:
    """把校准结果合并回全量数据，返回 (合并结果, 统计)。"""
    valid_labels = set(load_taxonomy().keys)
    reviewed = {row["id"]: row for row in review_rows if row.get("reviewed_labels")}
    corrected = 0
    confirmed = 0
    invalid_labels: Counter = Counter()
    changed_by_label: Counter = Counter()
    merged: list[dict] = []

    for row in prelabel_rows:
        review = reviewed.get(row["id"])
        if review is None:
            merged.append({**row, "review_status": "not_reviewed"})
            continue

        original = list(row.get("labels") or [])
        raw_labels = [str(label).strip() for label in review["reviewed_labels"]]
        final_labels = [label for label in raw_labels if label in valid_labels]
        for label in raw_labels:
            if label not in valid_labels:
                invalid_labels[label] += 1
        if not final_labels:
            # 校准标签全都不合法：视为未校准，保留预标注结果
            merged.append({**row, "review_status": "invalid_review"})
            continue
        if set(final_labels) == set(original):
            confirmed += 1
            status = "confirmed"
        else:
            corrected += 1
            status = "corrected"
            changed_by_label[original[0] if original else "unknown"] += 1

        merged.append(
            {
                **row,
                "labels": final_labels,
                "labeler": "human-calibrated",
                "review_status": status,
                "prelabel_labels": original,
                "note": review.get("note", ""),
            }
        )

    reviewed_count = len(reviewed)
    stats = {
        "total": len(prelabel_rows),
        "reviewed": reviewed_count,
        "confirmed": confirmed,
        "corrected": corrected,
        "adoption_rate": round(confirmed / reviewed_count, 4) if reviewed_count else 0.0,
        "corrected_by_label": dict(changed_by_label.most_common()),
        "pending_review": len([row for row in review_rows if not row.get("reviewed_labels")]),
        "invalid_labels": dict(invalid_labels.most_common()),
    }
    return merged, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="合并人工校准结果")
    parser.add_argument("--prelabel", type=Path, default=DEFAULT_PRELABEL)
    parser.add_argument(
        "--review",
        type=Path,
        default=None,
        help="校准文件；不指定时优先使用 calibration_v1.csv",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    prelabel_rows = load_jsonl(args.prelabel)
    if not prelabel_rows:
        print(f"没有预标注结果：{args.prelabel}")
        return
    review_path = resolve_review_path(args.review)
    if review_path is None:
        print("没有找到校准清单（calibration_v1.csv / calibration_v1.jsonl）")
        print("先跑 scripts/prepare_calibration.py 生成清单。")
        return
    review_rows = load_review(review_path)
    filled = sum(1 for row in review_rows if row.get("reviewed_labels"))
    print(f"校准文件：{review_path.name}（已填写 {filled}/{len(review_rows)} 条）")
    if filled == 0:
        other = DEFAULT_REVIEW_JSONL if review_path.suffix.lower() == ".csv" else DEFAULT_REVIEW_CSV
        if other.exists() and any(row.get("reviewed_labels") for row in load_review(other)):
            print(f"[提示] 另一份文件里有已填写的校准结果，试试 --review {other}")
        else:
            print("[提示] 这份文件里 reviewed_labels 还是空的，填完再跑。")
        return

    merged, stats = merge_calibration(prelabel_rows, review_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in merged) + "\n",
        encoding="utf-8",
    )

    print(f"全量数据：{stats['total']} 条")
    print(f"已校准：{stats['reviewed']} 条（未填写 {stats['pending_review']} 条）")
    print(f"  - 与预标注一致：{stats['confirmed']} 条")
    print(f"  - 人工修正：{stats['corrected']} 条")
    print(f"预标注采纳率：{stats['adoption_rate']:.1%}")
    if stats["corrected_by_label"]:
        summary = "，".join(
            f"{key} {value}" for key, value in list(stats["corrected_by_label"].items())[:5]
        )
        print(f"修正最多的原类别：{summary}")
    if stats["invalid_labels"]:
        print(
            "[warn] 以下标签不在 taxonomy 里，已忽略（请检查拼写，用英文 key 而不是中文名）："
            + "，".join(f"{key} {value}" for key, value in list(stats["invalid_labels"].items())[:8])
        )
    print(f"\n校准后数据：{args.output}")


if __name__ == "__main__":
    main()
