"""标注一致性检查：计算完全一致率与 Cohen's kappa。

用法（在 agent/ 目录下执行）：
    # 模式一：同一份文件里比较 labels 与 reviewed_labels（自己隔天重标）
    python scripts/compute_kappa.py --file data/intent/review/calibration_v1.jsonl

    # 模式二：两份标注文件按 id 对齐（两个人各标一份）
    python scripts/compute_kappa.py --a a.jsonl --b b.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from eval.metrics import cohens_kappa, confusion_matrix, label_agreement, macro_f1  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def primary(labels) -> str:
    """多标签按排序拼接，顺序无关地比较一致性。"""
    return "+".join(sorted(str(label) for label in labels)) if labels else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="标注一致性检查")
    parser.add_argument("--file", type=Path, help="含 labels 与 reviewed_labels 的文件")
    parser.add_argument("--a", type=Path, help="标注结果 A")
    parser.add_argument("--b", type=Path, help="标注结果 B")
    parser.add_argument("--show-mismatch", type=int, default=10)
    args = parser.parse_args()

    pairs: list[tuple[str, str, str]] = []
    if args.file:
        for row in load_jsonl(args.file):
            if row.get("reviewed_labels"):
                pairs.append((row["id"], primary(row["labels"]), primary(row["reviewed_labels"])))
    elif args.a and args.b:
        left = {row["id"]: row for row in load_jsonl(args.a)}
        right = {row["id"]: row for row in load_jsonl(args.b)}
        for row_id in sorted(set(left) & set(right)):
            pairs.append((row_id, primary(left[row_id]["labels"]), primary(right[row_id]["labels"])))
    else:
        print("请指定 --file，或同时指定 --a 与 --b")
        return

    if not pairs:
        print("没有可比对的样本（reviewed_labels 为空，或两份文件 id 无交集）")
        return

    left_labels = [left for _, left, _ in pairs]
    right_labels = [right for _, _, right in pairs]
    kappa = cohens_kappa(left_labels, right_labels)
    agreement = label_agreement(left_labels, right_labels)
    f1 = macro_f1(left_labels, right_labels)

    print(f"比对样本：{len(pairs)} 条")
    print(f"完全一致率：{agreement:.1%}")
    print(f"Cohen's kappa：{kappa:.3f}")
    print(f"macro-F1（A 为真值、B 为预测）：{f1:.3f}")
    print("判定：kappa ≥ 0.75 表示体系边界清晰，可以继续；低于 0.75 建议先改 taxonomy。")

    mismatches = [(row_id, left, right) for row_id, left, right in pairs if left != right]
    if mismatches:
        print(f"\n不一致样例（共 {len(mismatches)} 条，展示前 {args.show_mismatch} 条）：")
        for row_id, left, right in mismatches[: args.show_mismatch]:
            print(f"  [{row_id}] A={left}  B={right}")

    matrix = confusion_matrix(left_labels, right_labels)
    labels = sorted(matrix)
    print("\n混淆矩阵（行=真值，列=预测）：")
    print("  " + " ".join(f"{label[:10]:>12}" for label in labels))
    for row_label in labels:
        cells = " ".join(f"{matrix[row_label][col]:>12}" for col in labels)
        print(f"{row_label[:12]:>14} {cells}")


if __name__ == "__main__":
    main()
