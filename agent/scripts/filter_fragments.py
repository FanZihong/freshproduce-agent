"""把标注语料拆成「信息自足」与「依赖上下文」两部分。

背景：JDDC 里的单句用户消息有大量碎片（只发订单号、"然后呢"、"你帮我看看"）。
这些句子单看没有独立意图，人工给它们打标签本质是"凭上下文猜"，会污染训练集。
正确做法是：
- 自足样本 → 进单句意图分类训练集；
- 碎片样本 → 走多轮样本（build_multiturn_samples.py）或在推理时走"澄清/追问"。

用法（在 agent/ 目录下执行）：
    python scripts/filter_fragments.py
    python scripts/filter_fragments.py --input data/intent/raw/calibrated_v1.jsonl \
        --review-csv data/intent/review/calibration_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DATA_DIR  # noqa: E402
from agent.rag.rewrite import KNOWN_PRODUCTS  # noqa: E402
from eval.metrics import cohens_kappa, label_agreement  # noqa: E402

RAW_DIR = DATA_DIR / "intent" / "raw"
DEFAULT_INPUT = RAW_DIR / "calibrated_v1.jsonl"
DEFAULT_FALLBACK = RAW_DIR / "prelabel_v1.jsonl"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "intent_fragment_split.md"

# 业务名词/动作：命中即认为这句话自带话题
BUSINESS_TERMS: tuple[str, ...] = (
    *KNOWN_PRODUCTS,
    "订单", "物流", "快递", "发票", "退款", "退货", "理赔", "赔付", "重量", "称重",
    "保存", "储存", "保鲜", "保质", "时效", "工单", "优惠券", "地址", "签收", "发货",
    "客服", "人工", "投诉", "差评", "换货", "保修", "维修", "售后", "品质", "变质",
    "发霉", "坏了", "烂了", "少称", "缺斤少两", "开票",
)

ORDER_PATTERN = re.compile(r"\d{6,}")
PLACEHOLDER_PATTERN = re.compile(r"\[[^\]]{1,12}\]")
POLITE_PATTERN = re.compile(r"^(谢谢|好的|好吧|嗯|哦|行|可以了|没事|没有了|没了|知道了|明白)")
FOLLOWUP_PATTERN = re.compile(r"^(那|然后|还有|所以|那么|另外|接着|这个|那个|它)")


def is_self_contained(text: str, *, short_length: int = 6) -> bool:
    """判断这句话是否自带话题、可以独立标注意图。"""
    cleaned = text.strip()
    has_business = any(term in cleaned for term in BUSINESS_TERMS)
    if has_business:
        return True

    has_order = bool(ORDER_PATTERN.search(cleaned))
    placeholder_only = bool(PLACEHOLDER_PATTERN.search(cleaned)) and len(
        PLACEHOLDER_PATTERN.sub("", cleaned).strip()
    ) <= 4
    if placeholder_only:
        return False
    if has_order:
        # 只发订单号/订单号+礼貌语，缺少业务意图
        return False
    if len(cleaned) <= short_length:
        return False
    if POLITE_PATTERN.match(cleaned) or FOLLOWUP_PATTERN.match(cleaned):
        return False
    return True


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_labels(value) -> str:
    """兼容三种写法：列表 ['a','b']、字符串 'a|b'、字符串 'a,b'。"""
    if isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value]
    else:
        parts = [part.strip() for part in re.split(r"[|，,]", str(value or ""))]
    return "|".join(sorted(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="拆分自足样本与碎片样本")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--review-csv", type=Path, default=None, help="可选：统计自足样本上的采纳率")
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    source = args.input if args.input.exists() else DEFAULT_FALLBACK
    rows = [row for row in load_jsonl(source) if row.get("labels")]
    if not rows:
        print(f"没有可用标注数据（已尝试 {args.input} 与 {DEFAULT_FALLBACK}）")
        return

    self_contained = [row for row in rows if is_self_contained(str(row.get("text") or ""))]
    fragments = [row for row in rows if not is_self_contained(str(row.get("text") or ""))]

    self_path = args.out_dir / "selfcontained_v1.jsonl"
    frag_path = args.out_dir / "context_dependent_v1.jsonl"
    for path, payload in ((self_path, self_contained), (frag_path, fragments)):
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in payload) + "\n",
            encoding="utf-8",
        )

    print(f"输入：{source.name}，共 {len(rows)} 条")
    print(f"  信息自足：{len(self_contained)} 条（{len(self_contained) / len(rows):.1%}）→ {self_path.name}")
    print(f"  依赖上下文：{len(fragments)} 条（{len(fragments) / len(rows):.1%}）→ {frag_path.name}")

    self_counts = Counter((row["labels"] or ["unknown"])[0] for row in self_contained)
    print("\n自足样本的类别分布：")
    for label, count in self_counts.most_common():
        print(f"  {label:<18}{count}")

    # 可选：只在自足样本上统计预标注采纳率（比全量口径更能代表真实质量）
    adoption = None
    kappa = None
    agreement = None
    if args.review_csv and args.review_csv.exists():
        # 只统计"信息自足"的那部分样本，并直接用 CSV 里的两列做对比：
        # labels = 预标注结果，reviewed_labels = 人工校准结果
        self_ids = {row["id"] for row in self_contained}
        pairs: list[tuple[str, str]] = []
        with args.review_csv.open(encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                if raw.get("id") in self_ids and (raw.get("reviewed_labels") or "").strip():
                    pairs.append(
                        (
                            normalize_labels(raw.get("labels")),
                            normalize_labels(raw.get("reviewed_labels")),
                        )
                    )
        if pairs:
            left = [left_label for left_label, _ in pairs]
            right = [right_label for _, right_label in pairs]
            confirmed = sum(1 for a, b in zip(left, right) if a == b)
            adoption = confirmed / len(pairs)
            kappa = cohens_kappa(left, right)
            agreement = label_agreement(left, right)
            print(
                f"\n自足样本上的人工校准：{len(pairs)} 条，"
                f"一致 {confirmed} 条 → 采纳率 {adoption:.1%}，"
                f"完全一致率 {agreement:.1%}，Cohen's kappa {kappa:.3f}"
            )
            print(
                "注意：校准清单来自「低置信度全量 + 高置信度抽样」，这批是最难判的样本，"
                "所以这两个数字是悲观下限，不代表全部 7227 条的质量。"
            )
            print("（kappa ≥ 0.75 表示体系边界清晰；低于该值建议先改 taxonomy）")

    lines = [
        "# 语料碎片拆分报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 输入文件：{source.name}（{len(rows)} 条）",
        f"- 信息自足：{len(self_contained)} 条（{len(self_contained) / len(rows):.1%}）",
        f"- 依赖上下文：{len(fragments)} 条（{len(fragments) / len(rows):.1%}）",
        "",
        "## 自足样本类别分布",
        "",
        "| 类别 | 条数 |",
        "| --- | --- |",
    ]
    lines += [f"| {label} | {count} |" for label, count in self_counts.most_common()]
    if adoption is not None:
        lines += [
            "",
            f"- 自足样本上的预标注采纳率：{adoption:.1%}",
            f"- 自足样本上的人工一致性：完全一致 {agreement:.1%}，Cohen's kappa {kappa:.3f}",
        ]
    lines += [
        "",
        "## 结论",
        "",
        "碎片样本单看没有独立意图，人工给它们打标签属于凭上下文猜测；",
        "应改为：进多轮样本（带 context）或在推理时走澄清追问，而不是进单句分类训练集。",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告：{args.report}")


if __name__ == "__main__":
    main()
