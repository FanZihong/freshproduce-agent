"""均衡意图数据集：高频类下采样 + 合成多标签样本 + 输出类别权重。

三件事：
1. 高频类下采样到上限（默认 400/类），缓解 13:1 的类别不平衡；
2. 用模板合成多标签样本（只写入 train，dev/test 保持真实数据用于诚实评测）；
3. 计算类别权重，供训练时使用。

用法（在 agent/ 目录下执行）：
    python scripts/rebalance_intent_dataset.py
    python scripts/rebalance_intent_dataset.py --max-per-class 400 --multi-label-count 240
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DATA_DIR  # noqa: E402
from agent.intent.taxonomy import load_taxonomy  # noqa: E402

DATASET_DIR = DATA_DIR / "intent" / "dataset_v1"
BALANCED_DIR = DATA_DIR / "intent" / "dataset_v1_balanced"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "intent_dataset_balanced.md"

# 多标签合成用的句式；{a} 是主体诉求，{b} 是附带诉求
MULTI_LABEL_TEMPLATES: tuple[str, ...] = (
    "{a}，而且{b}",
    "{a}，顺便{b}",
    "{a}。另外{b}",
    "{a}，还有{b}",
    "{a}，同时想{b}",
)

# 主体诉求与附带诉求的素材（按意图）
CLAUSE_BANK: dict[str, tuple[str, ...]] = {
    "bad_fruit_claim": ("车厘子坏了一半", "草莓到货发霉了", "蓝莓压坏了不少"),
    "short_weight": ("重量比标注少了一截", "少称了半斤", "数量少了一盒"),
    "urge_logistics": ("三天了还没发货", "快递一直不动", "想催一下派送"),
    "query_order": ("想知道订单到哪了", "想查一下物流", "看看什么时候能到"),
    "refund_consult": ("想问退款多久到账", "退款失败了想问问", "想知道退款进度"),
    "quality_qa": ("想知道怎么保存", "想问能放多久", "想知道能不能退"),
    "invoice": ("想开发票", "发票抬头要改", "想补开发票"),
    "complaint": ("想投诉客服", "准备给差评", "对服务很不满意"),
}

# 合理的组合（主体 = 动作型诉求，附带 = 另一件事）
PAIR_POOL: tuple[tuple[str, str], ...] = (
    ("bad_fruit_claim", "urge_logistics"),
    ("bad_fruit_claim", "refund_consult"),
    ("bad_fruit_claim", "invoice"),
    ("short_weight", "refund_consult"),
    ("short_weight", "urge_logistics"),
    ("urge_logistics", "invoice"),
    ("query_order", "invoice"),
    ("query_order", "refund_consult"),
    ("quality_qa", "invoice"),
    ("complaint", "human_handoff"),
    ("refund_consult", "human_handoff"),
    ("bad_fruit_claim", "human_handoff"),
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def primary_label(row: dict) -> str:
    labels = row.get("labels") or []
    return labels[0] if labels else "unknown"


def downsample(
    rows: list[dict],
    max_per_class: int,
    seed: int,
    overrides: dict[str, int] | None = None,
) -> list[dict]:
    """按主标签对高频类下采样；低频类全保留。overrides 可对单个类别设更低的配额。"""
    rng = random.Random(seed)
    overrides = overrides or {}
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(primary_label(row), []).append(row)

    kept: list[dict] = []
    for label, group in grouped.items():
        cap = overrides.get(label, max_per_class)
        if len(group) <= cap:
            kept.extend(group)
            continue
        kept.extend(rng.sample(group, cap))
    rng.shuffle(kept)
    return kept


def synthesize_multi_label(count: int, seed: int) -> list[dict]:
    """用模板合成多标签样本（只进训练集）。"""
    rng = random.Random(seed)
    samples: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    while len(samples) < count and attempts < count * 20:
        attempts += 1
        first, second = rng.choice(PAIR_POOL)
        clause_a = rng.choice(CLAUSE_BANK[first])
        if second == "human_handoff":
            clause_b = rng.choice(("想转人工处理", "麻烦帮我接人工", "要人工客服"))
        else:
            clause_b = rng.choice(CLAUSE_BANK[second])
        template = rng.choice(MULTI_LABEL_TEMPLATES)
        text = template.format(a=clause_a, b=clause_b)
        if text in seen:
            continue
        seen.add(text)
        samples.append(
            {
                "id": f"syn-multi-{len(samples) + 1:04d}",
                "text": text,
                "context": None,
                "labels": [first, second],
                "source": "synthetic_multilabel",
                "labeler": "template",
                "confidence": 1.0,
                "review_status": "synthetic",
            }
        )
    return samples


def compute_class_weights(counter: Counter, labels: list[str]) -> dict[str, float]:
    total = sum(counter.values()) or 1
    n_classes = len(labels) or 1
    return {
        label: round(total / (n_classes * counter.get(label, 1) or 1), 3) for label in labels
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="均衡意图数据集")
    parser.add_argument("--input-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=BALANCED_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-per-class", type=int, default=400, help="每类训练样本上限")
    parser.add_argument("--multi-label-count", type=int, default=240, help="合成的多标签样本数")
    parser.add_argument(
        "--chitchat-cap",
        type=int,
        default=300,
        help="域外/闲聊类的单独上限（这类过多会让模型倾向拒答）",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = load_jsonl(args.input_dir / "train.jsonl")
    dev = load_jsonl(args.input_dir / "dev.jsonl")
    test = load_jsonl(args.input_dir / "test.jsonl")
    if not train:
        print(f"没有找到训练数据：{args.input_dir / 'train.jsonl'}")
        return

    balanced_train = downsample(
        train, args.max_per_class, args.seed, overrides={"chitchat": args.chitchat_cap}
    )
    synthetic = synthesize_multi_label(args.multi_label_count, args.seed)
    balanced_train.extend(synthetic)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", balanced_train), ("dev", dev), ("test", test)):
        (args.out_dir / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    before = Counter(primary_label(row) for row in train)
    after = Counter(primary_label(row) for row in balanced_train)
    taxonomy = load_taxonomy()
    labels = [spec.key for spec in taxonomy.labels]
    weights = compute_class_weights(after, labels)

    stats = {
        "source": str(args.input_dir),
        "train_before": len(train),
        "train_after": len(balanced_train),
        "synthetic_multi_label": len(synthetic),
        "dev": len(dev),
        "test": len(test),
        "per_label_before": {label: before.get(label, 0) for label in labels},
        "per_label_after": {label: after.get(label, 0) for label in labels},
        "class_weights": weights,
        "multi_label_ratio": round(
            sum(1 for row in balanced_train if len(row.get("labels") or []) > 1) / len(balanced_train), 4
        ),
    }
    (args.out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"训练集：{len(train)} → {len(balanced_train)} 条（下采样 + 合成 {len(synthetic)} 条多标签）")
    print(f"dev {len(dev)} / test {len(test)} 保持不变（评测必须用真实数据）")
    print(f"多标签占比：{stats['multi_label_ratio']:.1%}")
    print(f"\n{'类别':<18}{'原始':<8}{'均衡后':<8}类别权重")
    for label in labels:
        print(f"{label:<18}{before.get(label, 0):<8}{after.get(label, 0):<8}{weights[label]}")

    imbalance = max(after.values()) / max(1, min(after.values()))
    print(f"\n最大/最小类样本比：{imbalance:.1f} : 1（原始 {max(before.values()) / max(1, min(before.values())):.1f} : 1）")

    lines = [
        "# 意图数据集均衡报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 训练集：{len(train)} → {len(balanced_train)} 条（每类上限 {args.max_per_class}）",
        f"- 合成多标签样本：{len(synthetic)} 条（只进 train）",
        f"- dev / test 保持真实数据：{len(dev)} / {len(test)} 条",
        f"- 多标签占比：{stats['multi_label_ratio']:.1%}",
        "",
        "| 类别 | 原始 | 均衡后 | 类别权重 |",
        "| --- | --- | --- | --- |",
    ]
    for label in labels:
        lines.append(
            f"| {label} | {before.get(label, 0)} | {after.get(label, 0)} | {weights[label]} |"
        )
    lines += ["", f"最大/最小类样本比：{imbalance:.1f} : 1"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n输出目录：{args.out_dir}")
    print(f"报告：{args.report}")


if __name__ == "__main__":
    main()
