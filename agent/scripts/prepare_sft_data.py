"""构造意图识别的 SFT 训练数据：多标签编码、长尾处理、域外配比。

产出（默认写到数据目录下）：
- train_sft.jsonl / dev_sft.jsonl / test_sft.jsonl：主用方案（两段式）的指令数据；
- train_sft_split.jsonl：对照方案（每个标签一条样本 + 多标签样本集）；
- sft_stats.json / class_weights.json：统计与类别权重；
- eval/reports/intent_sft_report.md：数据层对比与结论。

用法（在 agent/ 目录下执行）：
    python scripts/prepare_sft_data.py
    python scripts/prepare_sft_data.py --multi-label split          # 只出对照方案
    python scripts/prepare_sft_data.py --label-format name          # 目标串用中文名
    python scripts/prepare_sft_data.py --oversample-target 300      # 长尾过采样到 300
    python scripts/prepare_sft_data.py --ood-min 0.08 --ood-max 0.15

设计取舍（面试会被问）：
- 训练提示词不带 few-shot 例句：否则模型在背提示词，而不是学"给体系就能判类"；
- dev/test 一律两段式、不做重采样：评测必须落在真实分布上，动了就不可比；
- 长尾优先用类别权重，过采样只作为可选开关（重复样本容易过拟合）。
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
from agent.intent.baseline import build_messages, build_training_system_prompt  # noqa: E402
from agent.intent.target import TARGET_STYLES, format_target, parse_target  # noqa: E402
from agent.intent.taxonomy import load_taxonomy  # noqa: E402
from scripts.rebalance_intent_dataset import compute_class_weights  # noqa: E402

DATASET_DIRS: tuple[Path, ...] = (
    DATA_DIR / "intent" / "dataset_v1_balanced",
    DATA_DIR / "intent" / "dataset_v1",
)
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "intent_sft_report.md"
SPLITS = ("train", "dev", "test")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _width(text: str) -> int:
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )


def resolve_dataset_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in DATASET_DIRS:
        if (candidate / "train.jsonl").exists():
            return candidate
    raise SystemExit(f"找不到数据集，预期位置：{DATASET_DIRS[0]}")


def primary_label(row: dict) -> str:
    labels = row.get("labels") or []
    return labels[0] if labels else "unknown"


def build_record(
    row: dict,
    labels: list[str],
    strategy: str,
    system_prompt: str,
    taxonomy,
    style: str,
    max_labels: int,
) -> dict:
    """把一行数据渲染成一条 SFT 样本（Qwen chat 格式：system / user / assistant）。"""
    target = format_target(labels, style=style, taxonomy=taxonomy, max_labels=max_labels)
    # 用户轮的收尾提示必须和目标格式一致，否则会教出自相矛盾的输出习惯
    hint = "请输出 JSON：" if style == "json" else "请输出意图标签："
    messages = build_messages(
        row.get("text", ""),
        row.get("context"),
        taxonomy=taxonomy,
        system_prompt=system_prompt,
        output_hint=hint,
    )
    messages.append({"role": "assistant", "content": target})
    return {
        "id": row.get("id"),
        "text": row.get("text", ""),
        "context": row.get("context"),
        "labels": list(labels),
        "target": target,
        "strategy": strategy,
        "messages": messages,
        "source": row.get("source"),
        "dup": 1,
    }


def build_two_part(rows: list[dict], system_prompt: str, taxonomy, style: str, max_labels: int) -> list[dict]:
    """方案 A：一条样本输出主+辅（与评测口径一致，能表达多标签）。"""
    return [
        build_record(row, list(row.get("labels") or []), "two_part", system_prompt, taxonomy, style, max_labels)
        for row in rows
        if row.get("labels")
    ]


def build_split(rows: list[dict], system_prompt: str, taxonomy, style: str, max_labels: int) -> list[dict]:
    """方案 B：每个标签一条样本（多标签行拆成两条同文本、不同目标的样本）。"""
    records: list[dict] = []
    for row in rows:
        for label in row.get("labels") or []:
            records.append(
                build_record(row, [label], "split", system_prompt, taxonomy, style, max_labels)
            )
    return records


def oversample(records: list[dict], target: int, seed: int = 42) -> tuple[list[dict], dict]:
    """把样本数少于 target 的类别复制到 target（只用于训练集）。"""
    if target <= 0:
        return records, {"enabled": False, "added": 0}
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["labels"][0], []).append(record)

    added: Counter = Counter()
    result = list(records)
    for label, group in grouped.items():
        missing = target - len(group)
        if missing <= 0:
            continue
        for index in range(missing):
            clone = dict(group[index % len(group)])
            clone["dup"] = int(clone.get("dup", 1)) + 1
            clone["id"] = f"{clone['id']}#dup{clone['dup']}"
            result.append(clone)
            added[label] += 1
    rng.shuffle(result)
    return result, {"enabled": True, "target": target, "added": int(sum(added.values())), "by_label": dict(added)}


def adjust_ood(
    records: list[dict], ood_label: str, min_ratio: float, max_ratio: float, seed: int = 42
) -> tuple[list[dict], dict]:
    """把域外样本（本体系里是 chitchat）占比拉进 [min_ratio, max_ratio]。"""
    rng = random.Random(seed)
    total = len(records)
    current = sum(1 for record in records if record["labels"][0] == ood_label)
    stats = {
        "label": ood_label,
        "before": round(current / total, 4) if total else 0.0,
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
        "added": 0,
        "dropped": 0,
    }

    if min_ratio > 0 and total and current / total < min_ratio:
        # 解 (current + x) / (total + x) >= min_ratio
        need = int((min_ratio * total - current) / (1 - min_ratio) + 0.9999)
        pool = [record for record in records if record["labels"][0] == ood_label]
        for index in range(max(0, need)):
            clone = dict(pool[index % len(pool)])
            clone["dup"] = int(clone.get("dup", 1)) + 1
            clone["id"] = f"{clone['id']}#ood{clone['dup']}"
            records = records + [clone]
            stats["added"] += 1

    if max_ratio > 0:
        ood_records = [record for record in records if record["labels"][0] == ood_label]
        other = len(records) - len(ood_records)
        allowed = int(max_ratio * other / (1 - max_ratio)) if max_ratio < 1 else len(ood_records)
        if allowed < 1:
            # 其他类太少时按上限裁会把域外类整个删光，得不偿失，这里保留原样并记录
            stats["note"] = "其他类别样本过少，按上限裁剪会把域外类删空，本次不裁剪"
        elif len(ood_records) > allowed:
            keep = set(id(record) for record in rng.sample(ood_records, allowed))
            records = [record for record in records if record["labels"][0] != ood_label or id(record) in keep]
            stats["dropped"] = len(ood_records) - len(keep)

    final_total = len(records)
    final = sum(1 for record in records if record["labels"][0] == ood_label)
    stats["after"] = round(final / final_total, 4) if final_total else 0.0
    stats["count_after"] = final
    return records, stats


def distribution(records: list[dict], labels: list[str]) -> dict[str, int]:
    counter = Counter(record["labels"][0] for record in records)
    return {label: counter.get(label, 0) for label in labels}


def target_conflicts(records: list[dict]) -> int:
    """**同一份输入**（文本 + 上下文）对应多个不同目标串的条数——这才是真正的标签冲突。

    只看文本会误报：多轮语料里"就是这个订单"在"客服在不在"之后和在无上下文时
    本该有不同标签，上下文不同就不是同一条输入。
    """
    seen: dict[tuple, set[str]] = {}
    for record in records:
        key = (
            record["text"],
            json.dumps(record.get("context"), ensure_ascii=False, sort_keys=True),
        )
        seen.setdefault(key, set()).add(record["target"])
    return sum(1 for targets in seen.values() if len(targets) > 1)


def strategy_stats(records: list[dict], labels: list[str]) -> dict:
    per_class = distribution(records, labels)
    positives = Counter(label for record in records for label in record["labels"])
    non_empty = [count for count in per_class.values() if count]
    return {
        "samples": len(records),
        "multi_label_rows": sum(1 for record in records if len(record["labels"]) > 1),
        "target_conflicts": target_conflicts(records),
        "avg_target_chars": round(
            sum(len(record["target"]) for record in records) / len(records), 1
        )
        if records
        else 0.0,
        "min_class": min(non_empty) if non_empty else 0,
        "max_class": max(per_class.values()) if per_class else 0,
        "per_class_primary": per_class,
        "per_class_positive": {label: positives.get(label, 0) for label in labels},
    }


def write_report(
    path: Path,
    dataset_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    labels: list[str],
    taxonomy,
    sizes: dict[str, int],
    comparison: list[tuple[str, dict]],
    train_dist: dict[str, int],
    weights: dict[str, float],
    ood_stats: dict,
    oversample_stats: dict,
    notes: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 意图 SFT 数据构造报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据来源：{dataset_dir}（train {sizes.get('train', 0)} / dev {sizes.get('dev', 0)} / test {sizes.get('test', 0)}）",
        f"- 输出目录：{out_dir}",
        f"- 目标串风格：{args.label_format}；训练提示词：{args.prompt_detail}",
        f"- 类别体系：{len(labels)} 类（v{taxonomy.version}，更新时间 {taxonomy.updated}）",
        "",
        "## 一、多标签编码：两种方案的数据层对比",
        "",
        "（前两行同源、都未重采样，避免被长尾处理干扰；第三行是最终用于训练的数据）",
        "",
        "| 方案 | 训练样本数 | 可表达多标签的样本 | 同输入目标冲突 | 平均目标字符数 | 每类最少/最多样本 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, stats in comparison:
        lines.append(
            f"| {name} | {stats['samples']} | {stats['multi_label_rows']} | "
            f"{stats['target_conflicts']} | {stats['avg_target_chars']} | "
            f"{stats['min_class']} / {stats['max_class']} |"
        )
    lines += [
        "",
        "怎么读这张表：",
        "- **同输入目标冲突**只出现在拆分方案：同一段输入（文本 + 上下文完全一致）"
        "在不同样本里被要求输出不同标签，这种自相矛盾的监督信号会来回拉扯小模型；"
        "两段式天然为 0。",
        "- 拆分方案的辅助标签变成了独立样本，推理时模型**只能吐一个标签**，"
        "多标签能力直接丢失，所以它更适合作为对照实验，而不是主线。",
        "",
        "## 二、长尾处理",
        "",
        f"- 过采样：{'开启，目标每类 ' + str(oversample_stats.get('target')) + ' 条，新增 ' + str(oversample_stats.get('added')) + ' 条' if oversample_stats.get('enabled') else '未开启（默认只算类别权重，重复样本容易过拟合）'}",
        "",
        "| 类别 | 训练样本数（主标签） | 类别权重（loss 加权用） |",
        "| --- | --- | --- |",
    ]
    for label in labels:
        lines.append(f"| {label} | {train_dist.get(label, 0)} | {weights.get(label, 0)} |")
    lines += [
        "",
        "权重口径：`总样本数 / (类别数 × 该类样本数)`，训练时作为 CrossEntropyLoss 的 "
        "`weight` 传入即可（`class_weights.json` 里按类别顺序给出）。",
        "",
        "## 三、域外样本配比",
        "",
        f"- 域外类别：`{ood_stats.get('label')}`（本体系里承担 fallback 职责）",
        f"- 调整前占比：{ood_stats.get('before')}；调整后：{ood_stats.get('after')}"
        f"（补入 {ood_stats.get('added')} 条，裁掉 {ood_stats.get('dropped')} 条）",
        f"- 目标区间：[{ood_stats.get('min_ratio')}, "
        + ("不设上限]" if ood_stats.get("max_ratio") == 0 else f"{ood_stats.get('max_ratio')}]"),
        "",
        "## 四、结论与下一步",
        "",
    ]
    lines += [f"- {note}" for note in notes] if notes else ["- 暂无"]
    lines += [
        "",
        "复现命令：",
        "",
        "```bash",
        "python scripts/prepare_sft_data.py " + " ".join(sys.argv[1:]),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入：{path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="构造意图识别 SFT 训练数据")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--label-format", choices=TARGET_STYLES, default="key")
    parser.add_argument("--prompt-detail", choices=["catalog", "compact", "minimal"], default="catalog")
    parser.add_argument("--multi-label", choices=["two-part", "split", "both"], default="both")
    parser.add_argument("--oversample-target", type=int, default=0, help="长尾类过采样到多少条（0=不做）")
    parser.add_argument("--ood-min", type=float, default=0.05, help="域外样本占比下限")
    parser.add_argument("--ood-max", type=float, default=0.0, help="域外样本占比上限（0=不设上限）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    dataset_dir = resolve_dataset_dir(args.data_dir)
    out_dir = args.out_dir or dataset_dir
    taxonomy = load_taxonomy()
    labels = list(taxonomy.keys)
    max_labels = int(taxonomy.multi_intent_policy.get("max_labels", 2))
    ood_label = "chitchat"

    splits = {name: load_jsonl(dataset_dir / f"{name}.jsonl") for name in SPLITS}
    if not splits["train"]:
        raise SystemExit(f"没找到训练数据：{dataset_dir / 'train.jsonl'}")

    system_prompt = build_training_system_prompt(
        taxonomy, target_style=args.label_format, max_labels=max_labels, detail=args.prompt_detail
    )
    print(f"数据集：{dataset_dir}")
    print(f"划分：train {len(splits['train'])} / dev {len(splits['dev'])} / test {len(splits['test'])}")
    print(f"训练提示词：{args.prompt_detail}（{len(system_prompt)} 字符，不含 few-shot 例句）")
    print(f"目标串风格：{args.label_format}；多标签方案：{args.multi_label}\n")

    train_two_part = build_two_part(splits["train"], system_prompt, taxonomy, args.label_format, max_labels)
    train_split = build_split(splits["train"], system_prompt, taxonomy, args.label_format, max_labels)
    # 对比表必须"同源"：先用未做任何重采样的两套编码比，再单独看长尾/域外调整后的结果
    base_two_part_stats = strategy_stats(train_two_part, labels)

    # dev/test 永远两段式、不重采样：评测口径要稳定可比
    dev_sft = build_two_part(splits["dev"], system_prompt, taxonomy, args.label_format, max_labels)
    test_sft = build_two_part(splits["test"], system_prompt, taxonomy, args.label_format, max_labels)

    train_two_part, oversample_stats = oversample(train_two_part, args.oversample_target, args.seed)
    train_two_part, ood_stats = adjust_ood(train_two_part, ood_label, args.ood_min, args.ood_max, args.seed)
    random.Random(args.seed).shuffle(train_two_part)  # 打散顺序，避免训练脚本忘了 shuffle

    train_dist = distribution(train_two_part, labels)
    weights = compute_class_weights(Counter(train_dist), labels)
    for record in train_two_part:
        record["sample_weight"] = weights.get(record["labels"][0], 1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train_sft.jsonl", train_two_part)
    write_jsonl(out_dir / "dev_sft.jsonl", dev_sft)
    write_jsonl(out_dir / "test_sft.jsonl", test_sft)
    if args.multi_label in {"split", "both"}:
        write_jsonl(out_dir / "train_sft_split.jsonl", train_split)
    (out_dir / "class_weights.json").write_text(
        json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    two_part_stats = strategy_stats(train_two_part, labels)
    split_stats = strategy_stats(train_split, labels)
    stats = {
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "label_format": args.label_format,
        "prompt_detail": args.prompt_detail,
        "system_prompt_chars": len(system_prompt),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "sft_sizes": {
            "train": len(train_two_part),
            "dev": len(dev_sft),
            "test": len(test_sft),
            "train_split": len(train_split),
        },
        "two_part": two_part_stats,
        "two_part_before_resample": base_two_part_stats,
        "split": split_stats,
        "class_weights": weights,
        "oversample": oversample_stats,
        "ood": ood_stats,
    }
    (out_dir / "sft_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 冒烟：目标串必须能解析回原标签，否则训练数据本身就是错的
    bad = 0
    for record in train_two_part:
        try:
            parsed, _, _ = parse_target(record["target"], labels, taxonomy=taxonomy, max_labels=max_labels)
        except ValueError:
            parsed = ()
        if set(parsed) != set(record["labels"]):
            bad += 1
    print(f"往返校验（目标串 -> 标签）：{len(train_two_part) - bad}/{len(train_two_part)} 一致")
    if bad:
        print(f"[warn] {bad} 条往返不一致，先修数据再训练")

    print("\n两种多标签方案（训练集）：")
    header = (
        _pad("方案", 16)
        + _pad("样本数", 10)
        + _pad("多标签", 10)
        + _pad("目标冲突", 12)
        + _pad("平均目标字符", 16)
        + _pad("最少类", 10)
    )
    print(header)
    print("-" * _width(header))
    print(
        _pad("两段式（未重采样）", 16)
        + _pad(str(base_two_part_stats["samples"]), 10)
        + _pad(str(base_two_part_stats["multi_label_rows"]), 10)
        + _pad(str(base_two_part_stats["target_conflicts"]), 12)
        + _pad(str(base_two_part_stats["avg_target_chars"]), 16)
        + _pad(str(base_two_part_stats["min_class"]), 10)
    )
    print(
        _pad("拆分（未重采样）", 16)
        + _pad(str(split_stats["samples"]), 10)
        + _pad(str(split_stats["multi_label_rows"]), 10)
        + _pad(str(split_stats["target_conflicts"]), 12)
        + _pad(str(split_stats["avg_target_chars"]), 16)
        + _pad(str(split_stats["min_class"]), 10)
    )
    print(
        _pad("两段式（实际训练）", 16)
        + _pad(str(two_part_stats["samples"]), 10)
        + _pad(str(two_part_stats["multi_label_rows"]), 10)
        + _pad(str(two_part_stats["target_conflicts"]), 12)
        + _pad(str(two_part_stats["avg_target_chars"]), 16)
        + _pad(str(two_part_stats["min_class"]), 10)
    )

    imbalance = max(train_dist.values()) / max(1, min(train_dist.values()))
    print(
        f"\n训练集类别比（最大/最小）：{imbalance:.1f} : 1；"
        f"域外（{ood_label}）占比 {ood_stats['after']:.1%}"
    )
    print(f"输出：{out_dir}/train_sft.jsonl（{len(train_two_part)} 条）、dev_sft.jsonl、test_sft.jsonl"
          + ("、train_sft_split.jsonl" if args.multi_label in {"split", "both"} else ""))

    notes = [
        f"主线用两段式：与 `scripts/eval_intent.py` 的评测口径一致，且能表达多标签"
        f"（训练集里 {two_part_stats['multi_label_rows']} 条多标签样本）。",
        f"拆分方案仅作对照：它有 {split_stats['target_conflicts']} 处同文本目标冲突"
        f"（对比表里两套编码同源，避免被重采样干扰），"
        "推理阶段只能输出单标签，多标签能力会退化。",
        f"类别不均衡收敛到 {imbalance:.1f} : 1，训练时用 `class_weights.json` 做 loss 加权"
        "（比复制样本更稳，重复样本容易过拟合长尾）。",
        f"域外样本（{ood_label}）占比 {ood_stats['after']:.1%}：dev/test 的真实占比也在 "
        "10%–15% 之间，训练集跟着走才能让模型在评测集上既不乱拒答、也不硬猜。",
        "下一步：`scripts/train_intent.py` 用 Qwen2.5-0.5B-Instruct + LoRA 训练，"
        "保证训练与推理用的是同一段 system prompt（都在本脚本产出里）。",
    ]

    if not args.no_report:
        write_report(
            args.report,
            dataset_dir,
            out_dir,
            args,
            labels,
            taxonomy,
            {name: len(rows) for name, rows in splits.items()},
            [
                ("两段式（主+辅）", base_two_part_stats),
                ("拆分（每标签一条）", split_stats),
                ("两段式（重采样后，实际训练）", two_part_stats),
            ],
            train_dist,
            weights,
            ood_stats,
            oversample_stats,
            notes,
        )


if __name__ == "__main__":
    main()
