"""多意图路由策略对比：priority（优先动作型）vs parallel（并行两个分支）。

两个策略只影响**路由**，不影响识别指标——所以这里的核心指标是覆盖，不是 F1：
- 主分支命中率：预测主标签对应的分支与真值主标签一致；
- 次诉求覆盖率：真值的次标签也被路由到（parallel 应接近 1，priority 天然偏低）；
- 双诉求全覆盖率：两条都被覆盖（等价于多标签场景下的"达成率"代理指标）。

为什么不用 test 集里的多标签样本：那边只有 9 条（dev 5 条），比不出任何差异。
所以用自造的多意图金标集 `eval/datasets/multi_intent.jsonl`（50 条，覆盖 8 种常见组合）。
端到端"目标达成率"要等接进图之后用对话验收用例测，这里先给分支层指标。

用法（在 agent/ 目录下执行）：
    python scripts/eval_multi_intent.py                              # 规则版打通流程
    python scripts/eval_multi_intent.py --backend fewshot            # 基线
    python scripts/eval_multi_intent.py --backend local --local-dir data/intent/models/v2
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

from agent.intent.multi_intent import MULTI_STRATEGIES, coverage, plan  # noqa: E402
from agent.intent.target import normalize_labels  # noqa: E402
from agent.intent.taxonomy import load_taxonomy  # noqa: E402
from eval.metrics import exact_match_ratio, macro_f1, multilabel_f1  # noqa: E402
from scripts.eval_ood import build_backend  # noqa: E402  复用后端构造，避免两处维护

DEFAULT_DATASET = AGENT_ROOT / "eval" / "datasets" / "multi_intent.jsonl"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "intent_multi.md"
DEFAULT_JSON = AGENT_ROOT / "eval" / "reports" / "intent_multi.json"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(records: list[dict], strategy: str, taxonomy) -> dict:
    """按策略算覆盖指标 + 识别指标（识别指标与策略无关，只是顺带记录）。"""
    primary_hits = secondary_hits = both_hits = 0
    secondary_total = 0
    gold_sets: list[list[str]] = []
    pred_sets: list[list[str]] = []
    gold_primary: list[str] = []
    pred_primary: list[str] = []
    for record in records:
        gold = list(normalize_labels(record["gold"], set(taxonomy.keys)))
        predicted = list(normalize_labels(record["pred"], set(taxonomy.keys)))
        gold_sets.append(gold)
        pred_sets.append(predicted)
        gold_primary.append(gold[0] if gold else "chitchat")
        pred_primary.append(predicted[0] if predicted else "<未识别>")

        result = plan(predicted, strategy=strategy, taxonomy=taxonomy)
        metrics = coverage(result, gold, taxonomy)
        primary_hits += 1 if metrics["primary_hit"] else 0
        if metrics["secondary_covered"] is not None:
            secondary_total += 1
            secondary_hits += 1 if metrics["secondary_covered"] else 0
        both_hits += 1 if metrics["both_covered"] else 0

    total = len(records)
    labels = list(taxonomy.keys)
    return {
        "strategy": strategy,
        "samples": total,
        "primary_hit_rate": round(primary_hits / total, 4) if total else 0.0,
        "secondary_coverage": (
            round(secondary_hits / secondary_total, 4) if secondary_total else None
        ),
        "both_covered_rate": round(both_hits / total, 4) if total else 0.0,
        "macro_f1": macro_f1(gold_primary, pred_primary, labels),
        "exact_match": exact_match_ratio(gold_sets, pred_sets),
        "multilabel": multilabel_f1(gold_sets, pred_sets, labels),
    }


def print_table(rows: list[tuple[str, dict]]) -> None:
    header = (
        f"{'后端 / 策略':<34}{'主分支命中':<12}{'次诉求覆盖':<12}{'双诉求覆盖':<12}"
        f"{'多标签完全一致':<14}"
    )
    print(header)
    print("-" * 88)
    for name, metrics in rows:
        secondary = metrics["secondary_coverage"]
        print(
            f"{name:<34}{metrics['primary_hit_rate']:<12.4f}"
            f"{(f'{secondary:.4f}' if secondary is not None else '-'):<12}"
            f"{metrics['both_covered_rate']:<12.4f}{metrics['exact_match']:<14.4f}"
        )


def write_report(
    path: Path,
    args: argparse.Namespace,
    backend_label: str,
    results: list[tuple[str, dict]],
    records: list[dict],
    taxonomy,
    verdict: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 多意图路由策略对比报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 后端：{backend_label}",
        f"- 金标集：{args.dataset.name}（{len(records)} 条多意图问句）",
        f"- 策略：priority（优先动作型，取 routing_priority 最靠前的分支）/ "
        f"parallel（两个分支都执行，回复合并）",
        "",
        "## 结果",
        "",
        "| 后端 / 策略 | 主分支命中率 | 次诉求覆盖率 | 双诉求覆盖率 | 多标签完全一致率 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, metrics in results:
        secondary = metrics["secondary_coverage"]
        lines.append(
            f"| {name} | {metrics['primary_hit_rate']:.4f} | "
            f"{(f'{secondary:.4f}' if secondary is not None else '-')} | "
            f"{metrics['both_covered_rate']:.4f} | {metrics['exact_match']:.4f} |"
        )
    lines += [
        "",
        "> 识别指标（macro-F1 / 多标签 F1）与策略无关：策略只决定「走哪些分支」，"
        "不改变模型输出的标签集合。所以策略要靠覆盖指标比，而不是靠 F1。",
        "",
        "## 逐条明细（预测 vs 真值 + 分支覆盖）",
        "",
        "| id | 问句 | 真值 | 预测 | priority 分支 | parallel 分支 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        predicted = list(normalize_labels(record["pred"], set(taxonomy.keys)))
        primary_plan = plan(predicted, "priority", taxonomy)
        parallel_plan = plan(predicted, "parallel", taxonomy)
        lines.append(
            f"| {record['id']} | {record['text'][:26]} | {'/'.join(record['gold'])} | "
            f"{'/'.join(predicted) or '-'} | {'→'.join(primary_plan.branches) or '-'} | "
            f"{'→'.join(parallel_plan.branches) or '-'} |"
        )
    lines += ["", "## 验收判定", ""]
    lines += [f"- {item}" for item in verdict]
    lines += [
        "",
        "复现命令：",
        "",
        "```bash",
        "python scripts/eval_multi_intent.py " + " ".join(sys.argv[1:]),
        "```",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入：{path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="多意图路由策略对比")
    parser.add_argument("--backend", choices=["rule", "fewshot", "local"], default="rule")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--local-dir", type=Path, default=None)
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    taxonomy = load_taxonomy()
    rows = load_jsonl(args.dataset)
    if not rows:
        raise SystemExit(f"金标集为空：{args.dataset}")
    if args.limit:
        rows = rows[: args.limit]
    print(f"多意图金标集：{args.dataset.name}（{len(rows)} 条）")
    print("组合分布：" + "，".join(
        f"{'+'.join(key)}×{value}"
        for key, value in Counter(tuple(row["labels"]) for row in rows).most_common()
    ))

    predict, backend_label = build_backend(args.backend, args)
    records: list[dict] = []
    for index, row in enumerate(rows, start=1):
        labels, confidence, error = predict(row["text"], row.get("context"))
        records.append(
            {
                "id": row["id"],
                "text": row["text"],
                "gold": list(row["labels"]),
                "pred": list(labels),
                "confidence": confidence,
                "error": error,
            }
        )
        if index % 25 == 0:
            print(f"  已处理 {index}/{len(rows)} 条")

    results = [
        (f"{backend_label} · {strategy}", evaluate(records, strategy, taxonomy))
        for strategy in MULTI_STRATEGIES
    ]
    print()
    print_table(results)

    by_strategy = {metrics["strategy"]: metrics for _, metrics in results}
    priority, parallel = by_strategy["priority"], by_strategy["parallel"]
    verdict: list[str] = [
        f"识别层：多标签完全一致率 {priority['exact_match']:.4f}、"
        f"多标签 macro-F1 {priority['multilabel']['macro']:.4f}"
        f"（这部分与策略无关，是模型能力上限）"
    ]
    gain = parallel["both_covered_rate"] - priority["both_covered_rate"]
    verdict.append(
        f"priority 双诉求覆盖率 {priority['both_covered_rate']:.4f}，"
        f"parallel {parallel['both_covered_rate']:.4f}（提升 {gain:+.4f}）"
    )
    if gain <= 0.01:
        verdict.append(
            f"两种策略这次几乎完全相同——因为该后端基本只输出单标签"
            f"（次诉求覆盖率 {priority['secondary_coverage']}），没有第二个标签就没有第二条分支。"
            "策略对比要在 few-shot / local 这类能吐两个标签的后端上才有意义。"
        )
    else:
        verdict.append(
            f"parallel 把双诉求覆盖率从 {priority['both_covered_rate']:.4f} 提到 "
            f"{parallel['both_covered_rate']:.4f}（{gain:+.4f}）：代价是回复更长、要跑两个分支。"
        )
    verdict.append(
        "策略建议：priority 适合「动作型 + 咨询型」组合（先给赔付结论，次诉求补一句入口）；"
        "parallel 适合「两个动作型」组合（都要真做事）。上线前以本报告的数字为准。"
    )
    print("\n验收判定：")
    for item in verdict:
        print(f"  - {item}")

    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backend": backend_label,
        "dataset": args.dataset.name,
        "samples": len(records),
        "strategies": {metrics["strategy"]: metrics for _, metrics in results},
        "verdict": verdict,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"指标 JSON：{args.json}")
    if not args.no_report:
        write_report(args.report, args, backend_label, results, records, taxonomy, verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
