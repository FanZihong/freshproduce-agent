"""坏案例分析：逐条定位瓶颈到底在改写、分块、召回还是排序。

诊断思路——对同一条查询跑三个对照配置，用"排名是否改善"做因果判断：

    1. 关掉改写反而更好      -> 瓶颈在改写
    2. 换成段落分块更好      -> 瓶颈在分块粒度
    3. 黄金 chunk 没进候选池  -> 瓶颈在召回（embedding / BM25）
    4. 进了池但排到 top-k 外  -> 瓶颈在排序（融合 / 精排）

用法（在 agent/ 目录下执行）：
    python scripts/analyze_failures.py
    python scripts/analyze_failures.py --top-n 8 --embedder hash
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from eval.metrics import evaluate  # noqa: E402
from eval.runner import (  # noqa: E402
    ExperimentConfig,
    ExperimentRunner,
    load_dataset,
    map_dataset_to_strategy,
)

DEFAULT_DATASET = AGENT_ROOT / "eval" / "datasets" / "qa_golden.jsonl"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "retrieval_failure_analysis.md"

# 推荐配置（当前最优）与两个对照
PRIMARY = ExperimentConfig("最优配置", kind="vector", rewrite="rule", rerank="bge")
NO_REWRITE = ExperimentConfig("关闭改写", kind="vector", rewrite="off", rerank="bge")
PARAGRAPH = ExperimentConfig(
    "段落分块", kind="vector", rewrite="rule", rerank="bge", strategy="paragraph"
)


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def rank_of(gold: list[str], ids: list[str]) -> int | None:
    """黄金 chunk 在结果列表里的排名；没出现返回 None。"""
    gold_set = set(gold)
    for index, chunk_id in enumerate(ids, start=1):
        if chunk_id in gold_set:
            return index
    return None


def _better(candidate: int | None, baseline: int | None) -> bool:
    """candidate 的排名是否严格优于 baseline（None 视为最差）。"""
    if candidate is None:
        return False
    return baseline is None or candidate < baseline


def attribute(
    primary_rank: int | None,
    primary_pool_rank: int | None,
    no_rewrite_rank: int | None,
    paragraph_rank: int | None,
) -> tuple[str, str]:
    """返回 (瓶颈环节, 说明)。"""
    if _better(no_rewrite_rank, primary_rank):
        return "改写", f"关闭改写后排名从 {primary_rank or '未命中'} 改善到 {no_rewrite_rank}"
    if _better(paragraph_rank, primary_rank):
        return "分块", f"段落分块后排名从 {primary_rank or '未命中'} 改善到 {paragraph_rank}"
    if primary_pool_rank is None:
        return "召回", "黄金 chunk 没进候选池（检索/向量化没捞到）"
    if primary_rank is None:
        return "排序", f"进了候选池（池内第 {primary_pool_rank}）但没排进 top-k"
    return "排序", f"进了候选池（池内第 {primary_pool_rank}）但最终排在第 {primary_rank}"


def main() -> None:
    parser = argparse.ArgumentParser(description="检索坏案例分析")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--top-n", type=int, default=5, help="分析最差的 N 条")
    parser.add_argument("--ks", nargs="*", type=int, default=[1, 3, 5])
    parser.add_argument("--embedder", choices=["auto", "bge", "hash"], default="auto")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    mapped_paragraph, _ = map_dataset_to_strategy(dataset, "paragraph")
    runner = ExperimentRunner(embedder_kind=args.embedder, ks=args.ks)

    print(f"评测集：{args.dataset.name}（{len(dataset)} 条）")
    print(f"向量模型：{runner.embedder.name}\n")

    primary = runner.run(PRIMARY, dataset)
    no_rewrite = runner.run(NO_REWRITE, dataset)
    paragraph = runner.run(PARAGRAPH, mapped_paragraph)

    metrics = primary["metrics"]
    print(
        f"最优配置指标："
        + "  ".join(f"recall@{k}={metrics[f'recall@{k}']:.4f}" for k in args.ks)
        + f"  MRR={metrics['mrr']:.4f}"
    )
    baseline_metrics = evaluate(
        [{"retrieved": row["retrieved"], "relevant": row["relevant"]} for row in no_rewrite["rows"]],
        args.ks,
    )
    print(f"（对照：关闭改写 recall@5={baseline_metrics[f'recall@{max(args.ks)}']:.4f}）\n")

    no_rewrite_rows = {row["id"]: row for row in no_rewrite["rows"]}
    paragraph_rows = {row["id"]: row for row in paragraph["rows"]}
    # 段落分块的标准答案是按文本包含关系映射过的，比较时必须用映射后的 id
    paragraph_gold = {item["id"]: item["relevant_chunk_ids"] for item in mapped_paragraph}

    cases = []
    for row in primary["rows"]:
        gold = row["relevant"]
        primary_rank = rank_of(gold, row["retrieved"])
        record = {
            "id": row["id"],
            "question": row["question"],
            "search_query": row["search_query"],
            "gold": gold,
            "primary_rank": primary_rank,
            "pool_rank": rank_of(gold, row["pool"]),
            "no_rewrite_rank": rank_of(gold, no_rewrite_rows[row["id"]]["retrieved"]),
            "paragraph_rank": rank_of(
                paragraph_gold[row["id"]], paragraph_rows[row["id"]]["retrieved"]
            ),
            "topk": row["retrieved"][: max(args.ks)],
        }
        record["stage"], record["reason"] = attribute(
            record["primary_rank"],
            record["pool_rank"],
            record["no_rewrite_rank"],
            record["paragraph_rank"],
        )
        cases.append(record)

    # 最差排序：未命中优先，其次排名越靠后越差
    cases.sort(key=lambda item: (item["primary_rank"] is not None, -(item["primary_rank"] or 0)))
    worst = cases[: args.top_n]

    print(f"{'#':<4}{'用例':<8}{'提问':<20}{'黄金 chunk':<28}{'排名':<8}{'池内':<8}{'归因':<8}原因")
    for index, case in enumerate(worst, start=1):
        print(
            f"{index:<4}{case['id']:<8}{_pad(case['question'], 20)}"
            f"{_pad(', '.join(case['gold']), 28)}"
            f"{str(case['primary_rank'] or '未命中'):<8}{str(case['pool_rank'] or '-'):<8}"
            f"{_pad(case['stage'], 8)}{case['reason']}"
        )

    stage_counts: dict[str, int] = {}
    for case in cases:
        stage_counts[case["stage"]] = stage_counts.get(case["stage"], 0) + 1
    missed = sum(1 for case in cases if case["primary_rank"] is None)
    print(f"\n全部 {len(cases)} 条：未命中 top-{max(args.ks)} 共 {missed} 条；归因分布 {stage_counts}")

    # 反向对照：改写和分块各自帮到了谁
    rewrite_helped = [case for case in cases if _better(case["primary_rank"], case["no_rewrite_rank"])]
    rewrite_rescued = [
        case for case in rewrite_helped if case["no_rewrite_rank"] is None
    ]
    paragraph_changed = [
        case for case in cases if case["paragraph_rank"] != case["primary_rank"]
    ]
    print(
        f"改写帮助 {len(rewrite_helped)} 条（其中 {len(rewrite_rescued)} 条由未命中变命中）；"
        f"段落分块改变排名 {len(paragraph_changed)} 条"
    )

    lines = [
        "# 检索坏案例分析",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 评测集：{args.dataset.name}（{len(dataset)} 条）",
        f"- 最优配置：向量 + 规则改写 + CrossEncoder 精排",
        f"- 指标："
        + "，".join(f"recall@{k} {metrics[f'recall@{k}']:.4f}" for k in args.ks)
        + f"，MRR {metrics['mrr']:.4f}",
        "",
        f"## 最差的 {len(worst)} 条",
        "",
        "| # | 用例 | 提问 | 改写后 | 黄金 chunk | 最终排名 | 池内排名 | 关改写后 | 段落分块后 | 归因 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, case in enumerate(worst, start=1):
        lines.append(
            f"| {index} | {case['id']} | {case['question']} | {case['search_query']} | "
            f"{', '.join(case['gold'])} | {case['primary_rank'] or '未命中'} | "
            f"{case['pool_rank'] or '-'} | {case['no_rewrite_rank'] or '未命中'} | "
            f"{case['paragraph_rank'] or '未命中'} | {case['stage']} |"
        )
    lines += [
        "",
        "## 归因分布（全部用例）",
        "",
        f"- 未命中 top-{max(args.ks)}：{missed} / {len(cases)} 条",
    ]
    for stage, count in sorted(stage_counts.items(), key=lambda item: -item[1]):
        lines.append(f"- {stage}：{count} 条")
    lines += [
        "",
        "## 反向对照：改写与分块各自帮到了谁",
        "",
        f"- 改写帮助 {len(rewrite_helped)} 条（其中 {len(rewrite_rescued)} 条由未命中变为命中）",
        f"- 段落分块改变排名 {len(paragraph_changed)} 条",
    ]
    if rewrite_helped:
        lines += [
            "",
            "| 用例 | 提问 | 改写后 | 最优配置排名 | 关闭改写排名 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for case in rewrite_helped:
            lines.append(
                f"| {case['id']} | {case['question']} | {case['search_query']} | "
                f"{case['primary_rank'] or '未命中'} | {case['no_rewrite_rank'] or '未命中'} |"
            )
    lines += [
        "",
        "## 结论：召回低先查哪一环",
        "",
        "排查顺序固定为 **改写 → 分块 → 召回 → 排序**：",
        "",
        "1. 先看改写：多轮指代没补全的话，后面全错，而且关掉改写能立刻验证；",
        "2. 再看分块：答案是否被切到相邻窗口（换段落块能否召回）；",
        "3. 再看召回：黄金 chunk 是否进了候选池——没进说明 embedding/BM25 层面就没捞到，此时调融合与精排都是白费；",
        "4. 最后才看排序：进了池却没排进 top-k，才是融合权重和精排该解决的问题。",
        "",
        "一句话：**召回是上限，排序只是逼近上限；先修上限，再修逼近。**",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入：{args.report}")


if __name__ == "__main__":
    main()
