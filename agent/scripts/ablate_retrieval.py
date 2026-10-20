"""检索消融实验：一次性跑完所有对比，输出一张 markdown 报告。

用法（在 agent/ 目录下执行）：
    python scripts/ablate_retrieval.py                     # 全部四组
    python scripts/ablate_retrieval.py --groups ladder k   # 只跑指定组
    python scripts/ablate_retrieval.py --embedder hash     # 不下载模型，快速看流程
    python scripts/ablate_retrieval.py --no-rerank-model   # 跳过 CrossEncoder 组
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from eval.runner import (  # noqa: E402
    ExperimentConfig,
    ExperimentRunner,
    load_dataset,
    map_dataset_to_strategy,
)

DEFAULT_DATASET = AGENT_ROOT / "eval" / "datasets" / "qa_golden.jsonl"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "retrieval_ablation.md"
DEFAULT_CACHE = AGENT_ROOT / "eval" / "reports" / "rewrite_cache.json"


def ladder_hybrid(use_rerank: bool) -> list[ExperimentConfig]:
    """混合路线：逐级叠加，每次只加一个环节。"""
    configs = [
        ExperimentConfig("① BM25 单路", kind="bm25", note="关键词检索起点"),
        ExperimentConfig("② + 向量（RRF 混合）", kind="hybrid", note="加入稠密检索"),
        ExperimentConfig("③ + 规则改写", kind="hybrid", rewrite="rule", note="多轮指代消解"),
    ]
    if use_rerank:
        configs.append(
            ExperimentConfig(
                "④ + CrossEncoder 精排", kind="hybrid", rewrite="rule", rerank="bge", note="语义精排"
            )
        )
    return configs


def ladder_vector(use_rerank: bool) -> list[ExperimentConfig]:
    """向量路线：同样的阶梯，但不掺 BM25。"""
    configs = [
        ExperimentConfig("① 向量单路", kind="vector", note="稠密检索起点"),
        ExperimentConfig("② + 规则改写", kind="vector", rewrite="rule", note="多轮指代消解"),
    ]
    if use_rerank:
        configs.append(
            ExperimentConfig(
                "③ + CrossEncoder 精排", kind="vector", rewrite="rule", rerank="bge", note="语义精排"
            )
        )
    return configs


def group_rrf_k() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(f"RRF k={k}", kind="hybrid", rewrite="rule", rrf_k=k, note="融合参数扫描")
        for k in (10, 20, 30, 60, 100)
    ]


def group_chunking(use_rerank: bool) -> list[ExperimentConfig]:
    rerank = "bge" if use_rerank else "off"
    return [
        ExperimentConfig(
            f"分块={strategy}", kind="vector", rewrite="rule", rerank=rerank, strategy=strategy,
            note="rule=按规则条，paragraph=按段落窗口",
        )
        for strategy in ("rule", "paragraph")
    ]


def group_rerank() -> list[ExperimentConfig]:
    return [
        ExperimentConfig("重排=关闭", kind="vector", rewrite="rule", note="对照"),
        ExperimentConfig("重排=融合分数（0 成本）", kind="vector", rewrite="rule", rerank="score", note="基线"),
        ExperimentConfig("重排=CrossEncoder", kind="vector", rewrite="rule", rerank="bge", note="精排"),
    ]


def group_weights() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            f"权重 BM25:向量={label}", kind="hybrid", rewrite="rule", rrf_k=10,
            weights=weights, note="k 取 10（扫描中最优）",
        )
        for label, weights in (
            ("1:1", {"bm25": 1.0, "vector": 1.0}),
            ("1:2", {"bm25": 1.0, "vector": 2.0}),
            ("1:4", {"bm25": 1.0, "vector": 4.0}),
        )
    ]


GROUPS = {
    "ladder-hybrid": ("逐级增益（混合路线）", ladder_hybrid),
    "ladder-vector": ("逐级增益（向量路线）", ladder_vector),
    "k": ("RRF k 值扫描", group_rrf_k),
    "chunking": ("分块策略对比", group_chunking),
    "rerank": ("重排方式对比", group_rerank),
    "weights": ("融合权重扫描", group_weights),
}


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def _delta(current: float | None, baseline: float | None, *, is_baseline: bool = False) -> str:
    if current is None or baseline is None:
        return "—"
    if is_baseline:
        return "基线"
    return f"{current - baseline:+.4f}"


def render_group_table(results: list[dict], ks: list[int]) -> str:
    header = _pad("配置", 30) + "".join(_pad(f"recall@{k}", 12) for k in ks) + _pad("MRR", 10)
    header += _pad("Δrecall@5", 12) + "说明"
    lines = [header, "-" * _width(header)]
    baseline = results[0]["metrics"].get(f"recall@{max(ks)}") if results else None

    for index, payload in enumerate(results):
        metrics = payload["metrics"]
        config = payload["config"]
        recall_max = metrics.get(f"recall@{max(ks)}")
        line = _pad(config.name, 30)
        line += "".join(_pad(f"{metrics[f'recall@{k}']:.4f}", 12) for k in ks)
        line += _pad(f"{metrics['mrr']:.4f}", 10)
        line += _pad(_delta(recall_max, baseline, is_baseline=index == 0), 12)
        line += config.note
        lines.append(line)
    return "\n".join(lines)


def render_group_markdown(title: str, results: list[dict], ks: list[int]) -> list[str]:
    lines = [f"## {title}", ""]
    lines.append("| 配置 | " + " | ".join(f"recall@{k}" for k in ks) + " | MRR | Δrecall@5 | 说明 |")
    lines.append("| --- | " + " | ".join("---" for _ in ks) + " | --- | --- | --- |")
    baseline = results[0]["metrics"].get(f"recall@{max(ks)}") if results else None
    for index, payload in enumerate(results):
        metrics = payload["metrics"]
        config = payload["config"]
        cells = " | ".join(f"{metrics[f'recall@{k}']:.4f}" for k in ks)
        delta = _delta(metrics.get(f"recall@{max(ks)}"), baseline, is_baseline=index == 0)
        lines.append(f"| {config.name} | {cells} | {metrics['mrr']:.4f} | {delta} | {config.note} |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="检索消融实验")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ks", nargs="*", type=int, default=[1, 3, 5])
    parser.add_argument("--embedder", choices=["auto", "bge", "hash"], default="auto")
    parser.add_argument("--groups", nargs="*", default=list(GROUPS), choices=list(GROUPS))
    parser.add_argument("--no-rerank-model", action="store_true", help="跳过 CrossEncoder（耗时/占空间）")
    parser.add_argument("--no-cache", action="store_true", help="不使用 LLM 改写缓存")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    cache: dict = {}
    if not args.no_cache and DEFAULT_CACHE.exists():
        cache = json.loads(DEFAULT_CACHE.read_text(encoding="utf-8"))

    runner = ExperimentRunner(
        embedder_kind=args.embedder, ks=args.ks, rewrite_cache=None if args.no_cache else cache
    )
    print(f"评测集：{args.dataset.name}（{len(dataset)} 条）")
    print(f"向量模型：{runner.embedder.name}（维度 {runner.embedder.dim}）")
    print(f"实验分组：{', '.join(args.groups)}\n")

    use_rerank_model = not args.no_rerank_model
    markdown: list[str] = [
        "# 检索消融实验报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 评测集：{args.dataset.name}（{len(dataset)} 条）",
        f"- 向量模型：{runner.embedder.name}",
        "- 说明：每组只改一个变量，Δ 以组内第一行为基线",
        "",
    ]

    for key in args.groups:
        title, builder = GROUPS[key]
        if key in {"ladder-hybrid", "ladder-vector", "chunking"}:
            configs = builder(use_rerank_model)
        else:
            configs = builder()

        print(f"===== {title} =====")
        mapped_datasets: dict[str, list[dict]] = {}
        results = []
        for config in configs:
            dataset_for_run = dataset
            if config.strategy != "rule":
                if config.strategy not in mapped_datasets:
                    mapped_datasets[config.strategy], stats = map_dataset_to_strategy(
                        dataset, config.strategy
                    )
                    print(
                        f"  分块策略 {config.strategy}：标准答案已按文本包含关系映射"
                        f"（{stats['mapped_queries']}/{stats['total_queries']} 条可映射）"
                    )
                dataset_for_run = mapped_datasets[config.strategy]
            payload = runner.run(config, dataset_for_run)
            results.append(payload)
            metrics = payload["metrics"]
            print(
                f"  已跑：{config.name}  recall@5={metrics[f'recall@{max(args.ks)}']:.4f}  "
                f"MRR={metrics['mrr']:.4f}"
            )
        print()
        print(render_group_table(results, args.ks))
        print()
        markdown += render_group_markdown(title, results, args.ks)

    if not args.no_cache and cache:
        DEFAULT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"报告已写入：{args.report}")


if __name__ == "__main__":
    main()
