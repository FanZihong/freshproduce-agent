"""检索评测：跑评测集，出 recall@k / MRR，并对比多种配置。

用法（在 agent/ 目录下执行）：
    python scripts/eval_retrieval.py                      # 六种配置对比
    python scripts/eval_retrieval.py --configs hybrid+rw  # 只看某一个配置
    python scripts/eval_retrieval.py --embedder hash      # 不下载模型
    python scripts/eval_retrieval.py --report eval/reports/retrieval_baseline.md
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

from eval.metrics import evaluate  # noqa: E402
from agent.rag.embedder import build_embedder  # noqa: E402
from agent.rag.fusion import reciprocal_rank_fusion  # noqa: E402
from agent.rag.hybrid_search import set_tokenizer  # noqa: E402
from agent.rag.reranker import build_reranker  # noqa: E402
from agent.rag.retriever import Retriever  # noqa: E402
from agent.rag.rewrite import RewriteResult, load_llm_config, rewrite_with_cache  # noqa: E402

DEFAULT_DATASET = AGENT_ROOT / "eval" / "datasets" / "qa_golden.jsonl"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "retrieval_baseline.md"
DEFAULT_JSON = AGENT_ROOT / "eval" / "reports" / "retrieval_baseline.json"
DEFAULT_REWRITE_CACHE = AGENT_ROOT / "eval" / "reports" / "rewrite_cache.json"

# 配置名 -> (检索方式, 改写方式, 重排方式, 展示名)
CONFIG_TABLE: dict[str, tuple[str, str, str, str]] = {
    "bm25": ("bm25", "off", "off", "BM25（无改写）"),
    "vector": ("vector", "off", "off", "向量（无改写）"),
    "hybrid": ("hybrid", "off", "off", "RRF 混合（无改写）"),
    "bm25+rw": ("bm25", "rule", "off", "BM25（+规则改写）"),
    "vector+rw": ("vector", "rule", "off", "向量（+规则改写）"),
    "hybrid+rw": ("hybrid", "rule", "off", "RRF 混合（+规则改写）"),
    # 生产口径：search_faq 走 FAQ_RERANK=score，即"混合 + 规则改写 + 分数重排"
    "hybrid+rw+score": ("hybrid", "rule", "score", "RRF 混合（+改写+分数重排）"),
    "vector+llmrw": ("vector", "llm", "off", "向量（+LLM 改写）"),
    "hybrid+llmrw": ("hybrid", "llm", "off", "RRF 混合（+LLM 改写）"),
    "vector+rw+score": ("vector", "rule", "score", "向量+改写+分数重排"),
    "vector+rw+bge": ("vector", "rule", "bge", "向量+改写+CrossEncoder 重排"),
    "hybrid+rw+bge": ("hybrid", "rule", "bge", "混合+改写+CrossEncoder 重排"),
}
DEFAULT_CONFIGS = ["bm25", "vector", "hybrid", "bm25+rw", "vector+rw", "hybrid+rw"]


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def load_dataset(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def run_config(
    retriever: Retriever,
    dataset: list[dict],
    kind: str,
    rewrite_mode: str,
    rerank_kind: str,
    ks: list[int],
    cache: dict | None = None,
    reranker_cache: dict | None = None,
) -> dict:
    top_k = max(ks)
    reranker = None
    if rerank_kind != "off":
        reranker_cache = reranker_cache if reranker_cache is not None else {}
        if rerank_kind not in reranker_cache:
            reranker_cache[rerank_kind] = build_reranker(rerank_kind)
        reranker = reranker_cache[rerank_kind]

    rows: list[dict] = []
    for item in dataset:
        context = item.get("context")
        if rewrite_mode == "off":
            rewrite_result = None
            search_query = item["question"]
        else:
            model = load_llm_config().model if rewrite_mode in {"llm", "auto"} else "rule"
            rewrite_result = rewrite_with_cache(
                item["question"],
                context,
                mode=rewrite_mode,
                cache=cache,
                model=model,
            )
            search_query = rewrite_result.query
        debug = retriever.search_debug(
            item["question"],
            context,
            top_k,
            kind=kind,
            rerank=rerank_kind != "off",
            rewrite_mode=rewrite_mode,
            reranker=reranker,
        )
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "search_query": search_query,
                "rewrite": rewrite_result.to_dict() if rewrite_result else None,
                "rewritten": bool(rewrite_result and rewrite_result.rewritten),
                "relevant": item["relevant_chunk_ids"],
                "retrieved": [hit["chunk_id"] for hit in debug["reranked"]],
            }
        )
    return {"rows": rows, "metrics": evaluate(rows, ks)}


def rewrite_stats(rows: list[dict]) -> str:
    """改写相关的统计：改了几条、回落几条、平均耗时。"""
    with_rewrite = [row for row in rows if row.get("rewrite")]
    if not with_rewrite:
        return ""
    rewritten = sum(1 for row in with_rewrite if row["rewritten"])
    fallbacks = sum(
        1 for row in with_rewrite if "fallback" in str(row["rewrite"].get("strategy", ""))
    )
    latencies = [row["rewrite"]["latency_ms"] for row in with_rewrite if row["rewrite"]["latency_ms"]]
    average = round(sum(latencies) / len(latencies), 1) if latencies else 0
    return f"改写 {rewritten}/{len(with_rewrite)} 条，规则回落 {fallbacks} 条，平均耗时 {average} ms"


def print_table(results: list[tuple[str, dict]], ks: list[int]) -> None:
    header = _pad("配置", 22) + "".join(_pad(f"recall@{k}", 12) for k in ks) + _pad("MRR", 10)
    print(header)
    print("-" * _width(header))
    for name, payload in results:
        metrics = payload["metrics"]
        line = _pad(name, 22) + "".join(_pad(f"{metrics[f'recall@{k}']:.4f}", 12) for k in ks)
        line += _pad(f"{metrics['mrr']:.4f}", 10)
        print(line)


def print_misses(name: str, rows: list[dict], top_k: int) -> None:
    misses = [
        row
        for row in rows
        if not set(row["relevant"]) & set(row["retrieved"][:top_k])
    ]
    print(f"\n未命中用例（{name}，共 {len(misses)} 条）：")
    if not misses:
        print("  全部命中")
        return
    for row in misses:
        expected = ", ".join(row["relevant"])
        actual = ", ".join(row["retrieved"][:top_k]) or "（空）"
        shown_query = row["search_query"]
        if shown_query != row["question"]:
            shown_query = f"{row['question']} -> {shown_query}"
        print(f"  [{row['id']}] {shown_query}")
        print(f"      期望：{expected}")
        print(f"      实际：{actual}")


def write_report(
    path: Path,
    results: list[tuple[str, dict]],
    ks: list[int],
    dataset_path: Path,
    dataset: list[dict],
    retriever: Retriever,
    embedder_name: str,
    misses_config: tuple[str, dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 检索基线报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 评测集：{dataset_path.name}（{len(dataset)} 条）",
        f"- 分块策略：{retriever.strategy}（{len(retriever.chunks)} 个 chunk）",
        f"- 向量模型：{embedder_name}",
        f"- RRF k：{retriever.rrf_k}",
        "",
        "| 配置 | " + " | ".join(f"recall@{k}" for k in ks) + " | MRR |",
        "| --- | " + " | ".join("---" for _ in ks) + " | --- |",
    ]
    for name, payload in results:
        metrics = payload["metrics"]
        cells = " | ".join(f"{metrics[f'recall@{k}']:.4f}" for k in ks)
        lines.append(f"| {name} | {cells} | {metrics['mrr']:.4f} |")

    stats_lines = [
        f"- {name}：{rewrite_stats(payload['rows'])}"
        for name, payload in results
        if rewrite_stats(payload["rows"])
    ]
    if stats_lines:
        lines += ["", "## 改写统计", "", *stats_lines]

    name, payload = misses_config
    lines += ["", f"## 未命中用例（{name}）", ""]
    misses = [
        row for row in payload["rows"] if not set(row["relevant"]) & set(row["retrieved"])
    ]
    if not misses:
        lines.append("全部命中。")
    else:
        lines += ["| id | 提问 | 检索用查询 | 期望 chunk | 实际 top-k |", "| --- | --- | --- | --- | --- |"]
        for row in misses:
            lines.append(
                f"| {row['id']} | {row['question']} | {row['search_query']} | "
                f"{', '.join(row['relevant'])} | {', '.join(row['retrieved'][:5])} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入：{path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="检索评测与配置对比")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS, choices=list(CONFIG_TABLE))
    parser.add_argument("--ks", nargs="*", type=int, default=[1, 3, 5])
    parser.add_argument("--strategy", choices=["rule", "paragraph"], default="rule")
    parser.add_argument("--embedder", choices=["auto", "bge", "hash"], default="auto")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--tokenizer",
        choices=["bigram", "jieba"],
        default="bigram",
        help="BM25 分词方式，默认字符 bigram（确定性）",
    )
    parser.add_argument(
        "--weights",
        default="",
        help='按路加权，例如 "bm25=0.5,vector=2"，默认等权',
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="指标 JSON（自动回归读它）")
    parser.add_argument(
        "--rewrite-cache",
        type=Path,
        default=DEFAULT_REWRITE_CACHE,
        help="LLM 改写缓存文件，避免重复调用付费接口",
    )
    parser.add_argument("--no-cache", action="store_true", help="不使用也不写入改写缓存")
    parser.add_argument("--no-report", action="store_true", help="不写报告文件")
    args = parser.parse_args()

    weights: dict[str, float] = {}
    if args.weights:
        for pair in args.weights.split(","):
            if "=" not in pair:
                raise SystemExit(f"权重格式应为名字=数值，收到：{pair}")
            name, value = pair.split("=", 1)
            weights[name.strip()] = float(value)

    dataset = load_dataset(args.dataset)
    set_tokenizer(args.tokenizer)
    embedder = build_embedder(args.embedder)
    retriever = Retriever(
        embedder=embedder,
        strategy=args.strategy,
        rrf_k=args.rrf_k,
        weights=weights or None,
    )

    print(f"评测集：{args.dataset.name}（{len(dataset)} 条）")
    print(f"知识库：{len(retriever.chunks)} 个 chunk（策略 {args.strategy}）")
    print(f"向量模型：{embedder.name}（维度 {embedder.dim}），RRF k={args.rrf_k}\n")
    print(f"BM25 分词：{args.tokenizer}\n")
    if weights:
        print(f"融合权重：{weights}\n")

    uses_llm = any(CONFIG_TABLE[key][1] in {"llm", "auto"} for key in args.configs)
    if uses_llm:
        llm_config = load_llm_config()
        if llm_config.configured:
            print(f"LLM 改写：{llm_config.model} @ {llm_config.base_url}\n")
        else:
            print("[warn] LLM 改写未启用或缺少 API key（见 agent/.env），本次回落规则版\n")

    cache: dict = {}
    if uses_llm and not args.no_cache and args.rewrite_cache.exists():
        cache = json.loads(args.rewrite_cache.read_text(encoding="utf-8"))
        print(f"改写缓存：载入 {len(cache)} 条（{args.rewrite_cache}）\n")

    results: list[tuple[str, dict]] = []
    reranker_cache: dict = {}
    for key in args.configs:
        kind, rewrite_mode, rerank_kind, label = CONFIG_TABLE[key]
        payload = run_config(
            retriever,
            dataset,
            kind,
            rewrite_mode,
            rerank_kind,
            args.ks,
            cache,
            reranker_cache,
        )
        results.append((label, payload))
        stats = rewrite_stats(payload["rows"])
        if stats:
            print(f"[{label}] {stats}")

    print()
    print_table(results, args.ks)

    if uses_llm and not args.no_cache:
        args.rewrite_cache.parent.mkdir(parents=True, exist_ok=True)
        args.rewrite_cache.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n改写缓存已更新：{args.rewrite_cache}（{len(cache)} 条）")

    best_name, best_payload = max(
        results,
        key=lambda item: (item[1]["metrics"].get(f"recall@{max(args.ks)}", 0), item[1]["metrics"]["mrr"]),
    )
    print_misses(best_name, best_payload["rows"], max(args.ks))

    if not args.no_report:
        write_report(
            args.report,
            results,
            args.ks,
            args.dataset,
            dataset,
            retriever,
            embedder.name,
            (best_name, best_payload),
        )

    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset.name,
        "dataset_size": len(dataset),
        "strategy": args.strategy,
        "embedder": embedder.name,
        "tokenizer": args.tokenizer,
        "ks": list(args.ks),
        "best_config": best_name,
        "configs": [
            {
                "config": key,
                "label": CONFIG_TABLE[key][3],
                "metrics": item[1]["metrics"],
                # 未命中清单也写进 JSON：reranker 迭代要拿它当靶子，
                # 从 markdown 里刮表格太脆
                "misses": [
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "relevant": list(row["relevant"]),
                        "retrieved": list(row["retrieved"][: max(args.ks)]),
                    }
                    for row in item[1]["rows"]
                    if not set(row["relevant"]) & set(row["retrieved"])
                ],
            }
            for key, item in zip(args.configs, results)
        ],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n指标 JSON：{args.json}")


if __name__ == "__main__":
    main()
