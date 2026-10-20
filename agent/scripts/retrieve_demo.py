"""演示：多轮追问 -> 查询改写 -> 两路召回 -> RRF 融合。

用法（在 agent/ 目录下执行）：
    python scripts/retrieve_demo.py
    python scripts/retrieve_demo.py --query "那它呢" --last-question "草莓怎么保存" --product 草莓
    python scripts/retrieve_demo.py --embedder hash          # 不下载模型
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.rag.embedder import build_embedder  # noqa: E402
from agent.rag.reranker import build_reranker  # noqa: E402
from agent.rag.retriever import Retriever  # noqa: E402

DEFAULT_CASES = [
    {
        "title": "多轮指代：那它呢",
        "last_question": "草莓怎么保存",
        "product": "草莓",
        "query": "那它呢",
    },
    {
        "title": "指代 + 政策问题：这个能退吗",
        "last_question": "车厘子怎么保存",
        "product": "车厘子",
        "query": "这个能退吗",
    },
    {
        "title": "自包含问题（不需要改写）",
        "last_question": "草莓怎么保存",
        "product": "",
        "query": "坏果比例 50% 赔多少",
    },
]


def print_case(retriever: Retriever, case: dict, top_k: int, rerank_kind: str, reranker) -> None:
    context = {"last_question": case["last_question"], "product": case["product"]}
    pool_size = max(retriever.rerank_candidates, top_k)
    before = retriever.search_debug(case["query"], context, pool_size, rerank=False)
    debug = retriever.search_debug(
        case["query"],
        context,
        top_k,
        rerank=rerank_kind != "off",
        reranker=reranker,
    )
    rewrite = debug["rewrite"]

    print(f"\n=== {case['title']} ===")
    print(f"  上一轮问题：{case['last_question'] or '（无）'}")
    print(f"  当前提问  ：{case['query']}")
    print(f"  改写结果  ：{rewrite['query']}")
    print(
        f"  改写说明  ：{rewrite['reason']}"
        f"（策略 {rewrite['strategy']}，rewritten={rewrite['rewritten']}，耗时 {rewrite['latency_ms']} ms）"
    )

    print(f"\n  重排前：融合 top{pool_size}（'-' 表示未进该路 top{pool_size}）")
    print(f"    {'排名':<8}{'融合分':<10}{'BM25':<8}{'向量':<8}{'chunk_id':<30}内容")
    for index, hit in enumerate(before["fused"], start=1):
        bm25_rank = hit["ranks"].get("bm25", "-")
        vector_rank = hit["ranks"].get("vector", "-")
        preview = hit["text"] if len(hit["text"]) <= 30 else hit["text"][:30] + "..."
        print(
            f"    {index:<8}{hit['fused_score']:<10}{str(bm25_rank):<8}{str(vector_rank):<8}"
            f"{hit['chunk_id']:<30}{preview}"
        )

    print(f"\n  重排后 top{top_k}（重排方式：{debug['reranker']}）")
    print(f"    {'排名':<8}{'重排分':<12}{'变化':<8}{'原排名':<8}{'chunk_id':<30}内容")
    for index, hit in enumerate(debug["reranked"], start=1):
        movement = hit["movement"]
        arrow = "—" if movement == 0 else (f"↑{movement}" if movement > 0 else f"↓{-movement}")
        preview = hit["text"] if len(hit["text"]) <= 30 else hit["text"][:30] + "..."
        print(
            f"    {index:<8}{hit['rerank_score']:<12}{arrow:<8}{hit['before_rank']:<8}"
            f"{hit['chunk_id']:<30}{preview}"
        )

    both = sum(
        1 for hit in debug["reranked"] if "bm25" in hit["ranks"] and "vector" in hit["ranks"]
    )
    print(
        f"\n  最终 top{top_k} 中：两路都命中 {both} 条，"
        f"只有一路命中 {len(debug['reranked']) - both} 条"
    )
    print(f"  候选池：每路取 {debug['candidate_k']} 条后融合，RRF k={debug['rrf_k']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="查询改写 + RRF 融合演示")
    parser.add_argument("--query", default=None, help="只跑单个自定义查询")
    parser.add_argument("--last-question", default="", help="上一轮用户问题")
    parser.add_argument("--product", default="", help="上文识别到的商品")
    parser.add_argument("-k", "--top-k", type=int, default=3, help="最终返回条数")
    parser.add_argument("--strategy", choices=["rule", "paragraph"], default="rule")
    parser.add_argument("--embedder", choices=["auto", "bge", "hash"], default="auto")
    parser.add_argument("--model", default=None, help="覆盖默认向量模型名")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF 的 k 值")
    parser.add_argument(
        "--rewrite-mode",
        choices=["rule", "auto", "llm"],
        default="rule",
        help="改写方式：rule 规则版，llm 走大模型，auto 按 .env 配置决定",
    )
    parser.add_argument(
        "--rerank",
        choices=["off", "score", "bge"],
        default="score",
        help="重排方式：off 不重排，score 融合分数基线，bge 用 CrossEncoder 精排",
    )
    args = parser.parse_args()

    embedder = (
        build_embedder(model_name=args.model, kind=args.embedder)
        if args.model
        else build_embedder(args.embedder)
    )
    retriever = Retriever(
        embedder=embedder,
        strategy=args.strategy,
        rrf_k=args.rrf_k,
        rewrite_mode=args.rewrite_mode,
    )
    print(f"知识库：{len(retriever.chunks)} 个 chunk（策略 {args.strategy}）")
    print(f"向量模型：{embedder.name}（维度 {embedder.dim}），RRF k={args.rrf_k}")
    print(f"改写方式：{args.rewrite_mode}")
    reranker = build_reranker(args.rerank) if args.rerank != "off" else None
    print(f"重排方式：{reranker.name if reranker else 'off'}")

    if args.query:
        cases = [
            {
                "title": "自定义用例",
                "last_question": args.last_question,
                "product": args.product,
                "query": args.query,
            }
        ]
    else:
        cases = DEFAULT_CASES

    for case in cases:
        print_case(retriever, case, args.top_k, args.rerank, reranker)


if __name__ == "__main__":
    main()
