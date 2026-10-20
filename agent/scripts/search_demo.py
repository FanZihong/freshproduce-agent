"""命令行观察 BM25 与向量两路检索的差异。

用法（在 agent/ 目录下执行）：
    python scripts/search_demo.py "坏果怎么赔"
    python scripts/search_demo.py "车厘子怎么保存" -k 3 --strategy paragraph
    python scripts/search_demo.py "坏果怎么赔" --embedder hash     # 不下载模型
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.rag.embedder import build_embedder  # noqa: E402
from agent.rag.hybrid_search import HybridSearcher, SearchHit  # noqa: E402
from agent.rag.store import load_chunks  # noqa: E402


def print_hits(title: str, hits: list[SearchHit]) -> None:
    print(f"\n{title}")
    if not hits:
        print("  （无结果）")
        return
    print(f"  {'排名':<4}{'分数':<10}{'chunk_id':<32}内容")
    for hit in hits:
        preview = hit.text if len(hit.text) <= 34 else hit.text[:34] + "..."
        print(f"  {hit.rank:<6}{hit.score:<10}{hit.chunk_id:<32}{preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 与向量检索对比")
    parser.add_argument("query", nargs="?", default="坏果怎么赔", help="查询语句")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="每路返回条数")
    parser.add_argument("--strategy", choices=["rule", "paragraph"], default="rule", help="分块策略")
    parser.add_argument("--embedder", choices=["auto", "bge", "hash"], default="auto", help="向量化方式")
    parser.add_argument("--model", default=None, help="覆盖默认向量模型名")
    args = parser.parse_args()

    chunks = load_chunks(args.strategy)
    print(f"知识库：{len(chunks)} 个 chunk（策略 {args.strategy}）")

    embedder_kwargs = {"kind": args.embedder}
    embedder = (
        build_embedder(model_name=args.model, **embedder_kwargs)
        if args.model
        else build_embedder(**embedder_kwargs)
    )
    searcher = HybridSearcher(chunks, embedder)
    print(f"向量模型：{embedder.name}（维度 {embedder.dim}）")
    print(f"\n查询：{args.query}")

    result = searcher.compare(args.query, args.top_k)
    print_hits(f"BM25 top{args.top_k}", result["bm25"])
    print_hits(f"向量 top{args.top_k}", result["vector"])

    print("\n两路对比：")
    print(f"  重合 {len(result['overlap'])} 个：{result['overlap']}")
    print(f"  仅 BM25 命中 {len(result['only_bm25'])} 个：{result['only_bm25']}")
    print(f"  仅向量命中 {len(result['only_vector'])} 个：{result['only_vector']}")
    print("\n重合度低说明两路互补，这正是下一步用 RRF 融合的前提。")


if __name__ == "__main__":
    main()
