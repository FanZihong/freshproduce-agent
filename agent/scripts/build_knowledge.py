"""构建知识库索引：分块 + 落盘 data/knowledge_index.json。

用法（在 agent/ 目录下执行）：
    python scripts/build_knowledge.py
    python scripts/build_knowledge.py --sample 8
    python scripts/build_knowledge.py --max-chars 200 --overlap 30
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DATA_DIR  # noqa: E402
from agent.rag.chunker import (  # noqa: E402
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP,
    build_chunks,
    chunk_stats,
    load_documents,
)

KNOWLEDGE_DIR = DATA_DIR / "knowledge"
INDEX_PATH = DATA_DIR / "knowledge_index.json"
STRATEGIES = ("rule", "paragraph")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建知识库索引")
    parser.add_argument("--sample", type=int, default=5, help="抽样预览条数")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="段落策略的窗口大小")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="段落策略的窗口重叠")
    args = parser.parse_args()

    documents = load_documents(KNOWLEDGE_DIR)
    if not documents:
        print(f"没有在 {KNOWLEDGE_DIR} 找到知识库文档")
        return

    print("文档清单：")
    for doc in documents:
        print(f"  - {doc.title}（{doc.doc_id}，{len(doc.sections)} 个小节）")
        print(f"      来源：{doc.source or '未标注'}")

    index = {
        "version": 1,
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "knowledge_dir": str(KNOWLEDGE_DIR),
        "document_count": len(documents),
        "strategies": {},
    }

    chunks_by_strategy = {}
    for strategy in STRATEGIES:
        chunks = build_chunks(
            KNOWLEDGE_DIR,
            strategy=strategy,
            max_chars=args.max_chars,
            overlap=args.overlap,
        )
        chunks_by_strategy[strategy] = chunks
        stats = chunk_stats(chunks)
        index["strategies"][strategy] = {
            "stats": stats,
            "chunks": [chunk.as_dict() for chunk in chunks],
        }
        print(
            f"\n策略 {strategy}：共 {stats['chunk_count']} 个 chunk，"
            f"平均 {stats['avg_chars']} 字，最长 {stats['max_chars']} 字，最短 {stats['min_chars']} 字"
        )
        per_doc: dict[str, int] = {}
        for chunk in chunks:
            per_doc[chunk.doc_id] = per_doc.get(chunk.doc_id, 0) + 1
        for doc_id, count in per_doc.items():
            print(f"    {doc_id}: {count}")

    INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n索引已写入：{INDEX_PATH}")

    sample_chunks = chunks_by_strategy["rule"][: max(0, args.sample)]
    if sample_chunks:
        print(f"\n抽样预览（rule 策略，前 {len(sample_chunks)} 条）：")
        for order, chunk in enumerate(sample_chunks, start=1):
            preview = chunk.text if len(chunk.text) <= 60 else chunk.text[:60] + "..."
            print(f"  {order}. [{chunk.chunk_id}] {chunk.section} | {preview}")


if __name__ == "__main__":
    main()
