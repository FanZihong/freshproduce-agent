"""读取 chunk：优先用 build_knowledge.py 生成的索引，没有就现场构建。"""

from __future__ import annotations

import json
from pathlib import Path

from agent.db import DATA_DIR
from agent.rag.chunker import Chunk, build_chunks

KNOWLEDGE_DIR = DATA_DIR / "knowledge"
INDEX_PATH = DATA_DIR / "knowledge_index.json"


def load_chunks(
    strategy: str = "rule",
    *,
    index_path: Path = INDEX_PATH,
    knowledge_dir: Path = KNOWLEDGE_DIR,
) -> list[Chunk]:
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        strategy_payload = payload.get("strategies", {}).get(strategy)
        if strategy_payload:
            return [Chunk(**item) for item in strategy_payload["chunks"]]
    return build_chunks(knowledge_dir, strategy=strategy)
