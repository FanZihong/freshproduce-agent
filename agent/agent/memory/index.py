"""事实记忆的索引层：向量与文本的落盘表示（`data/memory_index.json`）。

三个存储各管一段（这也是 Mem0 的分工思路）：
- **SQLite**：事实原文与元数据（可查询、可事务更新）；
- **向量索引**（本文件）：事实嵌入，用于"换说法也能召回"；
- **BM25 索引**：由事实原文现算（量小时没必要落盘，量大再加倒排文件）。

为什么不用 FAISS/Chroma：实事量在千级以内，numpy 点积足够快，JSON 落盘零新依赖，
和 `knowledge_index.json` 一个套路；真要上生产再换向量库，接口不变。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

INDEX_VERSION = 1


class MemoryIndex:
    """内存里的索引 + 落盘文件；按 fact_id 去重。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        self.updated_at = ""

    def load(self) -> "MemoryIndex":
        if not self.path.exists():
            return self
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self
        if payload.get("version") != INDEX_VERSION:
            return self
        self.updated_at = payload.get("updated_at") or ""
        for item in payload.get("facts") or []:
            if item.get("id"):
                self.records[str(item["id"])] = item
        return self

    def upsert(
        self,
        fact_id: str,
        *,
        fact: str,
        session_id: str,
        entities: list[str] | None = None,
        vector: np.ndarray | None = None,
    ) -> None:
        values = vector if vector is not None else []
        self.records[fact_id] = {
            "id": fact_id,
            "session_id": session_id,
            "fact": fact,
            "entities": list(entities or []),
            # 注意不能写 `vector or []`：numpy 数组的真值判断会抛 Ambiguous
            "vector": [round(float(value), 6) for value in values],
        }
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def drop(self, fact_id: str) -> None:
        self.records.pop(fact_id, None)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": INDEX_VERSION,
            "updated_at": self.updated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "facts": list(self.records.values()),
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def vectors_for(self, session_id: str) -> tuple[list[str], np.ndarray]:
        """取某会话所有带向量的事实：返回 (ids, 归一化矩阵)。"""
        ids: list[str] = []
        rows: list[list[float]] = []
        for item in self.records.values():
            if item.get("session_id") != session_id or not item.get("vector"):
                continue
            ids.append(str(item["id"]))
            rows.append(item["vector"])
        if not rows:
            return [], np.zeros((0, 0), dtype="float32")
        matrix = np.asarray(rows, dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return ids, matrix / norms

    def __len__(self) -> int:
        return len(self.records)
