"""增量写入管线：先召回旧事实 → 抽增量 → 两层去重 → 三处落盘。

这才是 Mem0 范式里最值钱的流程，顺序不能反：
1. **召回**：取该会话已有事实（本段用"按会话取最近 N 条"，第二段会升级成混合检索）；
2. **抽增量**：把旧事实和对话一起给模型，只让它输出新增/更正的部分；
3. **两层去重**：内容哈希挡完全重复，嵌入相似度（默认 0.92）挡换说法重复，
   命中相似度就把旧事实"并"成新说法（update），而不是再插一条；
4. **落盘**：SQLite 存事实、`memory_index.json` 存向量、BM25 由事实原文现算。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from agent.config import get_env, get_float
from agent.db import DATA_DIR
from agent.memory import facts as facts_store
from agent.memory.extractor import extract_facts
from agent.memory.index import MemoryIndex

DEFAULT_INDEX_PATH = DATA_DIR / "memory_index.json"
DEFAULT_SIMILARITY = 0.92
VALID_MODES = ("off", "rule", "llm")

_EMBEDDER = None


def default_index_path() -> Path:
    """索引文件路径；`MEMORY_INDEX_PATH` 可覆盖（消融/演示用临时索引）。"""
    configured = get_env("MEMORY_INDEX_PATH", "")
    return Path(configured) if configured else DEFAULT_INDEX_PATH


def _default_embedder():
    """进程内复用嵌入模型：第一次压缩时加载，之后每次写记忆不再重新加载。"""
    global _EMBEDDER
    if _EMBEDDER is None:
        from agent.rag.embedder import build_embedder

        _EMBEDDER = build_embedder()
    return _EMBEDDER


@dataclass
class RememberStats:
    """一次写入的结果：各类计数 + 调用成本，便于演示与消融。"""

    mode: str = "off"
    candidates: int = 0
    added: int = 0
    updated: int = 0
    skipped: int = 0
    calls: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _resolve_mode(mode: str | None) -> str:
    value = (mode or get_env("MEMORY_FACTS", "off") or "off").strip().lower()
    if value not in VALID_MODES:
        print(f"[warn] MEMORY_FACTS 只能是 off/rule/llm，收到 {value!r}，按 off 处理")
        return "off"
    return value


def _encode(embedder, text: str) -> np.ndarray:
    vector = np.asarray(embedder.encode([text], is_query=False)[0], dtype="float32")
    norm = float(np.linalg.norm(vector)) or 1.0
    return vector / norm


def _best_match(ids: list[str], matrix: np.ndarray, vector: np.ndarray) -> tuple[str, float]:
    if not ids or matrix.size == 0:
        return "", 0.0
    scores = matrix @ vector
    index = int(np.argmax(scores))
    return ids[index], float(scores[index])


def remember(
    session_id: str,
    turns: list[dict],
    *,
    summary: str = "",
    mode: str | None = None,
    db_path: Path | str | None = None,
    index_path: Path | str | None = None,
    embedder=None,
    similarity_threshold: float | None = None,
) -> RememberStats:
    """跑一次增量写入；返回计数与成本。`mode=off` 时什么都不做（默认）。"""
    resolved = _resolve_mode(mode)
    stats = RememberStats(mode=resolved)
    if resolved == "off":
        return stats
    if not session_id:
        stats.error = "缺少 session_id"
        return stats
    if not turns:
        stats.error = "没有对话内容"
        return stats

    # 1) 召回旧事实（按会话取；过期的不算——它们既不该挡住新事实，也不该被"更新"）
    old_facts = facts_store.list_facts(
        session_id, limit=200, include_expired=False, db_path=db_path
    )
    known_ids = {item["id"] for item in old_facts}

    # 2) 只抽增量
    candidates, meta = extract_facts(
        turns,
        old_facts,
        mode="llm" if resolved == "llm" else "rule",
        summary=summary,
    )
    stats.candidates = len(candidates)
    stats.calls = int(meta.get("calls") or 0)
    stats.cost = float(meta.get("cost") or 0.0)
    stats.latency_ms = float(meta.get("latency_ms") or 0.0)

    if not candidates:
        return stats

    if embedder is None:
        embedder = _default_embedder()
    threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else get_float("MEMORY_DEDUPE_SIM", DEFAULT_SIMILARITY)
    )
    index = MemoryIndex(index_path or default_index_path()).load()

    # 旧事实的向量补齐（索引里可能没有：比如事实是别的进程写的）
    for item in old_facts:
        record = index.records.get(item["id"])
        if record and record.get("vector"):
            continue
        index.upsert(
            item["id"],
            fact=item["fact"],
            session_id=session_id,
            entities=item.get("entities") or [],
            vector=_encode(embedder, item["fact"]),
        )

    known_hashes = {item["content_hash"] for item in old_facts}
    for candidate in candidates:
        text = str(candidate.get("fact") or "").strip()
        if not text:
            stats.skipped += 1
            continue
        content_hash = facts_store.fact_hash(text)
        if content_hash in known_hashes:
            # 第一层去重：一字不差（可能是模型没听话，也可能是重复提交）
            stats.skipped += 1
            continue

        vector = _encode(embedder, text)
        ids, matrix = index.vectors_for(session_id)
        match_id, similarity = _best_match(ids, matrix, vector)
        # 索引里可能残留已过期事实的向量（还没跑到 purge），不能拿它当更新目标
        if match_id and similarity >= threshold and match_id in known_ids:
            # 第二层去重：换说法命中相似度 -> 把旧事实并成新说法
            facts_store.update_fact(
                match_id,
                fact=text,
                category=candidate.get("category") or "other",
                entities=candidate.get("entities") or None,
                db_path=db_path,
            )
            index.upsert(
                match_id,
                fact=text,
                session_id=session_id,
                entities=candidate.get("entities") or [],
                vector=vector,
            )
            known_hashes.add(content_hash)
            stats.updated += 1
            continue

        target_id = str(candidate.get("target_id") or "")
        if candidate.get("action") == "update" and target_id:
            updated = facts_store.update_fact(
                target_id,
                fact=text,
                category=candidate.get("category") or "other",
                entities=candidate.get("entities") or None,
                db_path=db_path,
            )
            if updated is not None:
                index.upsert(
                    target_id,
                    fact=text,
                    session_id=session_id,
                    entities=candidate.get("entities") or [],
                    vector=vector,
                )
                known_ids.add(target_id)
                known_hashes.add(content_hash)
                stats.updated += 1
                continue

        if candidate.get("action") == "skip":
            stats.skipped += 1
            continue

        record, created = facts_store.insert_fact(
            session_id,
            text,
            category=candidate.get("category") or "other",
            entities=candidate.get("entities") or [],
            source=resolved,
            db_path=db_path,
        )
        if not created:
            stats.skipped += 1
            continue
        index.upsert(
            record["id"],
            fact=text,
            session_id=session_id,
            entities=candidate.get("entities") or [],
            vector=vector,
        )
        known_ids.add(record["id"])
        known_hashes.add(content_hash)
        stats.added += 1

    index.save()
    return stats
