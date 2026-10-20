"""事实老化与过期清理的测试：衰减曲线、排序、读时过滤、批量清理、召回续期。"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pytest

from agent.memory import facts as facts_store
from agent.memory.index import MemoryIndex
from agent.memory.search import search_facts
from agent.db import ensure_fact_schema

OLD = "草莓坏果一半已赔付"
NEW = "草莓坏果一半待赔付"
QUERY = "草莓坏果赔付"
SESSION = "s-decay"


class EqualEmbedder:
    """把 OLD/NEW/查询映射到同一方向：相关性打平，差异只来自老化。"""

    name = "fake"

    def __init__(self) -> None:
        self.mapping = {OLD: [1, 0, 0, 0], NEW: [1, 0, 0, 0], QUERY: [1, 0, 0, 0]}

    def encode(self, texts, *, is_query: bool = False):
        return np.asarray(
            [self.mapping.get(str(text), [0, 1, 0, 0]) for text in texts],
            dtype="float32",
        )


def seed(db, ages: dict[str, int]) -> dict[str, str]:
    """插入事实并把 last_accessed_at 往前拨，模拟"很久没用"。"""
    now = datetime.now()
    ids: dict[str, str] = {}
    for text, days in ages.items():
        record, _ = facts_store.insert_fact(SESSION, text, db_path=db)
        facts_store.touch_facts([record["id"]], now=now - timedelta(days=days), db_path=db)
        ids[text] = record["id"]
    return ids


def test_decay_curve_half_life_and_floor(monkeypatch):
    monkeypatch.setenv("MEMORY_DECAY", "on")
    item = {"last_accessed_at": "2026-01-01 00:00:00"}
    kwargs = {"half_life_days": 14, "floor": 0.5}
    assert facts_store.decay_factor(item, now="2026-01-01 00:00:00", **kwargs) == 1.0
    assert facts_store.decay_factor(item, now="2026-01-15 00:00:00", **kwargs) == 0.75
    # 很久以后逼近 floor，但不会低于它
    assert facts_store.decay_factor(item, now="2027-01-01 00:00:00", **kwargs) == pytest.approx(
        0.5, abs=0.01
    )
    # 显式关闭时恒为 1
    assert facts_store.decay_factor(item, now="2027-01-01 00:00:00", enabled=False, **kwargs) == 1.0


def test_decay_uses_last_access_then_updated_at():
    fresh = {"last_accessed_at": None, "updated_at": "2026-01-01 00:00:00",
             "created_at": "2026-01-01 00:00:00"}
    assert facts_store.fact_age_days(fresh, now="2026-01-11 00:00:00") == 10.0
    touched = {**fresh, "last_accessed_at": "2026-01-10 00:00:00"}
    assert facts_store.fact_age_days(touched, now="2026-01-11 00:00:00") == 1.0


def test_decay_makes_newer_fact_rank_first(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DECAY", "on")
    monkeypatch.setenv("MEMORY_DECAY_HALF_LIFE_DAYS", "14")
    monkeypatch.setenv("MEMORY_DECAY_FLOOR", "0.5")
    monkeypatch.setenv("MEMORY_FACT_TTL_DAYS", "30")
    db = tmp_path / "facts.db"
    index = tmp_path / "index.json"
    ids = seed(db, {OLD: 25, NEW: 1})

    common = dict(db_path=db, index_path=index, embedder=EqualEmbedder(), top_k=5, touch=False)
    without = search_facts(QUERY, SESSION, decay=False, **common)
    with_decay = search_facts(QUERY, SESSION, decay=True, **common)

    # 相关性几乎打平：不降权时两条的分数差很小
    base = {row["id"]: row["base_score"] for row in without}
    assert abs(base[ids[OLD]] - base[ids[NEW]]) < 0.002
    # 25 天 vs 1 天，衰减后新事实必须排第一
    assert [row["id"] for row in with_decay][0] == ids[NEW]
    old_row = next(row for row in with_decay if row["id"] == ids[OLD])
    assert old_row["decay"] < 0.7
    assert old_row["age_days"] == 25.0


def test_expired_fact_is_hidden_from_search_and_list(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_FACT_TTL_DAYS", "30")
    db = tmp_path / "facts.db"
    ids = seed(db, {OLD: 40, NEW: 1})

    results = search_facts(
        QUERY, SESSION, db_path=db, index_path=tmp_path / "index.json",
        embedder=EqualEmbedder(), top_k=5, touch=False,
    )
    assert ids[OLD] not in [row["id"] for row in results]
    visible = facts_store.list_facts(SESSION, include_expired=False, db_path=db)
    assert [item["id"] for item in visible] == [ids[NEW]]
    everything = facts_store.list_facts(SESSION, include_expired=True, db_path=db)
    assert {item["id"] for item in everything} == set(ids.values())


def test_purge_removes_rows_and_vectors(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_FACT_TTL_DAYS", "30")
    db = tmp_path / "facts.db"
    index_path = tmp_path / "index.json"
    ids = seed(db, {OLD: 40, NEW: 1})

    # 过期前被检索过：索引里留着它的向量
    index = MemoryIndex(index_path)
    index.upsert(ids[OLD], fact=OLD, session_id=SESSION, vector=np.array([1, 0, 0, 0]))
    index.upsert(ids[NEW], fact=NEW, session_id=SESSION, vector=np.array([1, 0, 0, 0]))
    index.save()

    assert facts_store.list_expired_ids(db_path=db) == [ids[OLD]]
    removed = facts_store.purge_expired_facts(db_path=db, index_path=index_path)
    assert removed == [ids[OLD]]
    assert facts_store.get_fact(ids[OLD], db_path=db) is None
    assert facts_store.get_fact(ids[NEW], db_path=db) is not None
    assert ids[OLD] not in MemoryIndex(index_path).load().records
    assert facts_store.list_expired_ids(db_path=db) == []


def test_recall_touches_fact_and_extends_life(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_FACT_TTL_DAYS", "30")
    monkeypatch.setenv("MEMORY_DECAY", "on")
    db = tmp_path / "facts.db"
    ids = seed(db, {OLD: 29})

    results = search_facts(
        QUERY, SESSION, db_path=db, index_path=tmp_path / "index.json",
        embedder=EqualEmbedder(), top_k=5,
    )
    assert [row["id"] for row in results] == [ids[OLD]]

    refreshed = facts_store.get_fact(ids[OLD], db_path=db)
    assert refreshed["last_accessed_at"]
    assert facts_store.fact_age_days(refreshed) < 0.01
    assert facts_store.is_expired(refreshed) is False


def test_ttl_zero_disables_expiry(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_FACT_TTL_DAYS", "0")
    db = tmp_path / "facts.db"
    ids = seed(db, {OLD: 365})
    assert facts_store.list_expired_ids(db_path=db) == []
    assert facts_store.purge_expired_facts(db_path=db) == []
    visible = facts_store.list_facts(SESSION, include_expired=False, db_path=db)
    assert [item["id"] for item in visible] == [ids[OLD]]


def test_schema_migration_adds_last_accessed_at(tmp_path):
    """老库的 memory_facts 没有 last_accessed_at 时，ensure_fact_schema 要补列。"""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE memory_facts (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, fact TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            entities TEXT NOT NULL DEFAULT '[]',
            importance REAL NOT NULL DEFAULT 0.5, content_hash TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'llm',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    ensure_fact_schema(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memory_facts)")}
    conn.close()
    assert "last_accessed_at" in columns

    record, created = facts_store.insert_fact(SESSION, OLD, db_path=db)
    assert created is True
    facts_store.touch_fact(record["id"], db_path=db)     # 补列后这条路径不再报错
    assert facts_store.get_fact(record["id"], db_path=db)["last_accessed_at"]


def test_describe_fact_reports_aging_fields(monkeypatch):
    monkeypatch.setenv("MEMORY_FACT_TTL_DAYS", "30")
    monkeypatch.setenv("MEMORY_DECAY", "on")
    monkeypatch.setenv("MEMORY_DECAY_HALF_LIFE_DAYS", "14")
    monkeypatch.setenv("MEMORY_DECAY_FLOOR", "0.5")
    now = datetime(2026, 2, 1, 12, 0, 0)
    item = {"last_accessed_at": "2026-01-12 12:00:00", "updated_at": "2026-01-12 12:00:00"}
    status = facts_store.describe_fact(item, now=now)
    assert status["age_days"] == 20.0
    assert 0.5 < status["decay"] < 0.8
    assert status["expires_at"] == "2026-02-11 12:00:00"
    assert status["expired"] is False
