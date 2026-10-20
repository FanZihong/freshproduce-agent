"""事实记忆的事实库（SQLite）：一行一条"抽取出来的事实"。

为什么事实放 SQLite 而不是跟着 `MEMORY_STORE` 走 Redis：
事实需要**按会话查询、按哈希查重、按 id 更新、事务提交**——这些是关系库的活；
Redis 那边继续只放短期状态（会话活跃、摘要 KV）。Mem0 也是这么分工的
（history/DB + vector store 各管一段）。

去重有两层，这里是第一层：
- `normalize_fact` + `fact_hash`：把事实归一（空白、标点）后取 sha256，
  完全重复的事实进不来，DB 上再加 `(session_id, content_hash)` 唯一索引兜并发；
- 第二层（嵌入相似度）在 writer 里做，因为要算向量。
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from agent.config import get_env, get_float
from agent.db import DB_PATH, connect, ensure_fact_schema
from agent.memory.store import ttl_cutoff

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# 事实的默认寿命：30 天没被召回就清理。摘要只有 7 天，事实更长——
# 一次理赔跨月跟进是常态，而"再也没被用到"才是真正该忘的信号。
DEFAULT_TTL_DAYS = 30.0
# 老化降权：半衰期 14 天，最低保留 50% 权重。
# 半衰期太短会让"上个月买过草莓"这类仍然有用的记忆迅速沉底；
# 留一个 floor 是为了"很久没用但高度相关"的事实仍能靠相关性赢回来。
DEFAULT_HALF_LIFE_DAYS = 14.0
DEFAULT_DECAY_FLOOR = 0.5

_PUNCT = re.compile(r"[\s，。；！？、,.!?;:：\"'“”‘’（）()\[\]{}]+")


def normalize_fact(text: str) -> str:
    """事实归一：去首尾空白、合并空白、去掉标点，便于判重。"""
    return _PUNCT.sub("", str(text or "").strip()).lower()


def fact_hash(text: str) -> str:
    """事实内容哈希（归一后 sha256 前 32 位）。"""
    return hashlib.sha256(normalize_fact(text).encode("utf-8")).hexdigest()[:32]


def _row_to_dict(row: sqlite3.Row) -> dict:
    payload = dict(row)
    try:
        payload["entities"] = json.loads(payload.get("entities") or "[]")
    except json.JSONDecodeError:
        payload["entities"] = []
    return payload


def _parse_time(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, TIME_FORMAT)
    except ValueError:
        return None


def _now(now: datetime | str | None = None) -> datetime:
    if isinstance(now, datetime):
        return now
    parsed = _parse_time(str(now)) if now else None
    return parsed or datetime.now()


def last_seen_at(item: dict) -> datetime:
    """一条事实"最后一次有用"的时间：最近一次召回 > 最近一次更新 > 创建。"""
    for key in ("last_accessed_at", "updated_at", "created_at"):
        parsed = _parse_time(item.get(key))
        if parsed is not None:
            return parsed
    return datetime.now()


def fact_age_days(item: dict, *, now: datetime | str | None = None) -> float:
    """距离最后一次被用到过去了多少天（负数按 0 处理，防时钟回拨）。"""
    delta = _now(now) - last_seen_at(item)
    return round(max(0.0, delta.total_seconds() / 86400), 3)


def resolve_ttl_days(ttl_days: float | None = None) -> float:
    """过期天数：显式参数 > `MEMORY_FACT_TTL_DAYS` > 30；<=0 表示不过期。"""
    if ttl_days is not None:
        return float(ttl_days)
    raw = get_env("MEMORY_FACT_TTL_DAYS", "") or str(DEFAULT_TTL_DAYS)
    try:
        value = float(raw)
    except ValueError:
        value = DEFAULT_TTL_DAYS
    return value if value > 0 else 0.0


def fact_expires_at(
    item: dict,
    *,
    ttl_days: float | None = None,
) -> str:
    """按"最近一次使用 + TTL"算出过期时间；TTL<=0 时返回空串（永不过期）。"""
    days = resolve_ttl_days(ttl_days)
    if days <= 0:
        return ""
    return (last_seen_at(item) + timedelta(days=days)).strftime(TIME_FORMAT)


def is_expired(
    item: dict,
    *,
    ttl_days: float | None = None,
    now: datetime | str | None = None,
) -> bool:
    days = resolve_ttl_days(ttl_days)
    if days <= 0:
        return False
    return last_seen_at(item) <= _now(now) - timedelta(days=days)


def decay_factor(
    item: dict,
    *,
    now: datetime | str | None = None,
    half_life_days: float | None = None,
    floor: float | None = None,
    enabled: bool | None = None,
) -> float:
    """老化系数：`floor + (1-floor) * 0.5 ** (age / half_life)`。

    `MEMORY_DECAY=off` 时恒为 1（方便 A/B 对照）。
    """
    mode = (get_env("MEMORY_DECAY", "on") or "on").strip().lower()
    if enabled is False or (enabled is None and mode in {"off", "0", "false", "no"}):
        return 1.0
    half_life = (
        float(half_life_days)
        if half_life_days is not None
        else get_float("MEMORY_DECAY_HALF_LIFE_DAYS", DEFAULT_HALF_LIFE_DAYS)
    )
    bottom = (
        float(floor)
        if floor is not None
        else get_float("MEMORY_DECAY_FLOOR", DEFAULT_DECAY_FLOOR)
    )
    bottom = min(max(bottom, 0.0), 0.99)
    if half_life <= 0:
        return 1.0
    age = fact_age_days(item, now=now)
    return round(bottom + (1.0 - bottom) * (0.5 ** (age / half_life)), 6)


def describe_fact(item: dict, *, now: datetime | str | None = None) -> dict:
    """给接口/报告用的老化视图：年龄、衰减系数、过期时间、是否已过期。"""
    return {
        "age_days": fact_age_days(item, now=now),
        "decay": decay_factor(item, now=now),
        "expires_at": fact_expires_at(item),
        "expired": is_expired(item, now=now),
    }


def _conn(db_path: Path | str | None = None) -> sqlite3.Connection:
    # MEMORY_FACTS_DB 允许把事实库指到别处（消融/演示用临时库，不动正式数据）
    path = db_path or get_env("MEMORY_FACTS_DB", "") or DB_PATH
    conn = connect(path)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(memory_facts)")}
    if not columns:
        # 事实库自初始化：脚本/演示/单测不需要先手动 init_db
        conn.close()
        from agent.db import init_db

        init_db(path)
        conn = connect(path)
    ensure_fact_schema(conn)
    return conn


def insert_fact(
    session_id: str,
    fact: str,
    *,
    category: str = "general",
    entities: list[str] | None = None,
    importance: float = 0.5,
    source: str = "llm",
    fact_id: str | None = None,
    db_path: Path | str | None = None,
) -> tuple[dict, bool]:
    """插入一条事实；同会话同哈希已存在时返回已有行（`created=False`）。"""
    text = str(fact or "").strip()
    if not text:
        raise ValueError("事实不能为空")
    now = datetime.now().strftime(TIME_FORMAT)
    content_hash = fact_hash(text)
    conn = _conn(db_path)
    try:
        existing = conn.execute(
            "SELECT * FROM memory_facts WHERE session_id = ? AND content_hash = ?",
            (session_id, content_hash),
        ).fetchone()
        if existing is not None:
            return _row_to_dict(existing), False
        record_id = fact_id or uuid4().hex
        conn.execute(
            """
            INSERT INTO memory_facts
                (id, session_id, fact, category, entities, importance,
                 content_hash, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                session_id,
                text,
                category,
                json.dumps(list(entities or []), ensure_ascii=False),
                float(importance),
                content_hash,
                source,
                now,
                now,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM memory_facts WHERE id = ?", (record_id,)).fetchone()
        return _row_to_dict(row), True
    finally:
        conn.close()


def update_fact(
    fact_id: str,
    *,
    fact: str | None = None,
    category: str | None = None,
    entities: list[str] | None = None,
    importance: float | None = None,
    db_path: Path | str | None = None,
) -> dict | None:
    """更新一条事实（相似去重命中时把旧事实"并"成新说法）。"""
    conn = _conn(db_path)
    try:
        row = conn.execute("SELECT * FROM memory_facts WHERE id = ?", (fact_id,)).fetchone()
        if row is None:
            return None
        new_fact = str(fact).strip() if fact else row["fact"]
        fields = {
            "fact": new_fact,
            "category": category or row["category"],
            "entities": json.dumps(
                list(entities) if entities is not None else json.loads(row["entities"] or "[]"),
                ensure_ascii=False,
            ),
            "importance": float(importance if importance is not None else row["importance"]),
            "content_hash": fact_hash(new_fact),
            "updated_at": datetime.now().strftime(TIME_FORMAT),
        }
        conn.execute(
            """
            UPDATE memory_facts SET fact = ?, category = ?, entities = ?,
                   importance = ?, content_hash = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                fields["fact"],
                fields["category"],
                fields["entities"],
                fields["importance"],
                fields["content_hash"],
                fields["updated_at"],
                fact_id,
            ),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM memory_facts WHERE id = ?", (fact_id,)).fetchone()
        return _row_to_dict(updated)
    finally:
        conn.close()


def list_facts(
    session_id: str,
    *,
    limit: int = 50,
    include_expired: bool = True,
    now: datetime | str | None = None,
    db_path: Path | str | None = None,
) -> list[dict]:
    """按会话取事实，最近的在前。

    `include_expired=False` 时按 TTL 过滤（召回与增量去重要用这个口径：
    过期事实既不该被召回，也不该挡住同名的新事实重新写入）。
    """
    conn = _conn(db_path)
    try:
        sql = "SELECT * FROM memory_facts WHERE session_id = ?"
        params: list = [session_id]
        days = resolve_ttl_days()
        if not include_expired and days > 0:
            sql += " AND COALESCE(last_accessed_at, updated_at, created_at) > ?"
            params.append(ttl_cutoff(days, _now(now)))
        sql += " ORDER BY updated_at DESC, created_at DESC LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(row) for row in rows]
    finally:
        conn.close()


def list_all_facts(
    *,
    limit: int = 50,
    include_expired: bool = True,
    now: datetime | str | None = None,
    db_path: Path | str | None = None,
) -> list[dict]:
    """跨会话取最近的事实（`GET /api/memory` 不传 session 时用）。"""
    conn = _conn(db_path)
    try:
        sql = "SELECT * FROM memory_facts WHERE 1 = 1"
        params: list = []
        days = resolve_ttl_days()
        if not include_expired and days > 0:
            sql += " AND COALESCE(last_accessed_at, updated_at, created_at) > ?"
            params.append(ttl_cutoff(days, _now(now)))
        sql += " ORDER BY updated_at DESC, created_at DESC LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(row) for row in rows]
    finally:
        conn.close()


def get_fact(fact_id: str, *, db_path: Path | str | None = None) -> dict | None:
    conn = _conn(db_path)
    try:
        row = conn.execute("SELECT * FROM memory_facts WHERE id = ?", (fact_id,)).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def touch_fact(
    fact_id: str,
    *,
    now: datetime | str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """召回命中时打一次访问时间；`now` 可回填（测试/演示用）。"""
    conn = _conn(db_path)
    try:
        conn.execute(
            "UPDATE memory_facts SET last_accessed_at = ? WHERE id = ?",
            (_now(now).strftime(TIME_FORMAT), fact_id),
        )
        conn.commit()
    finally:
        conn.close()


def touch_facts(
    fact_ids: list[str],
    *,
    now: datetime | str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """批量刷新访问时间；返回实际更新的行数。

    召回一次命中多条时只开一个连接、一条 SQL——别在循环里调 `touch_fact`。
    """
    ids = [str(fact_id) for fact_id in fact_ids if fact_id]
    if not ids:
        return 0
    conn = _conn(db_path)
    try:
        placeholders = ",".join("?" for _ in ids)
        cursor = conn.execute(
            f"UPDATE memory_facts SET last_accessed_at = ? WHERE id IN ({placeholders})",
            [_now(now).strftime(TIME_FORMAT), *ids],
        )
        conn.commit()
        return int(cursor.rowcount or 0)
    finally:
        conn.close()


def list_expired_ids(
    *,
    ttl_days: float | None = None,
    now: datetime | str | None = None,
    db_path: Path | str | None = None,
) -> list[str]:
    """列出已过期事实的 id（按"最近使用 + TTL"判定，和召回同一口径）。"""
    days = resolve_ttl_days(ttl_days)
    if days <= 0:
        return []
    conn = _conn(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id FROM memory_facts
            WHERE COALESCE(last_accessed_at, updated_at, created_at) <= ?
            """,
            (ttl_cutoff(days, _now(now)),),
        ).fetchall()
        return [str(row["id"]) for row in rows]
    finally:
        conn.close()


def purge_expired_facts(
    *,
    ttl_days: float | None = None,
    now: datetime | str | None = None,
    db_path: Path | str | None = None,
    index_path: Path | str | None = None,
) -> list[str]:
    """删除过期事实，并同步删掉向量索引里的记录；返回被删的 id。

    这就是 store 里 `purge_summaries` 的同一套语义：读时过滤 + 定期批量清理。
    索引不同步会留下"幽灵向量"——检索还能算分，但事实已不在库里。
    """
    ids = list_expired_ids(ttl_days=ttl_days, now=now, db_path=db_path)
    if not ids:
        return []
    conn = _conn(db_path)
    try:
        placeholders = ",".join("?" for _ in ids)
        conn.execute(f"DELETE FROM memory_facts WHERE id IN ({placeholders})", ids)
        conn.commit()
    finally:
        conn.close()

    try:
        from agent.memory.index import MemoryIndex
        from agent.memory.writer import default_index_path

        index = MemoryIndex(index_path or default_index_path()).load()
        removed = 0
        for fact_id in ids:
            if fact_id in index.records:
                index.drop(fact_id)
                removed += 1
        if removed:
            index.save()
    except Exception as exc:  # noqa: BLE001 - 索引清理失败不该阻断数据库清理
        print(f"[warn] 过期事实的索引清理失败：{type(exc).__name__}: {exc}")
    return ids


def count_facts(session_id: str, *, db_path: Path | str | None = None) -> int:
    conn = _conn(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM memory_facts WHERE session_id = ?", (session_id,)
        ).fetchone()
        return int(row["n"])
    finally:
        conn.close()
