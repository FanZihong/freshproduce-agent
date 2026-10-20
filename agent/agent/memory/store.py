"""Memory 存取协议 + 双后端实现（SQLite 默认 / Redis 可选）。

为什么抽象这一层：短期会话活跃、长期用户摘要之前直接散在两处硬编码 SQLite，
换存储、加 TTL、做多实例部署都得改业务代码。抽成一个小协议后：

- **默认零变化**：`MEMORY_STORE` 不配、`REDIS_URL` 不配时，行为和以前完全一样（SQLite）；
- **多实例/生产可选**：配置 `REDIS_URL` 后自动切 Redis——会话活跃用原生 TTL 做垃圾回收，
  长期摘要用"过期时间 + TTL 双保险"；
- **可测试**：单测里注入假 store，不碰真数据库。

语义约定（两个后端必须一致，这也是单测要卡的东西）：
- 时间一律用 `%Y-%m-%d %H:%M:%S` 字符串比较，和现有代码口径相同；
- `get_summary` 遇到已过期的记录：删除并返回 None（读即清理）；
- Redis 的 TTL 只做垃圾回收，**不改变判定语义**——过期与否仍由值里的时间决定，
  这样两种后端在"时钟、TTL 抖动"面前行为一致。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from agent.config import get_env
from agent.db import DB_PATH, connect

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# 会话活跃键的 Redis TTL：取闲置阈值的两倍。
# 必须比阈值长——否则"闲置超时"和"键被回收"分不开，重置逻辑会失效；
# TTL 只是让长期不来的会话别在 Redis 里堆着。
ACTIVITY_TTL_SECONDS = 3600


def warn(exc: Exception, *, action: str = "memory") -> None:
    """记忆后端异常时的统一告警：memory 不是关键路径，按"无记忆"继续，别把对话带崩。"""
    print(f"[warn] {action} 失败，按无记忆继续：{type(exc).__name__}: {exc}")


def _parse(value: str) -> datetime:
    return datetime.strptime(value, TIME_FORMAT)


def _format(value: datetime) -> str:
    return value.strftime(TIME_FORMAT)


def ttl_cutoff(ttl_days: float, now: datetime | None = None) -> str:
    """TTL 的统一入口：返回"早于它就过期"的时间字符串。

    摘要的过期判定是"拿 expires_at 和当前时间比"，事实是"拿最近一次使用时间和
    截止时间比"——两者都归一到同一套字符串时间口径，避免出现"有的地方按秒、
    有的地方按天"的两套语义。
    """
    moment = now or datetime.now()
    return _format(moment - timedelta(days=float(ttl_days)))


class MemoryStore(Protocol):
    """记忆存取的统一协议。实现者：SQLiteMemoryStore / RedisMemoryStore。"""

    backend: str

    def set_activity(self, session_id: str, when: str, *, ttl_seconds: int | None = None) -> None: ...

    def get_activity(self, session_id: str) -> str | None: ...

    def clear_activity(self, session_id: str) -> None: ...

    def set_summary(self, session_id: str, summary: str, expires_at: str) -> None: ...

    def get_summary(self, session_id: str, now: str) -> str | None: ...

    def clear_summary(self, session_id: str) -> None: ...

    def purge_summaries(self, now: str) -> int: ...


class SQLiteMemoryStore:
    """现状的存储：session_activity / user_memory 两张表，行为与历史版本完全一致。"""

    backend = "sqlite"

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        return connect(self.db_path)

    def set_activity(self, session_id: str, when: str, *, ttl_seconds: int | None = None) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO session_activity (session_id, last_active_at)
                VALUES (?, ?)
                ON CONFLICT(session_id) DO UPDATE SET last_active_at = excluded.last_active_at
                """,
                (session_id, when),
            )
            conn.commit()
        finally:
            conn.close()

    def get_activity(self, session_id: str) -> str | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT last_active_at FROM session_activity WHERE session_id = ?", (session_id,)
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else str(row["last_active_at"])

    def clear_activity(self, session_id: str) -> None:
        conn = self._conn()
        try:
            conn.execute("DELETE FROM session_activity WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()

    def set_summary(self, session_id: str, summary: str, expires_at: str) -> None:
        now = datetime.now().strftime(TIME_FORMAT)
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO user_memory (session_id, summary, created_at, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at
                """,
                (session_id, summary, now, now, expires_at),
            )
            conn.commit()
        finally:
            conn.close()

    def get_summary(self, session_id: str, now: str) -> str | None:
        if not session_id:
            return None
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT summary, expires_at FROM user_memory WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            if _parse(str(row["expires_at"])) <= _parse(now):
                conn.execute("DELETE FROM user_memory WHERE session_id = ?", (session_id,))
                conn.commit()
                return None
            return str(row["summary"])
        finally:
            conn.close()

    def clear_summary(self, session_id: str) -> None:
        conn = self._conn()
        try:
            conn.execute("DELETE FROM user_memory WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()

    def purge_summaries(self, now: str) -> int:
        conn = self._conn()
        try:
            cursor = conn.execute(
                "DELETE FROM user_memory WHERE expires_at <= ?", (now,)
            )
            conn.commit()
            return cursor.rowcount or 0
        finally:
            conn.close()


class RedisMemoryStore:
    """Redis 后端：会话活跃/长期摘要各一个键，TTL 只做垃圾回收。

    键名带 `agent:` 前缀，方便和同实例上的其他业务数据隔开。
    值仍存时间字符串（活跃）或 {summary, expires_at}（摘要），判定语义与 SQLite 一致。
    """

    backend = "redis"

    def __init__(self, url: str, *, client=None) -> None:
        self.url = url
        if client is not None:
            self.client = client
            return
        import redis  # noqa: PLC0415 - 只在真用 Redis 时才引入这个依赖

        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.client.ping()  # 立刻暴露连接问题，别等到第一轮对话才炸

    def _activity_key(self, session_id: str) -> str:
        return f"agent:activity:{session_id}"

    def _summary_key(self, session_id: str) -> str:
        return f"agent:memory:{session_id}"

    def set_activity(self, session_id: str, when: str, *, ttl_seconds: int | None = None) -> None:
        key = self._activity_key(session_id)
        if ttl_seconds:
            self.client.set(key, when, ex=ttl_seconds)
        else:
            self.client.set(key, when)

    def get_activity(self, session_id: str) -> str | None:
        return self.client.get(self._activity_key(session_id))

    def clear_activity(self, session_id: str) -> None:
        self.client.delete(self._activity_key(session_id))

    def set_summary(self, session_id: str, summary: str, expires_at: str) -> None:
        key = self._summary_key(session_id)
        payload = json.dumps({"summary": summary, "expires_at": expires_at}, ensure_ascii=False)
        remaining = int((_parse(expires_at) - datetime.now()).total_seconds())
        if remaining > 0:
            self.client.set(key, payload, ex=remaining)
        else:
            self.client.set(key, payload)  # 已经过期：交给 get_summary 判死，不在这边写脏数据

    def get_summary(self, session_id: str, now: str) -> str | None:
        if not session_id:
            return None
        raw = self.client.get(self._summary_key(session_id))
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.client.delete(self._summary_key(session_id))
            return None
        if _parse(str(payload.get("expires_at") or now)) <= _parse(now):
            self.client.delete(self._summary_key(session_id))
            return None
        return str(payload.get("summary") or "")

    def clear_summary(self, session_id: str) -> None:
        self.client.delete(self._summary_key(session_id))

    def purge_summaries(self, now: str) -> int:
        # 键级 TTL 已经把过期数据回收了，这里没有可批量扫描的全量视图
        return 0


_STORE: MemoryStore | None = None


def get_store() -> MemoryStore:
    """按配置返回 memory 后端：auto 有 REDIS_URL 就用 Redis（失败回退 SQLite）。

    `MEMORY_STORE=redis` 是强要求：没有 REDIS_URL 或连不上就直接报错，
    避免"以为在用 Redis 其实在写 SQLite"这种最坑人的静默降级。
    """
    global _STORE
    if _STORE is not None:
        return _STORE

    mode = (get_env("MEMORY_STORE", "auto") or "auto").strip().lower()
    redis_url = (get_env("REDIS_URL", "") or "").strip()
    if mode not in {"auto", "sqlite", "redis"}:
        raise RuntimeError(f"MEMORY_STORE 只能是 auto/sqlite/redis，收到：{mode}")

    if mode == "redis" and not redis_url:
        raise RuntimeError("MEMORY_STORE=redis 需要配置 REDIS_URL（例如 redis://localhost:6379/0）")
    if mode in {"auto", "redis"} and redis_url:
        try:
            _STORE = RedisMemoryStore(redis_url)
            return _STORE
        except Exception as exc:  # noqa: BLE001
            if mode == "redis":
                raise RuntimeError(f"Redis 连接失败（{type(exc).__name__}: {exc}）") from exc
            print(f"[warn] REDIS_URL 已配置但连接失败，回落 SQLite：{type(exc).__name__}: {exc}")

    _STORE = SQLiteMemoryStore()
    return _STORE


def reset_store(store: MemoryStore | None = None) -> None:
    """测试/热切换用：清掉单例，下次 get_store 重新按环境解析。"""
    global _STORE
    _STORE = store
