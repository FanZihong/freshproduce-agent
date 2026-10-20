"""Memory 存取协议与双后端的测试：SQLite 现状行为、Redis 语义对齐、环境切换。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from agent.db import init_db
from agent.memory import long_term as long_term_module
from agent.memory import short_term as short_term_module
from agent.memory.store import (
    ACTIVITY_TTL_SECONDS,
    MemoryStore,
    RedisMemoryStore,
    SQLiteMemoryStore,
    get_store,
    reset_store,
)


class FakeRedis:
    """只实现 RedisMemoryStore 用到的五个命令，方便在没装 redis 的机器上测语义。"""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttl: dict[str, int | None] = {}

    def ping(self) -> bool:
        return True

    def set(self, key, value, ex=None):
        self.data[key] = value
        self.ttl[key] = ex
        return True

    def get(self, key):
        return self.data.get(key)

    def delete(self, *keys) -> int:
        removed = 0
        for key in keys:
            if key in self.data:
                self.data.pop(key)
                self.ttl.pop(key, None)
                removed += 1
        return removed


def run_scenario(store: MemoryStore) -> list:
    """同一段行为脚本跑在两个后端上，输出必须完全一致（这是协议的核心约定）。"""
    results = []
    store.set_activity("s1", "2026-10-19 10:00:00", ttl_seconds=ACTIVITY_TTL_SECONDS)
    results.append(store.get_activity("s1"))
    store.set_activity("s1", "2026-10-19 10:05:00", ttl_seconds=ACTIVITY_TTL_SECONDS)
    results.append(store.get_activity("s1"))
    store.clear_activity("s1")
    results.append(store.get_activity("s1"))

    store.set_summary("s2", "摘要内容", "2026-10-19 11:00:00")
    results.append(store.get_summary("s2", "2026-10-19 10:30:00"))   # 未过期
    store.set_summary("s3", "待删除摘要", "2026-10-19 11:00:00")
    store.clear_summary("s3")
    results.append(store.get_summary("s3", "2026-10-19 10:30:00"))   # 主动删除（用户要求删记录）
    results.append(store.get_summary("s2", "2026-10-19 11:00:00"))   # 恰好到期 -> 删除+None
    results.append(store.get_summary("s2", "2026-10-19 12:00:00"))   # 已经删了
    results.append(store.get_summary("没有这个会话", "2026-10-19 10:30:00"))
    return results


EXPECTED_SCENARIO = [
    "2026-10-19 10:00:00",
    "2026-10-19 10:05:00",
    None,
    "摘要内容",
    None,
    None,
    None,
    None,
]


def test_sqlite_store_scenario(tmp_path):
    db = tmp_path / "memory.db"
    init_db(db)
    assert run_scenario(SQLiteMemoryStore(db)) == EXPECTED_SCENARIO


def test_redis_store_scenario_matches_sqlite():
    client = FakeRedis()
    store = RedisMemoryStore("redis://localhost:6379/0", client=client)
    assert run_scenario(store) == EXPECTED_SCENARIO
    # 键名带 agent: 前缀，和同实例上的其他数据隔开
    assert all(key.startswith("agent:") for key in client.data)
    assert "agent:memory:s2" not in client.data          # 过期即删
    store.set_activity("s9", "2026-10-19 10:00:00", ttl_seconds=ACTIVITY_TTL_SECONDS)
    assert client.ttl.get("agent:activity:s9") == ACTIVITY_TTL_SECONDS


def test_redis_summary_uses_remaining_ttl():
    client = FakeRedis()
    store = RedisMemoryStore("redis://x", client=client)
    expires = datetime.now() + timedelta(days=7)
    store.set_summary("s", "摘要", expires.strftime("%Y-%m-%d %H:%M:%S"))
    raw = json.loads(client.data["agent:memory:s"])
    assert raw == {"summary": "摘要", "expires_at": expires.strftime("%Y-%m-%d %H:%M:%S")}
    assert 7 * 24 * 3600 - 10 < client.ttl["agent:memory:s"] <= 7 * 24 * 3600


def test_sqlite_store_expiry_and_purge(tmp_path):
    db = tmp_path / "memory.db"
    init_db(db)
    store = SQLiteMemoryStore(db)
    store.set_summary("旧", "过期记忆", "2026-10-01 00:00:00")
    store.set_summary("新", "有效记忆", "2099-01-01 00:00:00")
    assert store.purge_summaries("2026-10-19 00:00:00") == 1
    assert store.get_summary("旧", "2026-10-19 00:00:00") is None
    assert store.get_summary("新", "2026-10-19 00:00:00") == "有效记忆"


# ---------------- 短期/长期记忆改用协议后，行为保持不变 ----------------


def test_short_term_goes_through_store(monkeypatch):
    store = RedisMemoryStore("redis://x", client=FakeRedis())
    monkeypatch.setattr(short_term_module, "get_store", lambda: store)

    short_term_module.reset_for_test("s")
    assert short_term_module.idle_minutes("s") is None

    short_term_module.set_last_active("s", datetime.now() - timedelta(minutes=40))
    idle = short_term_module.idle_minutes("s")
    assert 39 < (idle or 0) < 41
    assert short_term_module.should_reset("s", timeout_minutes=30) is True
    assert short_term_module.should_reset("s", timeout_minutes=60) is False

    short_term_module.touch("s")
    assert (short_term_module.idle_minutes("s") or 0) < 1
    assert short_term_module.should_reset("s") is False
    short_term_module.reset_for_test("s")
    assert short_term_module.idle_minutes("s") is None


def test_long_term_goes_through_store_and_masks_pii(monkeypatch):
    store = RedisMemoryStore("redis://x", client=FakeRedis())
    monkeypatch.setattr(long_term_module, "get_store", lambda: store)

    saved = long_term_module.save_summary("s", "13812345678 在 花园小区 3 号楼，喜欢草莓")
    assert saved["session_id"] == "s" and saved["expires_at"]
    loaded = long_term_module.load_summary("s")
    assert loaded is not None
    assert "13812345678" not in loaded          # 脱敏仍然生效
    assert "[地址已脱敏]" in loaded
    assert "喜欢草莓" in loaded


def test_long_term_expired_memory_returns_none(tmp_path, monkeypatch):
    db = tmp_path / "memory.db"
    init_db(db)
    monkeypatch.setattr(long_term_module, "get_store", lambda: SQLiteMemoryStore(db))
    # ttl 为负 -> 立刻过期
    long_term_module.save_summary("s", "过期摘要", ttl_days=-1)
    assert long_term_module.load_summary("s") is None


# ---------------- 环境切换 ----------------


def test_get_store_defaults_to_sqlite(monkeypatch):
    monkeypatch.delenv("MEMORY_STORE", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    reset_store()
    try:
        assert get_store().backend == "sqlite"
    finally:
        reset_store()


def test_get_store_redis_mode_requires_url(monkeypatch):
    monkeypatch.setenv("MEMORY_STORE", "redis")
    monkeypatch.delenv("REDIS_URL", raising=False)
    reset_store()
    try:
        with pytest.raises(RuntimeError):
            get_store()
    finally:
        reset_store()


def test_get_store_redis_mode_surfaces_connection_error(monkeypatch):
    """MEMORY_STORE=redis 是强要求：连不上就报错，不静默降级。"""
    monkeypatch.setenv("MEMORY_STORE", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    reset_store()
    try:
        with pytest.raises(RuntimeError):
            get_store()
    finally:
        reset_store()


def test_get_store_auto_falls_back_to_sqlite(monkeypatch):
    monkeypatch.setenv("MEMORY_STORE", "auto")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    reset_store()
    try:
        assert get_store().backend == "sqlite"      # 没装 redis / 连不上 -> 回落
    finally:
        reset_store()


def test_get_store_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv("MEMORY_STORE", "bogus")
    monkeypatch.delenv("REDIS_URL", raising=False)
    reset_store()
    try:
        with pytest.raises(RuntimeError):
            get_store()
    finally:
        reset_store()


# ---------------- 运维护栏：后端挂掉时按"无记忆"继续 ----------------


class BrokenStore(FakeRedis):
    """所有操作都抛异常，模拟 Redis 宕机/网络抖动。"""

    def set(self, *a, **k):
        raise ConnectionError("redis 连接中断")

    def get(self, *a, **k):
        raise ConnectionError("redis 连接中断")


def test_memory_failures_do_not_break_conversation(monkeypatch):
    store = RedisMemoryStore("redis://x", client=BrokenStore())
    monkeypatch.setattr(short_term_module, "get_store", lambda: store)
    monkeypatch.setattr(long_term_module, "get_store", lambda: store)

    short_term_module.touch("s")                     # 不抛异常
    assert short_term_module.idle_minutes("s") is None
    assert short_term_module.should_reset("s") is False   # 判断不了 -> 不清空槽位

    assert long_term_module.load_summary("s") is None
    saved = long_term_module.save_summary("s", "摘要")
    assert saved["degraded"] is True and saved["expires_at"] == ""
    assert long_term_module.purge_expired() == 0
