"""建单幂等 + 网关重试安全的测试。

核心验收：同一「会话 + 订单 + 原因」连调三次，只落一行；重试/重复提交都返回同一张工单。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.db import connect, init_db
from agent.tools import gateway, tickets
from agent.tools.gateway import ToolSpec, call_tool
from agent.tools.schemas import QueryOrderInput

ORDER = "2026093000123"
REASON = "坏果理赔"


def make_db(tmp_path: Path) -> Path:
    db = tmp_path / "tickets.db"
    init_db(db)
    return db


def rows(db: Path) -> list:
    conn = connect(db)
    try:
        return conn.execute("SELECT * FROM tickets ORDER BY ticket_id").fetchall()
    finally:
        conn.close()


def create(db: Path, **overrides):
    kwargs = {
        "session_id": "s-1",
        "order_id": ORDER,
        "reason": REASON,
        "evidence": "文字描述（无照片）",
        "refund_amount": 12.9,
        "db_path": db,
    }
    kwargs.update(overrides)
    return tickets.create_ticket(**kwargs)


# ---------------- 幂等键与建单 ----------------


def test_idempotency_key_is_stable_and_scoped():
    base = tickets.build_idempotency_key("s-1", ORDER, REASON)
    assert base == tickets.build_idempotency_key(" s-1 ", f" {ORDER} ", f" {REASON} ")
    assert base != tickets.build_idempotency_key("s-2", ORDER, REASON)
    assert base != tickets.build_idempotency_key("s-1", "2026092800451", REASON)
    assert base != tickets.build_idempotency_key("s-1", ORDER, "缺斤少两")
    assert len(base) == 32


def test_three_calls_create_one_ticket(tmp_path):
    """验收主线：连调三次只插一行，后两次返回同一张工单。"""
    db = make_db(tmp_path)
    first = create(db)
    second = create(db)
    third = create(db)

    assert first["idempotent"] is False
    assert [second["idempotent"], third["idempotent"]] == [True, True]
    assert len({first["ticket_id"], second["ticket_id"], third["ticket_id"]}) == 1
    assert len(rows(db)) == 1


def test_different_reason_order_or_session_creates_new_ticket(tmp_path):
    db = make_db(tmp_path)
    create(db)
    create(db, reason="缺斤少两")
    create(db, order_id="2026092800451")
    create(db, session_id="s-2")
    assert len(rows(db)) == 4


def test_returned_ticket_keeps_original_fields(tmp_path):
    db = make_db(tmp_path)
    first = create(db)
    again = create(db, evidence="第二次的凭证", refund_amount=99.0)
    assert again["ticket_id"] == first["ticket_id"]
    assert again["refund_amount"] == first["refund_amount"]     # 不覆盖已有工单
    assert again["created_at"] == first["created_at"]


def test_concurrent_insert_returns_winner(tmp_path, monkeypatch):
    """并发场景：查完没插到、插入时撞唯一索引 -> 回读赢家，不抛异常。"""
    db = make_db(tmp_path)
    key = tickets.build_idempotency_key("s-1", ORDER, REASON)

    def race_next_ticket(conn):
        # 模拟另一个请求在 SELECT 之后、INSERT 之前抢先落库
        conn.execute(
            """
            INSERT INTO tickets
                (ticket_id, session_id, order_id, reason, evidence, refund_amount,
                 status, created_at, idempotency_key)
            VALUES ('TK-WINNER', 's-1', ?, ?, '对方的凭证', 20.0, 'pending',
                    '2026-10-19 10:00:00', ?)
            """,
            (ORDER, REASON, key),
        )
        conn.commit()
        return "TK-LOSER"

    monkeypatch.setattr(tickets, "_next_ticket_id", race_next_ticket)
    result = create(db)
    assert result["ticket_id"] == "TK-WINNER"
    assert result["idempotent"] is True
    assert len(rows(db)) == 1


def test_old_schema_is_migrated(tmp_path):
    """老库（tickets 无 idempotency_key 列）升级后照样能幂等建单。"""
    db = tmp_path / "old.db"
    conn = connect(db)
    conn.executescript(
        """
        CREATE TABLE tickets (
            ticket_id      TEXT PRIMARY KEY,
            session_id     TEXT NOT NULL,
            order_id       TEXT NOT NULL,
            reason         TEXT NOT NULL,
            evidence       TEXT,
            refund_amount  REAL,
            status         TEXT NOT NULL DEFAULT 'pending',
            created_at     TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

    init_db(db)      # 迁移：补列 + 建唯一索引
    conn = connect(db)
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(tickets)")}
    finally:
        conn.close()
    assert "idempotency_key" in columns

    create(db)
    create(db)
    assert len(rows(db)) == 1


# ---------------- 网关：重试安全 ----------------


def test_write_tool_is_marked_retry_safe():
    assert gateway.TOOL_REGISTRY["create_ticket"].retry_safe is True
    for name in ("query_order", "query_logistics", "query_policy", "search_faq"):
        assert gateway.TOOL_REGISTRY[name].retry_safe is True
    # 默认值必须保守：新加的写工具忘了标，就不许被重试
    assert ToolSpec(QueryOrderInput, lambda **kwargs: None).retry_safe is False


def test_gateway_retries_are_idempotent(tmp_path, monkeypatch):
    """走网关连调三次（等价于重试/重复提交）也只落一行。"""
    db = make_db(tmp_path)
    monkeypatch.setattr(tickets, "DB_PATH", db)

    results = [
        call_tool(
            "create_ticket",
            session_id="s-1",
            order_id=ORDER,
            reason=REASON,
            evidence="文字描述",
            refund_amount=12.9,
        )
        for _ in range(3)
    ]
    assert all(result.ok for result in results)
    assert len({result.data["ticket_id"] for result in results}) == 1
    assert [result.data["idempotent"] for result in results] == [False, True, True]
    assert len(rows(db)) == 1


def test_non_retry_safe_tool_is_called_once(monkeypatch):
    calls = {"n": 0}

    def boom(**kwargs):
        calls["n"] += 1
        raise RuntimeError("写入失败")

    monkeypatch.setitem(
        gateway.TOOL_REGISTRY,
        "unsafe_write",
        ToolSpec(QueryOrderInput, boom, attempts=3, base_delay=0.01, retry_safe=False),
    )
    result = call_tool("unsafe_write", order_id=ORDER)
    assert result.ok is False and result.kind == "error"
    assert result.attempts == 1 and calls["n"] == 1

    calls["n"] = 0
    monkeypatch.setitem(
        gateway.TOOL_REGISTRY,
        "safe_write",
        ToolSpec(QueryOrderInput, boom, attempts=3, base_delay=0.01, retry_safe=True),
    )
    result = call_tool("safe_write", order_id=ORDER)
    assert result.attempts == 3 and calls["n"] == 3
