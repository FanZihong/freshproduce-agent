"""工单工具：把理赔结果写进 SQLite，返回工单号。

**幂等**：网关对 create_ticket 是"重试安全"的（失败会重试、调用方也可能重复提交），
所以建单必须幂等——同一条「会话 + 订单 + 原因」只落一张工单，重复调用返回已有的那张。
这也是业务规则本身的要求（《赔付标准表》1.3：同一订单同一问题只赔付一次，
重复申请会被拦截并提示已有工单）。

实现是"幂等键 + 唯一索引"：键是三者拼接后的 sha256 前缀，唯一索引兜住并发；
插入冲突时回滚并回读赢家，保证"重试不会多建一张"。
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

from agent.db import DB_PATH, connect, ensure_ticket_schema

# 幂等键带版本号：将来规则变了（比如把"凭证"也纳入键）可以升级而不误撞老键
IDEMPOTENCY_VERSION = "v1"


def build_idempotency_key(session_id: str, order_id: str, reason: str) -> str:
    """幂等键 = sha256(版本 | 会话 | 订单 | 原因) 的前 32 位。

    只用三个稳定字段：金额、凭证这类会随重试轻微变化的内容不进键，
    否则"同一次理赔的重试"会因为参数细微差别被当成两次请求。
    """
    raw = "|".join(
        [
            IDEMPOTENCY_VERSION,
            str(session_id or "").strip(),
            str(order_id or "").strip(),
            str(reason or "").strip(),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _next_ticket_id(conn) -> str:
    today = datetime.now().strftime("%Y%m%d")
    row = conn.execute("SELECT COUNT(*) AS n FROM tickets").fetchone()
    seq = int(row["n"]) + 1
    return f"TK-{today}-{seq:04d}"


def create_ticket(
    session_id: str,
    order_id: str,
    reason: str,
    evidence: str | None,
    refund_amount: float | None,
    *,
    db_path: Path | str | None = None,
) -> dict:
    """创建工单（幂等）：已存在同键工单时直接返回它，`idempotent=True` 便于观测。"""
    key = build_idempotency_key(session_id, order_id, reason)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = connect(db_path or DB_PATH)
    try:
        ensure_ticket_schema(conn)  # 老库补列，保证唯一索引真的能建上
        existing = conn.execute(
            "SELECT * FROM tickets WHERE idempotency_key = ?", (key,)
        ).fetchone()
        if existing is not None:
            return _payload(existing, idempotent=True)

        ticket_id = _next_ticket_id(conn)
        try:
            conn.execute(
                """
                INSERT INTO tickets
                    (ticket_id, session_id, order_id, reason, evidence,
                     refund_amount, status, created_at, idempotency_key)
                VALUES
                    (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    ticket_id,
                    session_id,
                    order_id,
                    reason,
                    evidence,
                    refund_amount,
                    created_at,
                    key,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            # 并发下另一个请求先插了同一个键：回滚后回读赢家，返回它的工单号
            conn.rollback()
            winner = conn.execute(
                "SELECT * FROM tickets WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if winner is None:
                raise
            return _payload(winner, idempotent=True)
    finally:
        conn.close()

    return {
        "ticket_id": ticket_id,
        "status": "pending",
        "created_at": created_at,
        "refund_amount": refund_amount,
        "idempotent": False,
    }


def _payload(row: sqlite3.Row, *, idempotent: bool) -> dict:
    """把已存在的工单行转成和新建时一致的返回结构。"""
    return {
        "ticket_id": row["ticket_id"],
        "status": row["status"],
        "created_at": row["created_at"],
        "refund_amount": row["refund_amount"],
        "idempotent": idempotent,
    }
