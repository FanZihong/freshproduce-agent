"""会话与消息持久化（复用 agent 的 SQLite 文件）。"""

from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

from agent.db import connect
from agent.memory.short_term import touch

DEFAULT_TITLE = "新的售后咨询"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _row_to_session(row) -> dict:
    return {
        "id": row["session_id"],
        "title": row["title"],
        "status": row["status"],
        "lastMessage": row["last_message"] or "",
        "updatedAt": row["updated_at"],
    }


def ensure_session(session_id: str, title: str | None = None) -> dict:
    """会话不存在时自动创建，并刷新活跃时间。"""
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            now = _now()
            conn.execute(
                """
                INSERT INTO sessions (session_id, title, status, last_message, created_at, updated_at)
                VALUES (?, ?, 'active', '', ?, ?)
                """,
                (session_id, title or DEFAULT_TITLE, now, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        elif title and (row["title"] or "") in (DEFAULT_TITLE, ""):
            # 会话标题还是默认值时，用第一条用户消息命名
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (title, _now(), session_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
    finally:
        conn.close()
    touch(session_id)
    return _row_to_session(row)


def create_session(title: str | None = None) -> dict:
    return ensure_session(f"s-{uuid4().hex[:8]}", title)


def list_sessions() -> list[dict]:
    conn = connect()
    try:
        rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
    finally:
        conn.close()
    if not rows:
        return [create_session()]
    return [_row_to_session(row) for row in rows]


def get_session(session_id: str) -> dict | None:
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    finally:
        conn.close()
    return _row_to_session(row) if row else None


def update_session(
    session_id: str,
    *,
    title: str | None = None,
    status: str | None = None,
    last_message: str | None = None,
) -> None:
    fields = ["updated_at = ?"]
    values: list = [_now()]
    if title is not None:
        fields.append("title = ?")
        values.append(title)
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if last_message is not None:
        fields.append("last_message = ?")
        values.append(last_message)
    values.append(session_id)

    conn = connect()
    try:
        conn.execute(f"UPDATE sessions SET {', '.join(fields)} WHERE session_id = ?", values)
        conn.commit()
    finally:
        conn.close()


def append_message(
    session_id: str,
    role: str,
    content: str,
    *,
    intent: dict | None = None,
    slots: dict | None = None,
    tool_calls: list[dict] | None = None,
    trace_id: str | None = None,
) -> str:
    message_id = f"m-{uuid4().hex[:10]}"
    conn = connect()
    try:
        conn.execute(
            """
            INSERT INTO messages
                (message_id, session_id, role, content, intent, confidence, slots, tool_calls, trace_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session_id,
                role,
                content,
                json.dumps(intent, ensure_ascii=False) if intent else None,
                float(intent["confidence"]) if intent and intent.get("confidence") is not None else None,
                json.dumps(slots, ensure_ascii=False) if slots else None,
                json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                trace_id,
                _now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return message_id


def _row_to_message(row) -> dict:
    return {
        "id": row["message_id"],
        "sessionId": row["session_id"],
        "role": row["role"],
        "content": row["content"],
        "createdAt": row["created_at"],
        "intent": json.loads(row["intent"]) if row["intent"] else None,
        "slots": json.loads(row["slots"]) if row["slots"] else None,
        "toolCalls": json.loads(row["tool_calls"]) if row["tool_calls"] else None,
        "retrieval": None,
        "traceId": row["trace_id"],
    }


def get_messages(session_id: str) -> list[dict]:
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at, rowid",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_message(row) for row in rows]
