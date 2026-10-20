"""工单接口：数据来自 agent 的 SQLite。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agent.db import connect
from agent.tools.tickets import create_ticket

from app.api.schemas import TicketCreateRequest, TicketOut, TicketUpdateRequest

router = APIRouter(tags=["tickets"])

HIGH_PRIORITY_KEYWORDS = ("争议", "人工", "超", "投诉")


def _row_to_ticket(row) -> dict:
    reason = row["reason"] or ""
    priority = "high" if any(keyword in reason for keyword in HIGH_PRIORITY_KEYWORDS) else "normal"
    refund = row["refund_amount"]
    summary = f"凭证：{row['evidence'] or '无'}；赔付：{refund:.2f} 元" if refund else "等待人工核实"
    return {
        "id": row["ticket_id"],
        "sessionId": row["session_id"],
        "orderId": row["order_id"],
        "reason": reason,
        "summary": summary,
        "status": row["status"],
        "priority": priority,
        "createdAt": row["created_at"],
        "assignee": "待分配" if row["status"] == "pending" else "客服组",
    }


def _get_ticket_row(ticket_id: str):
    conn = connect()
    try:
        return conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
    finally:
        conn.close()


@router.get("/tickets", response_model=list[TicketOut])
def list_tickets() -> list[dict]:
    conn = connect()
    try:
        rows = conn.execute("SELECT * FROM tickets ORDER BY created_at DESC").fetchall()
    finally:
        conn.close()
    return [_row_to_ticket(row) for row in rows]


@router.get("/tickets/{ticket_id}", response_model=TicketOut)
def get_ticket(ticket_id: str) -> dict:
    row = _get_ticket_row(ticket_id)
    if row is None:
        raise HTTPException(status_code=404, detail="工单不存在")
    return _row_to_ticket(row)


@router.post("/tickets", response_model=TicketOut)
def post_ticket(payload: TicketCreateRequest) -> dict:
    created = create_ticket(
        session_id=payload.sessionId,
        order_id=payload.orderId,
        reason=payload.reason,
        evidence=payload.summary or None,
        refund_amount=None,
    )
    row = _get_ticket_row(created["ticket_id"])
    return _row_to_ticket(row)


@router.patch("/tickets/{ticket_id}", response_model=TicketOut)
def patch_ticket(ticket_id: str, payload: TicketUpdateRequest) -> dict:
    conn = connect()
    try:
        conn.execute("UPDATE tickets SET status = ? WHERE ticket_id = ?", (payload.status, ticket_id))
        conn.commit()
    finally:
        conn.close()
    row = _get_ticket_row(ticket_id)
    if row is None:
        raise HTTPException(status_code=404, detail="工单不存在")
    return _row_to_ticket(row)
