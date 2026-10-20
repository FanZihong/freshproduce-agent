"""订单与物流工具：直接查询 SQLite。"""

from __future__ import annotations

from agent.db import connect


def query_order(order_id: str) -> dict | None:
    """按订单号查订单，查不到返回 None。"""
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def query_logistics(order_id: str) -> dict:
    """今天先用订单状态模拟物流节点，明天接真实物流接口。"""
    order = query_order(order_id)
    if order is None:
        return {"ok": False, "message": f"没有查到订单 {order_id} 的物流记录"}

    status = order["status"]
    if status == "已签收":
        message = f"包裹已于 {order['signed_at']} 签收，签收人为本人"
    elif status == "运输中":
        message = "包裹运输中，预计明日 18:00 前送达"
    elif status == "已发货":
        message = "商家已发货，等待快递揽收"
    else:
        message = f"当前状态：{status}"
    return {"ok": True, "status": status, "message": message}
