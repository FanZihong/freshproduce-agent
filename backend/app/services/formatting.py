"""把 Agent 内部结构转成前端约定的字段格式。"""

from __future__ import annotations

from uuid import uuid4

from app.config import INTENT_LABELS, TOOL_LABELS

SLOT_LABELS = {
    "order_id": "订单号",
    "damage_ratio": "坏果比例",
    "evidence": "凭证",
    "refund_amount": "赔付金额",
}


def intent_payload(label: str, confidence: float) -> dict:
    return {
        "label": label,
        "name": INTENT_LABELS.get(label, label),
        "confidence": round(float(confidence or 0), 2),
    }


def _slot_value_text(value) -> str:
    if isinstance(value, float):
        return f"{value:.0%}"
    return str(value)


def slots_to_items(slots: dict | None) -> list[dict]:
    """槽位字典 -> 前端 SlotItem 列表。"""
    items: list[dict] = []
    for key, label in SLOT_LABELS.items():
        if key == "refund_amount" and not (slots or {}).get(key):
            continue
        value = (slots or {}).get(key)
        items.append(
            {
                "key": key,
                "label": label,
                "value": _slot_value_text(value) if value is not None else None,
                "required": key != "refund_amount",
            }
        )
    return items


def _summarize_tool_result(name: str, data) -> str:
    if not isinstance(data, dict):
        return str(data)
    if name == "query_order":
        return f"{data.get('product')}，实付 {float(data.get('amount', 0)):.2f} 元，状态：{data.get('status')}"
    if name == "query_logistics":
        return str(data.get("message") or data)
    if name == "query_policy":
        if data.get("matched") is False:
            return str(data.get("reason") or "规则未覆盖")
        return (
            f"赔付比例 {float(data.get('refund_ratio', 0)):.0%}，"
            f"金额 {float(data.get('refund_amount', 0)):.2f} 元"
        )
    if name == "create_ticket":
        return f"工单号 {data.get('ticket_id')}，状态 {data.get('status')}"
    return str(data)


def tool_call_payload(item: dict) -> dict:
    """Agent 的 tool_results 记录 -> 前端 ToolCall。"""
    ok = bool(item.get("ok"))
    error = item.get("error") or {}
    name = item.get("name") or "unknown"
    if ok:
        result_text = _summarize_tool_result(name, item.get("result"))
    else:
        result_text = error.get("message") or "调用失败"

    args = {}
    for key, value in (item.get("args") or {}).items():
        args[key] = value if isinstance(value, (str, int, float)) else str(value)

    return {
        "id": f"tc-{uuid4().hex[:8]}",
        "name": name,
        "label": TOOL_LABELS.get(name, name),
        "status": "success" if ok else "failed",
        "args": args,
        "result": result_text,
        "durationMs": item.get("duration_ms"),
    }
