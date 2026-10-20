"""对话编排：跑 Agent 图、产出 SSE 事件、落库会话消息、记录 trace。"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import AsyncIterator
from uuid import uuid4

from agent.graph.workflow import get_graph
from agent.memory.short_term import should_reset
from agent.session_lock import SessionBusy, async_session_lock

from app.config import (
    INTENT_LABELS,
    NODE_LABELS,
    STREAM_CHUNK_SIZE,
    STREAM_DELAY_SECONDS,
    TOOL_LABELS,
)
from app.services import session_service, trace_service
from app.services.formatting import intent_payload, slots_to_items, tool_call_payload

# 会话闲置超时后要重置的短期状态
IDLE_RESET_STATE = {"slots": {}, "active_flow": "", "pending_slot": "", "loop_count": 0}

NODE_SPAN_TYPE = {
    "rails_input": "workflow",
    "intent": "intent",
    "skill": "workflow",
    "order": "workflow",
    "claim": "workflow",
    "handoff": "workflow",
    "fallback": "workflow",
    "respond": "llm",
    "compress": "workflow",
    "rails_execution": "workflow",
    "rails_output": "workflow",
}


def sse(payload: dict) -> str:
    """把一个事件编码成 SSE 数据帧。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _run_turn(session_id: str, message: str) -> dict:
    """在工作线程里跑一轮图。

    放到线程里的原因：MCP 传输内部用 asyncio.run，不能在有事件循环的线程调用。
    """
    graph = get_graph(persistent=True)
    config = {"configurable": {"thread_id": session_id}}

    if should_reset(session_id):
        try:
            graph.update_state(config, dict(IDLE_RESET_STATE))
        except Exception:  # noqa: BLE001 - 还没有 checkpoint 时忽略
            pass

    before = graph.get_state(config).values
    baseline_tools = len(before.get("tool_results") or [])

    steps: list[dict] = []
    started = time.perf_counter()
    for chunk in graph.stream(
        {"user_input": message, "session_id": session_id},
        config=config,
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if isinstance(update, dict):
                steps.append({"node": node, "update": update, "at": time.perf_counter()})
    state = graph.get_state(config).values
    return {
        "steps": steps,
        "state": state,
        "baseline_tools": baseline_tools,
        "started": started,
        "finished": time.perf_counter(),
    }


def _chunks(text: str, size: int = STREAM_CHUNK_SIZE) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]


def _tool_span(trace_id: str, index: int, item: dict) -> dict:
    ok = bool(item.get("ok"))
    error = item.get("error") or {}
    name = item.get("name") or "unknown"
    return {
        "id": f"{trace_id}-tool-{index}",
        "name": f"tool.{name}",
        "type": "tool",
        "status": "success" if ok else "failed",
        "durationMs": int(item.get("duration_ms") or 0),
        "detail": {
            "transport": str(item.get("transport") or "direct"),
            "attempts": str(item.get("attempts") or 1),
            "result": error.get("message") if not ok else "ok",
        },
    }


def _node_span(trace_id: str, index: int, node: str, duration_ms: int, update: dict) -> dict:
    detail: dict[str, str] = {"node": node}
    if node == "intent" and update.get("intent"):
        detail["intent"] = str(update["intent"])
        detail["confidence"] = str(update.get("confidence"))
    if update.get("pending_slot"):
        detail["pending_slot"] = str(update["pending_slot"])
    if update.get("summary"):
        detail["summary"] = str(update["summary"])[:40]
    if update.get("handoff_reason"):
        detail["handoff_reason"] = str(update["handoff_reason"])
    if update.get("skill_name"):
        detail["skill"] = str(update["skill_name"])
        detail["skill_tokens"] = str(update.get("skill_tokens") or 0)
    if update.get("rails"):
        detail["rails"] = ",".join(str(item.get("rule") or "?") for item in update["rails"])
    return {
        "id": f"{trace_id}-node-{index}",
        "name": f"{node}.node",
        "type": NODE_SPAN_TYPE.get(node, "workflow"),
        "status": "success",
        "durationMs": duration_ms,
        "detail": detail,
        "children": [],
    }


async def stream_chat(session_id: str, message: str) -> AsyncIterator[str]:
    """产出 SSE 事件流：intent -> tool* -> token* -> done。"""
    session_service.ensure_session(session_id, title=message[:16])

    # 同一会话串行化：并发发两条消息时，后到的等待前一条跑完，
    # 否则两个请求会各自读同一份 checkpoint、后写的覆盖先写的（轮次乱、槽位丢）
    try:
        async with async_session_lock(session_id):
            result = await asyncio.to_thread(_run_turn, session_id, message)
    except SessionBusy:
        yield sse({"type": "token", "content": "上一条消息还在处理，请稍候再发。"})
        yield sse({"type": "done", "traceId": f"tr-{uuid4().hex[:8]}"})
        return
    steps = result["steps"]
    state = result["state"]
    trace_id = f"tr-{uuid4().hex[:8]}"

    seen_tools = result["baseline_tools"]
    intent: dict | None = None
    tool_calls: list[dict] = []
    reply = ""
    spans: list[dict] = []
    previous_at = result["started"]

    for index, step in enumerate(steps):
        node = step["node"]
        update = step["update"]
        duration_ms = int((step["at"] - previous_at) * 1000)
        previous_at = step["at"]

        node_span = _node_span(trace_id, index, node, duration_ms, update)

        if node == "intent" and update.get("intent"):
            intent = intent_payload(str(update["intent"]), float(update.get("confidence") or 0))
            yield sse({"type": "intent", "intent": intent})

        results = update.get("tool_results") or []
        if len(results) > seen_tools:
            for offset, item in enumerate(results[seen_tools:], start=1):
                payload = tool_call_payload(item)
                tool_calls.append(payload)
                node_span["children"].append(_tool_span(trace_id, len(tool_calls), item))
                yield sse({"type": "tool", "toolCall": payload})
            seen_tools = len(results)

        if update.get("reply"):
            reply = str(update["reply"])

        spans.append(node_span)

    if not reply:
        reply = "抱歉，这次没有生成有效回复，请再试一次或转人工客服。"

    for piece in _chunks(reply):
        yield sse({"type": "token", "content": piece})
        if STREAM_DELAY_SECONDS:
            await asyncio.sleep(STREAM_DELAY_SECONDS)

    yield sse({"type": "done", "traceId": trace_id})

    # ---- 落库与 trace 记录 ----
    handoff = bool(state.get("handoff"))
    slots = state.get("slots") or {}
    session_service.append_message(session_id, "user", message)
    session_service.append_message(
        session_id,
        "assistant",
        reply,
        intent=intent,
        slots=slots_to_items(slots),
        tool_calls=tool_calls,
        trace_id=trace_id,
    )
    session_service.update_session(
        session_id,
        status="handoff" if handoff else "active",
        last_message=reply,
        title=None,
    )

    total_ms = int((result["finished"] - result["started"]) * 1000)
    failure_type = str(state.get("handoff_reason") or "") or None
    summary = {
        "id": trace_id,
        "sessionId": session_id,
        "intent": (intent or {}).get("name") or "未识别",
        "status": "failed" if handoff else "success",
        "failureType": failure_type,
        "durationMs": max(total_ms, 1),
        "tokenCost": round(0.0005 + 0.000003 * (len(message) + len(reply)), 4),
        "createdAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    root_span = {
        "id": f"{trace_id}-root",
        "name": "langgraph.workflow",
        "type": "workflow",
        "status": summary["status"],
        "durationMs": summary["durationMs"],
        "detail": {
            "session_id": session_id,
            "thread_id": session_id,
            "nodes": str(len(spans)),
            "tools": str(len(tool_calls)),
        },
        "children": spans,
    }
    trace_service.record(summary, [root_span])
