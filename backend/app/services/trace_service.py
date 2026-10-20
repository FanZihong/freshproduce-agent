"""Trace 暂存（内存实现，进程重启后清空；后续可换成 Langfuse）。"""

from __future__ import annotations

MAX_TRACES = 200

_TRACES: dict[str, dict] = {}
_ORDER: list[str] = []


def record(summary: dict, spans: list[dict]) -> None:
    trace_id = summary["id"]
    _TRACES[trace_id] = {"summary": summary, "spans": spans}
    _ORDER.append(trace_id)
    while len(_ORDER) > MAX_TRACES:
        expired = _ORDER.pop(0)
        _TRACES.pop(expired, None)


def list_summaries() -> list[dict]:
    return [_TRACES[trace_id]["summary"] for trace_id in reversed(_ORDER) if trace_id in _TRACES]


def get(trace_id: str) -> dict | None:
    return _TRACES.get(trace_id)
