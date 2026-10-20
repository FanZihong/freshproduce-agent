"""Trace 查询接口（内存暂存）。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import TraceSummaryOut
from app.services import trace_service

router = APIRouter(tags=["traces"])


@router.get("/traces", response_model=list[TraceSummaryOut])
def list_traces() -> list[dict]:
    return trace_service.list_summaries()


@router.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> list[dict]:
    record = trace_service.get(trace_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Trace 不存在")
    return record["spans"]
