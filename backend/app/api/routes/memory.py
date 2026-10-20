"""事实记忆只读接口。"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.schemas import MemoryListOut
from app.services import memory_service

router = APIRouter(tags=["memory"])


@router.get("/memory", response_model=MemoryListOut)
def list_memory(
    sessionId: str | None = Query(default=None, description="不传则返回全局最近事实"),
    limit: int = Query(default=50, ge=1, le=200),
    includeExpired: bool = Query(default=False),
) -> dict:
    return memory_service.list_memory(
        sessionId, limit=limit, include_expired=includeExpired
    )
