"""健康检查：顺带报告 Memory 后端，方便确认"到底在写哪个存储"。"""

from __future__ import annotations

import sys
from datetime import datetime

from fastapi import APIRouter

from app.config import AGENT_ROOT

router = APIRouter(tags=["health"])

if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))


@router.get("/health")
def health() -> dict:
    payload = {
        "status": "ok",
        "service": "fresh-service-agent",
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        from agent.memory.store import get_store  # noqa: PLC0415

        payload["memory"] = {"backend": get_store().backend}
    except Exception as exc:  # noqa: BLE001 - 健康检查本身不能被存储问题带挂
        payload["memory"] = {"backend": "error", "error": f"{type(exc).__name__}: {exc}"}
    return payload
