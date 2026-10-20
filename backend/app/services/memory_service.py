"""事实记忆的只读视图：给 `GET /api/memory` 用。

路由只做参数校验，老化/过期口径全部复用 agent 层（`agent.memory.facts`），
避免后端和 Agent 各算一套 age/decay，最后对不上。
"""

from __future__ import annotations

from agent.config import get_float
from agent.memory import facts as facts_store


def _to_api(item: dict) -> dict:
    status = facts_store.describe_fact(item)
    return {
        "id": item["id"],
        "sessionId": item["session_id"],
        "fact": item["fact"],
        "category": item.get("category") or "general",
        "entities": item.get("entities") or [],
        "importance": float(item.get("importance") or 0.0),
        "source": item.get("source") or "",
        "createdAt": item.get("created_at") or "",
        "updatedAt": item.get("updated_at") or "",
        "lastAccessedAt": item.get("last_accessed_at"),
        "ageDays": status["age_days"],
        "decay": status["decay"],
        "expiresAt": status["expires_at"],
        "expired": status["expired"],
    }


def list_memory(
    session_id: str | None = None,
    *,
    limit: int = 50,
    include_expired: bool = False,
) -> dict:
    """按会话（或全局）列出事实；默认隐藏已过期条目。"""
    size = max(1, min(int(limit), 200))
    if session_id:
        items = facts_store.list_facts(
            session_id, limit=size, include_expired=include_expired
        )
    else:
        items = facts_store.list_all_facts(limit=size, include_expired=include_expired)
    return {
        "sessionId": session_id,
        "ttlDays": facts_store.resolve_ttl_days(),
        "decayHalfLifeDays": get_float("MEMORY_DECAY_HALF_LIFE_DAYS", 14.0),
        "decayFloor": get_float("MEMORY_DECAY_FLOOR", 0.5),
        "total": len(items),
        "items": [_to_api(item) for item in items],
    }
