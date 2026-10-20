"""长期记忆：跨会话摘要（带 TTL 与脱敏）。

存储走 agent.memory.store 的 MemoryStore 协议（默认 SQLite 的 user_memory 表，
配了 REDIS_URL 则切 Redis）；读取时先检查 expires_at，过期即删除并返回 None。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from agent.memory.store import get_store, warn

MEMORY_TTL_DAYS = 7

PHONE_PATTERN = re.compile(r"1[3-9]\d{9}")
ADDRESS_KEYWORDS = ("地址", "小区", "号楼", "街道", "门牌")


def mask_pii(text: str) -> str:
    """手机号与地址做粗粒度脱敏，避免把敏感信息写进长期记忆。"""
    masked = PHONE_PATTERN.sub(
        lambda match: f"{match.group(0)[:3]}****{match.group(0)[-4:]}",
        text,
    )
    for keyword in ADDRESS_KEYWORDS:
        if keyword in masked:
            masked = re.sub(rf"[^，。；\s]*{keyword}[^，。；\s]*", "[地址已脱敏]", masked)
    return masked


def save_summary(
    session_id: str,
    summary: str,
    ttl_days: int = MEMORY_TTL_DAYS,
) -> dict:
    """写入或更新长期摘要，返回落库信息；后端不可用时降级为 `degraded=True`。"""
    now = datetime.now()
    expires_at = now + timedelta(days=ttl_days)
    masked = mask_pii(summary)
    try:
        get_store().set_summary(
            session_id, masked, expires_at.strftime("%Y-%m-%d %H:%M:%S")
        )
    except Exception as exc:  # noqa: BLE001 - 长期记忆写失败不该影响本轮对话
        warn(exc, action="写入长期摘要")
        return {"session_id": session_id, "expires_at": "", "degraded": True}
    return {"session_id": session_id, "expires_at": expires_at.strftime("%Y-%m-%d %H:%M:%S")}


def load_summary(session_id: str) -> str | None:
    """读取未过期的长期摘要；已过期或后端不可用时返回 None。"""
    try:
        return get_store().get_summary(
            session_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    except Exception as exc:  # noqa: BLE001
        warn(exc, action="读取长期摘要")
        return None


def purge_expired() -> int:
    """清理所有过期记忆，返回删除条数。"""
    try:
        return get_store().purge_summaries(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    except Exception as exc:  # noqa: BLE001
        warn(exc, action="清理过期记忆")
        return 0
