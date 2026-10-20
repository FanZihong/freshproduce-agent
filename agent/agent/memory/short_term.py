"""短期记忆：消息压缩 + 会话闲置超时。

压缩默认用规则折叠（每条消息压一行写进 summary，并用 RemoveMessage 真正删掉旧消息）；
`MEMORY_SUMMARY=llm` 时改用 LLM 摘要（见 agent.memory.summarizer），
两种模式共用同一套"压缩阈值 + 保留最近几条"的策略。
"""

from __future__ import annotations

from datetime import datetime

from langchain_core.messages import RemoveMessage

from agent.memory.store import get_store, warn
from agent.memory.summarizer import SUMMARY_LIMIT, summarize_dialogue, summarize_messages

COMPRESS_THRESHOLD = 8  # 消息条数超过它就压缩
KEEP_RECENT = 4  # 压缩后保留最近几条
IDLE_TIMEOUT_MINUTES = 30  # 会话闲置多久后重置槽位

def compress_messages(state) -> dict:
    """消息超过阈值时折叠旧消息：写 summary + 返回 RemoveMessage 列表。"""
    messages = list(state.get("messages") or [])
    if len(messages) <= COMPRESS_THRESHOLD:
        return {}

    old = messages[:-KEEP_RECENT]
    previous = (state.get("summary") or "").strip()
    result = summarize_dialogue(old, previous, limit=SUMMARY_LIMIT)
    if not result.text:
        return {}

    if result.replaces:
        # LLM 摘要已经把历史融合进来，直接替换，避免摘要无限增长
        summary = result.text
    else:
        summary = f"{previous}\n{result.text}".strip() if previous else result.text
    return {
        "summary": summary,
        "messages": [RemoveMessage(id=message.id) for message in old if getattr(message, "id", None)],
    }


def touch(session_id: str) -> None:
    """记录会话活跃时间。

    TTL 给活跃阈值的两倍：它只是让 Redis 不堆积死键，
    真正的"该不该重置"仍由 should_reset 按时间值判定，两种后端行为一致。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        get_store().set_activity(
            session_id, now, ttl_seconds=2 * IDLE_TIMEOUT_MINUTES * 60
        )
    except Exception as exc:  # noqa: BLE001 - 记不活跃时间失败不该影响对话
        warn(exc, action="记录会话活跃时间")


def set_last_active(session_id: str, when: datetime) -> None:
    """测试与演示用：把活跃时间改到指定时刻。"""
    try:
        get_store().set_activity(session_id, when.strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as exc:  # noqa: BLE001
        warn(exc, action="设置会话活跃时间")


def idle_minutes(session_id: str) -> float | None:
    """返回会话已闲置的分钟数；没有记录（或记忆后端不可用）时返回 None。

    返回 None 的语义是"判断不了，就不重置槽位"——宁可保留用户已填的槽位，
    也不要在存储抖动的瞬间把会话清空。
    """
    try:
        last_active = get_store().get_activity(session_id)
    except Exception as exc:  # noqa: BLE001
        warn(exc, action="读取会话活跃时间")
        return None
    if last_active is None:
        return None
    parsed = datetime.strptime(last_active, "%Y-%m-%d %H:%M:%S")
    return (datetime.now() - parsed).total_seconds() / 60


def should_reset(session_id: str, timeout_minutes: int = IDLE_TIMEOUT_MINUTES) -> bool:
    """闲置超过阈值就认为应当重置短期状态。"""
    idle = idle_minutes(session_id)
    return idle is not None and idle > timeout_minutes


def apply_idle_reset(state, timeout_minutes: int = IDLE_TIMEOUT_MINUTES) -> dict:
    """闲置超时的会话：清空槽位与流程状态，但保留长期 summary。"""
    session_id = state.get("session_id")
    if not session_id or not should_reset(session_id, timeout_minutes):
        return {}
    return {
        "slots": {},
        "active_flow": "",
        "pending_slot": "",
        "loop_count": 0,
    }


def reset_for_test(session_id: str) -> None:
    """测试辅助：删除会话活跃记录。"""
    try:
        get_store().clear_activity(session_id)
    except Exception as exc:  # noqa: BLE001
        warn(exc, action="清理会话活跃记录")
