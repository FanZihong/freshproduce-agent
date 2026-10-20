"""同一会话的请求串行化：一个会话同一时刻只允许一轮在跑。

为什么需要：图的状态是按 `thread_id`（= session_id）存 checkpoint 的。同一会话并发发两条
消息时，两个请求会各自读到同一份状态、各自写回，后写的把先写的覆盖掉——用户看到的是
"轮次乱了 / 槽位丢了"。串行化是最小、最可靠的修法（真实系统也可以换成队列，
但锁的语义最直白，且不需要额外的消费者进程）。

两个约束：
1. **等待不阻塞事件循环**：FastAPI 那侧是 async，等待锁必须用 `asyncio.to_thread`；
2. **锁不能泄漏**：按会话做引用计数，最后一个使用者释放后把锁从注册表里摘掉，
   否则跑一晚上会积累几万个锁对象。

超时会抛 `SessionBusy`，由上层决定怎么回复用户（HTTP 可以回 409，SSE 回一句提示）。
默认超时 30s，可用 `SESSION_LOCK_TIMEOUT` 覆盖。
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field

from agent.config import get_float

DEFAULT_TIMEOUT_SECONDS = 30.0


class SessionBusy(RuntimeError):
    """同一会话已有请求在处理，等待超时。"""

    def __init__(self, session_id: str, timeout: float | None) -> None:
        limit = "无限" if timeout is None else f"{timeout:.1f}s"
        super().__init__(f"会话 {session_id} 正在处理上一条消息（等待 {limit} 超时）")
        self.session_id = session_id
        self.timeout = timeout


@dataclass
class _Entry:
    lock: threading.Lock = field(default_factory=threading.Lock)
    refs: int = 0


_REGISTRY: dict[str, _Entry] = {}
_REGISTRY_GUARD = threading.Lock()


def default_timeout() -> float:
    return get_float("SESSION_LOCK_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)


def _acquire_entry(session_id: str) -> threading.Lock:
    with _REGISTRY_GUARD:
        entry = _REGISTRY.get(session_id)
        if entry is None:
            entry = _Entry()
            _REGISTRY[session_id] = entry
        entry.refs += 1
        return entry.lock


def _release_entry(session_id: str) -> None:
    with _REGISTRY_GUARD:
        entry = _REGISTRY.get(session_id)
        if entry is None:
            return
        entry.refs -= 1
        if entry.refs <= 0:
            _REGISTRY.pop(session_id, None)


def active_sessions() -> list[str]:
    """当前注册表里的会话（测试与观测用）。"""
    with _REGISTRY_GUARD:
        return sorted(_REGISTRY)


def reset_locks() -> None:
    """测试用：清空注册表。"""
    with _REGISTRY_GUARD:
        _REGISTRY.clear()


@contextmanager
def session_lock(session_id: str, *, timeout: float | None = None):
    """同步上下文：拿到锁才允许跑这一轮；超时抛 SessionBusy。"""
    limit = default_timeout() if timeout is None else timeout
    lock = _acquire_entry(session_id)
    acquired = lock.acquire(timeout=-1 if limit is None else limit)
    if not acquired:
        _release_entry(session_id)
        raise SessionBusy(session_id, limit)
    try:
        yield
    finally:
        lock.release()
        _release_entry(session_id)


@asynccontextmanager
async def async_session_lock(session_id: str, *, timeout: float | None = None):
    """异步上下文：等待锁放到线程里，避免卡住事件循环；与同步版本共用同一把锁。"""
    limit = default_timeout() if timeout is None else timeout
    lock = _acquire_entry(session_id)
    acquired = False
    try:
        if limit is None:
            acquired = await asyncio.to_thread(lock.acquire)
        else:
            acquired = await asyncio.to_thread(lock.acquire, True, limit)
        if not acquired:
            raise SessionBusy(session_id, limit)
        try:
            yield
        finally:
            lock.release()
    finally:
        _release_entry(session_id)
