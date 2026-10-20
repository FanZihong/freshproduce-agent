"""会话级并发的测试：同会话串行、不同会话并行、等待超时、注册表不泄漏。"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from agent.session_lock import (
    SessionBusy,
    active_sessions,
    async_session_lock,
    reset_locks,
    session_lock,
)


def test_same_session_is_serialized():
    reset_locks()
    events: list[str] = []
    inside = {"now": 0, "max": 0}
    guard = threading.Lock()

    def worker() -> None:
        with session_lock("s-1"):
            with guard:
                inside["now"] += 1
                inside["max"] = max(inside["max"], inside["now"])
            events.append("start")
            time.sleep(0.05)
            events.append("end")
            with guard:
                inside["now"] -= 1

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert inside["max"] == 1                       # 同一时刻只有一轮在跑
    assert events == ["start", "end", "start", "end"]  # 轮次没有交错
    assert active_sessions() == []                  # 用完就摘掉，不泄漏


def test_different_sessions_run_concurrently():
    reset_locks()
    barrier = threading.Barrier(2, timeout=3)

    def worker(session_id: str) -> None:
        with session_lock(session_id):
            barrier.wait()      # 两个不同会话必须能同时进来，否则这里会超时

    threads = [threading.Thread(target=worker, args=(sid,)) for sid in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not any(thread.is_alive() for thread in threads)
    assert active_sessions() == []


def test_waiting_times_out_with_session_busy():
    reset_locks()
    holding = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with session_lock("busy"):
            holding.set()
            release.wait(3)

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    try:
        assert holding.wait(2)
        with pytest.raises(SessionBusy) as excinfo:
            with session_lock("busy", timeout=0.05):
                pass
        assert "正在处理上一条消息" in str(excinfo.value)
    finally:
        release.set()
        thread.join(timeout=3)
    assert active_sessions() == []


def test_async_lock_serializes_same_session():
    reset_locks()
    order: list[str] = []

    async def worker(tag: str) -> None:
        async with async_session_lock("s-async"):
            order.append(f"{tag}-start")
            await asyncio.sleep(0.05)
            order.append(f"{tag}-end")

    async def main() -> None:
        await asyncio.gather(worker("a"), worker("b"))

    asyncio.run(main())
    assert order in (
        ["a-start", "a-end", "b-start", "b-end"],
        ["b-start", "b-end", "a-start", "a-end"],
    )
    assert active_sessions() == []


def test_async_lock_times_out_and_shares_sync_lock():
    """同步与异步共用同一把锁：同步占着时，异步等待超时要抛 SessionBusy。"""
    reset_locks()
    holding = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with session_lock("shared"):
            holding.set()
            release.wait(3)

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    try:
        assert holding.wait(2)

        async def try_async() -> None:
            async with async_session_lock("shared", timeout=0.05):
                pass

        with pytest.raises(SessionBusy):
            asyncio.run(try_async())
    finally:
        release.set()
        thread.join(timeout=3)
    assert active_sessions() == []


def test_default_timeout_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("SESSION_LOCK_TIMEOUT", "0.01")
    reset_locks()
    holding = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with session_lock("env-to"):
            holding.set()
            release.wait(3)

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    try:
        assert holding.wait(2)
        with pytest.raises(SessionBusy):
            with session_lock("env-to"):
                pass
    finally:
        release.set()
        thread.join(timeout=3)
    assert active_sessions() == []
