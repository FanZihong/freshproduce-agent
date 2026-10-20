"""记忆能力冒烟：消息压缩、TTL 过期、进程重启后恢复、闲置超时。

用法（在 agent/ 目录下执行）：
    python scripts/smoke_memory.py                 # 四段演示
    python scripts/smoke_memory.py --phase write   # 内部使用（重启演示）
    python scripts/smoke_memory.py --phase read
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import init_db, seed_orders  # noqa: E402
from agent.graph.workflow import build_graph, get_graph  # noqa: E402
from agent.memory.long_term import load_summary, purge_expired, save_summary  # noqa: E402
from agent.memory.short_term import (  # noqa: E402
    apply_idle_reset,
    set_last_active,
    should_reset,
    touch,
)

TURNS = [
    "我的订单到哪了",
    "2026093000123",
    "草莓坏了一半，怎么赔",
    "2026093000123",
    "大概一半",
    "照片传不上来",
]

RESTART_TURNS = ["草莓坏了一半，怎么赔", "2026093000123", "大概一半"]
RESTART_THREAD = "smoke-restart"


def section(title: str) -> None:
    print()
    print(f"---------- {title} ----------")


def demo_compress() -> None:
    section("1. 长对话压缩（超过 8 条折叠，保留最近 4 条）")
    graph = build_graph()  # 内存 saver，避免污染演示数据
    config = {"configurable": {"thread_id": "smoke-compress"}}

    for index, turn in enumerate(TURNS, start=1):
        graph.invoke({"user_input": turn}, config=config)
        state = graph.get_state(config).values
        messages = state.get("messages") or []
        has_summary = "有" if state.get("summary") else "无"
        print(f"  第 {index} 轮：{turn} -> 消息数={len(messages)}，summary={has_summary}")

    state = graph.get_state(config).values
    summary = (state.get("summary") or "").replace("\n", " ")
    print(f"\n  最终消息数：{len(state.get('messages') or [])}")
    print(f"  summary 内容：{summary[:120]}")


def demo_expiry() -> None:
    section("2. 长期记忆 TTL（7 天）与脱敏")
    save_summary("smoke-ttl", "订单 2026093000123 坏果理赔完成，联系电话 13812345678")
    print(f"  正常写入后读取：{load_summary('smoke-ttl')}")

    save_summary("smoke-expired", "过期测试：赔付 12.90 元", ttl_days=-1)
    print(f"  已过期条目读取：{load_summary('smoke-expired')}")
    save_summary("smoke-expired-2", "另一条过期测试", ttl_days=-1)
    print(f"  清理过期条目：{purge_expired()} 条")


def demo_restart() -> None:
    section("3. 进程重启后恢复（SqliteSaver + data/checkpoints.db）")
    script = str(Path(__file__).resolve())
    for phase in ("write", "read"):
        proc = subprocess.run(
            [sys.executable, script, "--phase", phase],
            capture_output=True,
            text=True,
        )
        output = (proc.stdout or proc.stderr).strip()
        print(f"  {output}")


def demo_idle() -> None:
    section("4. 会话闲置超时（超过 30 分钟重置槽位）")
    session_id = "smoke-idle"
    touch(session_id)
    set_last_active(session_id, datetime.now() - timedelta(minutes=45))
    print(f"  闲置 45 分钟后 should_reset：{should_reset(session_id)}")
    state = {"session_id": session_id, "slots": {"order_id": "2026093000123"}}
    print(f"  重置字段：{apply_idle_reset(state)}")


def phase_write() -> None:
    init_db()
    seed_orders()
    graph = get_graph(persistent=True)
    config = {"configurable": {"thread_id": RESTART_THREAD}}
    for turn in RESTART_TURNS:
        graph.invoke({"user_input": turn}, config=config)
    state = graph.get_state(config).values
    print(f"phase write：槽位 = {state.get('slots')}")


def phase_read() -> None:
    init_db()
    graph = get_graph(persistent=True)
    config = {"configurable": {"thread_id": RESTART_THREAD}}
    state = graph.get_state(config).values
    print(f"phase read（新进程）：槽位 = {state.get('slots')}，待补槽位 = {state.get('pending_slot')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="记忆能力冒烟")
    parser.add_argument("--phase", choices=["write", "read"], default=None)
    args = parser.parse_args()

    if args.phase == "write":
        phase_write()
        return
    if args.phase == "read":
        phase_read()
        return

    init_db()
    seed_orders()
    demo_compress()
    demo_expiry()
    demo_restart()
    demo_idle()


if __name__ == "__main__":
    main()
