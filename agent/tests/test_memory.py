"""记忆层测试：压缩、脱敏、TTL 过期、闲置重置、跨实例恢复。"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.graph.workflow import build_graph
from agent.memory.long_term import load_summary, mask_pii, save_summary
from agent.memory.short_term import (
    apply_idle_reset,
    compress_messages,
    reset_for_test,
    set_last_active,
    should_reset,
    touch,
)


def _messages(count: int):
    result = []
    for index in range(count):
        if index % 2 == 0:
            result.append(HumanMessage(content=f"用户第 {index} 句", id=f"h-{index}"))
        else:
            result.append(AIMessage(content=f"Agent 第 {index} 句", id=f"a-{index}"))
    return result


def test_compress_keeps_recent_and_writes_summary():
    update = compress_messages({"messages": _messages(10), "summary": ""})
    assert update["summary"]
    assert len(update["messages"]) == 6  # 10 - 4(保留)
    assert all(message.type == "remove" for message in update["messages"])


def test_compress_skips_short_history():
    assert compress_messages({"messages": _messages(4), "summary": ""}) == {}


def test_mask_pii_masks_phone_and_address():
    masked = mask_pii("联系电话 13812345678，地址是朝阳区某小区 3 号楼")
    assert "138****5678" in masked
    assert "13812345678" not in masked
    assert "[地址已脱敏]" in masked


def test_long_term_memory_ttl_expiry():
    save_summary("t-ttl", "摘要内容")
    assert load_summary("t-ttl") == "摘要内容"

    save_summary("t-expired", "旧摘要", ttl_days=-1)
    assert load_summary("t-expired") is None


def test_idle_timeout_resets_slots():
    session_id = "t-idle"
    touch(session_id)
    set_last_active(session_id, datetime.now() - timedelta(minutes=45))

    assert should_reset(session_id) is True

    state = {"session_id": session_id, "slots": {"order_id": "2026093000123"}, "summary": "旧摘要"}
    update = apply_idle_reset(state)
    assert update["slots"] == {}
    assert update["active_flow"] == ""
    assert "summary" not in update  # 长期摘要不受短期重置影响

    reset_for_test(session_id)


def test_checkpoint_persists_across_graph_instances(tmp_path):
    db_path = tmp_path / "checkpoints.db"
    config = {"configurable": {"thread_id": "t-persist"}}

    conn1 = sqlite3.connect(str(db_path), check_same_thread=False)
    graph1 = build_graph(SqliteSaver(conn1))
    graph1.invoke({"user_input": "草莓坏了一半，怎么赔"}, config=config)
    graph1.invoke({"user_input": "2026093000123"}, config=config)
    assert graph1.get_state(config).values["slots"]["order_id"] == "2026093000123"

    # 模拟进程重启：换一个连接和新的图实例，从磁盘恢复
    conn2 = sqlite3.connect(str(db_path), check_same_thread=False)
    graph2 = build_graph(SqliteSaver(conn2))
    restored = graph2.get_state(config).values
    assert restored["slots"]["order_id"] == "2026093000123"
    assert restored["pending_slot"] == "damage_ratio"
