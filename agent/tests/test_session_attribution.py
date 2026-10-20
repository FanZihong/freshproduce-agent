"""会话归属回归：CLI/后端只传 user_input 时，session_id 必须从 checkpoint 的 thread_id 推出来。

背景：这里踩过一个真坑——所有入口都不传 session_id，于是工单和长期摘要全记到 "unknown"，
25 张工单混在一个 bucket 里。现在由 intent_node 统一兜底。
"""

from __future__ import annotations

from agent.graph import nodes as nodes_module
from agent.graph.nodes import handoff_node, intent_node


def make_state(**overrides) -> dict:
    state = {
        "user_input": "草莓坏了一半",
        "slots": {},
        "tool_results": [],
        "handoff": False,
        "active_flow": "",
        "pending_slot": "",
        "loop_count": 0,
        "clarify_rounds": 0,
    }
    state.update(overrides)
    return state


def test_intent_node_uses_thread_id_when_state_missing(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(nodes_module, "load_summary", lambda session_id: seen.append(session_id) or None)

    update = intent_node(make_state(), {"configurable": {"thread_id": "t-42"}})
    assert update["session_id"] == "t-42"          # 从 thread_id 兜底推导
    assert seen == ["t-42"]                        # 长期记忆也按它去读


def test_intent_node_keeps_explicit_session_id(monkeypatch):
    monkeypatch.setattr(nodes_module, "load_summary", lambda session_id: None)
    update = intent_node(
        make_state(session_id="s-explicit"), {"configurable": {"thread_id": "t-1"}}
    )
    assert update["session_id"] == "s-explicit"    # 显式传的优先


def test_intent_node_without_config_yields_empty_session():
    update = intent_node(make_state())
    assert update["session_id"] == ""              # 没有会话上下文时不编造


def test_handoff_skips_long_term_memory_without_session(monkeypatch):
    saved: list[tuple] = []
    monkeypatch.setattr(
        nodes_module, "save_summary", lambda session_id, summary: saved.append((session_id, summary))
    )

    handoff_node(make_state(handoff_reason="用户要求投诉"))
    assert saved == []                             # 不写 "unknown"，避免串用户

    handoff_node(make_state(handoff_reason="用户要求投诉", session_id="s-9"))
    assert len(saved) == 1 and saved[0][0] == "s-9"
