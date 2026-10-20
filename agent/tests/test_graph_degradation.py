"""图级测试：工具持续失败时，工作流应降级到转人工。"""

from __future__ import annotations

from agent.graph.workflow import build_graph
from agent.tools import gateway
from agent.tools.gateway import ToolSpec
from agent.tools.schemas import CreateTicketInput

TURNS = ["草莓坏了一半，怎么赔", "2026093000123", "大概一半", "照片传不上来"]


def test_claim_flow_ends_with_handoff_when_ticket_service_down(monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("工单服务不可用")

    monkeypatch.setitem(
        gateway.TOOL_REGISTRY,
        "create_ticket",
        ToolSpec(CreateTicketInput, broken, attempts=2, base_delay=0.01, retry_safe=True),
    )

    graph = build_graph()
    config = {"configurable": {"thread_id": "test-degradation"}}
    for turn in TURNS:
        graph.invoke({"user_input": turn}, config=config)

    state = graph.get_state(config).values
    assert state["handoff"] is True
    assert "人工" in state["reply"]
    assert "create_ticket" in state["handoff_summary"]


def test_claim_flow_succeeds_with_real_tools():
    graph = build_graph()
    config = {"configurable": {"thread_id": "test-success"}}
    for turn in TURNS:
        graph.invoke({"user_input": turn}, config=config)

    state = graph.get_state(config).values
    assert state["handoff"] is False
    assert "工单号" in state["reply"]
