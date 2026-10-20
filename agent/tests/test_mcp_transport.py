"""MCP 传输测试：列工具、调工具、Agent 全程走 MCP 完成理赔。"""

from __future__ import annotations

import pytest

pytest.importorskip("fastmcp")

from agent.graph.workflow import build_graph  # noqa: E402
from agent.tools.degradation import ASK_AGAIN  # noqa: E402
from agent.tools.gateway import call_tool, set_transport  # noqa: E402
from agent.tools.mcp_client import call_mcp_tool, list_tools  # noqa: E402

CLAIM_TURNS = ["草莓坏了一半，怎么赔", "2026093000123", "大概一半", "照片传不上来"]


def test_mcp_server_exposes_four_tools():
    names = {tool["name"] for tool in list_tools()}
    assert {"query_order", "query_logistics", "query_policy", "create_ticket"} <= names


def test_call_query_order_via_mcp():
    data = call_mcp_tool("query_order", {"order_id": "2026093000123"})
    assert isinstance(data, dict)
    assert data["product"] == "红颜草莓 500g"


def test_mcp_mode_unknown_order_degrades_to_ask_again():
    set_transport("mcp")
    try:
        result = call_tool("query_order", order_id="9999999999999")
        assert result.ok is False
        assert result.kind == "not_found"
        assert result.action == ASK_AGAIN
        assert result.transport == "mcp"
    finally:
        set_transport("direct")


def test_claim_flow_uses_mcp_transport():
    set_transport("mcp")
    try:
        graph = build_graph()
        config = {"configurable": {"thread_id": "t-mcp"}}
        for turn in CLAIM_TURNS:
            graph.invoke({"user_input": turn}, config=config)

        state = graph.get_state(config).values
        assert "工单号" in state["reply"]
        assert state["tool_results"]
        assert all(item["transport"] == "mcp" for item in state["tool_results"])
    finally:
        set_transport("direct")
