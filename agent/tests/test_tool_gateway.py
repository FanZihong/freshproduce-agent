"""工具网关单元测试：参数校验、重试、降级。"""

from __future__ import annotations

import pytest

from agent.tools import gateway
from agent.tools.degradation import ASK_AGAIN, HANDOFF
from agent.tools.gateway import ToolSpec, call_tool
from agent.tools.retry import ToolExecutionError, run_with_retry
from agent.tools.schemas import CreateTicketInput


def test_invalid_order_id_is_rejected():
    result = call_tool("query_order", order_id="abc")
    assert result.ok is False
    assert result.kind == "validation"
    assert result.action == ASK_AGAIN
    assert "订单号" in (result.reply or "")


def test_unknown_order_asks_again():
    result = call_tool("query_order", order_id="9999999999999")
    assert result.ok is False
    assert result.kind == "not_found"
    assert result.action == ASK_AGAIN


def test_seeded_order_is_returned():
    result = call_tool("query_order", order_id="2026093000123")
    assert result.ok is True
    assert result.data["product"] == "红颜草莓 500g"
    assert result.attempts == 1


def test_refund_over_limit_needs_manual_review():
    result = call_tool("query_policy", amount=500.0, ratio=1.0, category="车厘子")
    assert result.ok is False
    assert result.kind == "not_found"
    assert result.action == HANDOFF
    assert "人工" in (result.reply or "")


def test_retry_succeeds_on_second_attempt():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("第一次失败")
        return "ok"

    outcome = run_with_retry(flaky, attempts=3, base_delay=0.01)
    assert outcome.value == "ok"
    assert outcome.attempts == 2


def test_retry_exhausted_raises_with_attempts():
    def always_fail():
        raise RuntimeError("一直失败")

    with pytest.raises(ToolExecutionError) as excinfo:
        run_with_retry(always_fail, attempts=3, base_delay=0.01)
    assert excinfo.value.attempts == 3


def test_create_ticket_failure_degrades_to_handoff(monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("工单服务不可用")

    monkeypatch.setitem(
        gateway.TOOL_REGISTRY,
        "create_ticket",
        # 显式声明"重试安全"才会重试：这条用例要验证的就是重试 2 次后降级
        ToolSpec(CreateTicketInput, broken, attempts=2, base_delay=0.01, retry_safe=True),
    )

    result = call_tool(
        "create_ticket",
        session_id="s-test",
        order_id="2026093000123",
        reason="坏果理赔",
        evidence="照片",
        refund_amount=12.9,
    )
    assert result.ok is False
    assert result.kind == "error"
    assert result.action == HANDOFF
    assert result.attempts == 2
    assert "工单系统" in (result.reply or "")
