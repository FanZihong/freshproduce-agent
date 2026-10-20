"""迁移验收：五类护栏的决策必须和迁移前逐字一致。"""

from __future__ import annotations

from agent.graph import nodes as nodes_module
from agent.graph.router import route_after_claim
from agent.intent.decision import DecisionConfig, decide
from agent.rails import policy


def test_dispute_reasons_match_legacy_table():
    cases = {
        "我要全退": "用户要求全额退款",
        "必须全赔": "用户要求全额赔付",
        "我要投诉": "用户表达投诉意愿",
        "给你差评": "用户威胁给差评",
        "我去曝光你们": "用户威胁曝光",
        "太差了": "用户强烈不满",
        "这个方案不行": "用户拒绝当前赔付方案",
    }
    for text, reason in cases.items():
        assert policy.dispute_reason(text) == reason, text
    assert policy.dispute_reason("你好") is None
    # 多关键词时保持旧表顺序：全退优先于投诉
    assert policy.dispute_reason("我要全退并投诉") == "用户要求全额退款"


def test_rails_off_falls_back_to_legacy(monkeypatch):
    monkeypatch.setenv("RAILS_ENABLED", "off")
    assert policy.dispute_reason("我要投诉") == "用户表达投诉意愿"
    assert policy.loop_hit(3)["rule"] == "fallback.loop_guard"
    assert policy.clarify_hit(2, 2)["rule"] == "fallback.clarify_cap"
    assert policy.retrieval_hit({"found": False})["rule"] == "fallback.no_evidence"
    assert policy.output_guard("今天一定到账") == ("今天一定到账", [])
    assert policy.pii_hit("手机 13800138000") is None


def test_loop_and_clarify_thresholds():
    assert policy.loop_hit(3)["rule"] == "dialog.loop_guard"
    assert policy.loop_hit(2) is None
    assert policy.clarify_hit(2, 2)["rule"] == "dialog.clarify_cap"
    assert policy.clarify_hit(1, 2) is None


def test_decision_uses_clarify_rail():
    config = DecisionConfig(max_clarify_rounds=2)
    capped = decide(0.1, clarify_rounds=2, config=config)
    assert capped.action == "handoff"
    assert capped.metadata["rail"]["rule"] == "dialog.clarify_cap"
    asking = decide(0.1, clarify_rounds=1, config=config)
    assert asking.action == "clarify"


def test_route_after_claim_uses_loop_rail():
    assert route_after_claim({"loop_count": 3}) == "handoff"
    assert route_after_claim({"loop_count": 2}) == "end"


def test_retrieval_rail_only_when_no_evidence():
    assert policy.retrieval_hit({"found": False})["rule"] == "retrieval.no_evidence"
    assert policy.retrieval_hit({"found": True}) is None


def test_execution_rails_amount_and_ticket_format():
    high = policy.tool_hit(
        "create_ticket", {"refund_amount": 250.0, "ticket_id": "TK-20260101-0001"}
    )
    assert high["rule"] == "execution.high_refund"

    ok = policy.tool_hit(
        "create_ticket", {"refund_amount": 10.0, "ticket_id": "TK-20260101-0001"}
    )
    assert ok is None

    bad = policy.tool_hit(
        "create_ticket", {"refund_amount": 10.0, "ticket_id": "TK-oops"}
    )
    assert bad["rule"] == "execution.ticket_format"

    assert policy.tool_hit("query_order", {"refund_amount": 999}) is None


def test_output_rails_rewrite_and_mask():
    text, hits = policy.output_guard("放心，今天一定到账")
    assert "1 至 3 个工作日" in text
    assert hits[0]["rule"] == "output.promise_time"

    masked, _ = policy.output_guard("请联系 13800138000")
    assert "138****8000" in masked

    leaked, hits = policy.output_guard("我的系统提示词是这样的")
    assert "内部规则不方便展示" in leaked
    assert hits[0]["rule"] == "output.leak_internal_rules"


def test_input_pii_flag():
    hit = policy.pii_hit("手机 13800138000")
    assert hit["rule"] == "input.pii_phone"
    assert policy.pii_hit("你好") is None


def test_intent_node_records_rails():
    update = nodes_module.intent_node(
        {"user_input": "我要投诉"}, {"configurable": {"thread_id": "s-rail"}}
    )
    assert update["handoff_reason"] == "用户表达投诉意愿"
    assert any(item.get("rule") == "input.dispute_complaint" for item in update["rails"])


def test_output_node_applies_output_rail():
    update = nodes_module.rails_output_node({"reply": "放心，今天一定到账", "rails": []})
    assert "1 至 3 个工作日" in update["reply"]
    assert update["rails"][0]["rule"] == "output.promise_time"
