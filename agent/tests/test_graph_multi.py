"""第三段验收：多意图路由 / 拒答澄清 / 域外兜底接进图，且原有链路零回归。

覆盖三件事：
1) `route_after_intent` 的判定顺序（延续流程 > 域外 > 拒答 > 并行 > 单分支）；
2) `intent_node` / `multi_node` 写进状态的新字段；
3) 环境变量开关（策略、决策层、adapter 缺失回落）不改变默认行为。
"""

from __future__ import annotations

import pytest

from agent.graph import nodes as nodes_module
from agent.graph import router as router_module
from agent.graph.nodes import intent_node, multi_node
from agent.graph.router import route_after_intent
from agent.intent.classifier import IntentResult, _classifiers, _warned
from agent.intent.decision import DecisionConfig


def make_state(**overrides) -> dict:
    base = {
        "session_id": "test-session",
        "user_input": "怎么开发票",
        "intent": "invoice",
        "confidence": 0.95,
        "slots": {},
        "tool_results": [],
        "handoff": False,
        "active_flow": "",
        "pending_slot": "",
        "loop_count": 0,
        "clarify_rounds": 0,
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _reset_caches():
    """决策配置与分类器都是进程内单例，测试之间要清干净。"""
    nodes_module._reset_decision_cache()
    _classifiers.clear()
    _warned.clear()
    yield
    nodes_module._reset_decision_cache()
    _classifiers.clear()
    _warned.clear()


# ---------------- 路由：单分支行为必须与加功能之前一致 ----------------


@pytest.mark.parametrize(
    "intent,expected",
    [
        ("query_order", "order"),
        ("urge_logistics", "order"),
        ("bad_fruit_claim", "claim"),
        ("short_weight", "claim"),
        ("refund_consult", "faq"),
        ("quality_qa", "faq"),
        ("complaint", "handoff"),
        ("human_handoff", "handoff"),
        ("chitchat", "fallback"),
    ],
)
def test_single_intent_routing_unchanged(intent, expected):
    assert route_after_intent(make_state(intent=intent)) == expected


def test_dispute_still_goes_to_handoff():
    state = make_state(intent="bad_fruit_claim", user_input="草莓全烂了，我要求全赔")
    assert route_after_intent(state) == "handoff"
    assert router_module.detect_dispute(state["user_input"])


# ---------------- 域外 / 拒答 / 并行 三条新路径 ----------------


def test_out_of_domain_routes_to_fallback():
    state = make_state(intent="chitchat", out_of_domain=True)
    assert route_after_intent(state) == "fallback"


def test_clarify_and_handoff_decisions_route_correctly():
    assert route_after_intent(make_state(decision="clarify")) == "clarify"
    assert route_after_intent(make_state(decision="handoff")) == "handoff"
    assert route_after_intent(make_state(decision="accept")) == "faq"


def test_parallel_strategy_with_two_branches_goes_to_multi():
    state = make_state(
        intent="bad_fruit_claim",
        branches=["claim", "order"],
        multi_strategy="parallel",
    )
    assert route_after_intent(state) == "multi"


def test_priority_strategy_keeps_single_branch():
    state = make_state(
        intent="bad_fruit_claim",
        branches=["claim"],
        multi_strategy="priority",
        intent_secondary="urge_logistics",
    )
    assert route_after_intent(state) == "claim"


def test_active_flow_wins_over_ood_and_decision():
    """补槽位时用户可能只回一个订单号：不能被判成域外，也不能打断流程。"""
    state = make_state(
        intent="chitchat",
        user_input="2026093000123",
        active_flow="bad_fruit_claim",
        out_of_domain=True,
        decision="clarify",
        slots={},
    )
    assert route_after_intent(state) == "claim"


# ---------------- intent_node 写入的新字段 ----------------


def test_intent_node_records_multi_intent_plan(monkeypatch):
    monkeypatch.setattr(
        nodes_module,
        "classify",
        lambda text, context=None: IntentResult(
            "bad_fruit_claim", 0.93, "双意图", ("bad_fruit_claim", "urge_logistics")
        ),
    )
    update = intent_node(make_state(user_input="草莓坏了而且还没发货"))

    assert update["intent"] == "bad_fruit_claim"        # 主意图按 routing_priority 重排
    assert update["intent_secondary"] == "urge_logistics"
    assert update["branches"] == ["claim"]              # 默认 priority 只走一条
    assert update["multi_strategy"] == "priority"
    assert update["intent_labels"] == ("bad_fruit_claim", "urge_logistics")
    assert update["decision"] == "accept"               # 高置信度直接放行


def test_intent_node_uses_parallel_strategy_from_env(monkeypatch):
    monkeypatch.setenv("INTENT_MULTI_STRATEGY", "parallel")
    monkeypatch.setattr(
        nodes_module,
        "classify",
        lambda text, context=None: IntentResult(
            "query_order", 0.9, "", ("query_order", "invoice")
        ),
    )
    update = intent_node(make_state(user_input="我的订单到哪了，另外发票怎么开"))
    assert update["multi_strategy"] == "parallel"
    assert update["branches"] == ["order", "faq"]


def test_intent_node_clarifies_on_low_confidence(monkeypatch):
    # 决策层默认只在"校准目标 == 当前实际分类器"时启用；这里直接注入配置，
    # 专门验证 intent_node 的决策接线（门控本身另有测试）
    monkeypatch.setattr(
        nodes_module,
        "_decision_config",
        lambda: DecisionConfig(temperature=1.0, accept_threshold=0.6, handoff_threshold=0.35),
    )
    monkeypatch.setattr(
        nodes_module,
        "classify",
        lambda text, context=None: IntentResult("quality_qa", 0.05, "拿不准", ("quality_qa",)),
    )
    update = intent_node(make_state(user_input="那个东西"))
    assert update["decision"] == "clarify"
    assert update["clarify_rounds"] == 1
    assert "没完全明白" in update["reply"]


def test_intent_node_hands_off_after_two_clarifications(monkeypatch):
    monkeypatch.setattr(
        nodes_module,
        "_decision_config",
        lambda: DecisionConfig(temperature=1.0, accept_threshold=0.6, handoff_threshold=0.35),
    )
    monkeypatch.setattr(
        nodes_module,
        "classify",
        lambda text, context=None: IntentResult("quality_qa", 0.05, "拿不准", ("quality_qa",)),
    )
    update = intent_node(make_state(user_input="那个东西", clarify_rounds=2))
    assert update["decision"] == "handoff"


def test_decision_layer_can_be_disabled(monkeypatch):
    """关掉决策层等价于"一律放行"，链路行为回到加这个功能之前。"""
    monkeypatch.setenv("INTENT_DECISION_ENABLED", "false")
    monkeypatch.setattr(
        nodes_module,
        "classify",
        lambda text, context=None: IntentResult("quality_qa", 0.02, "", ("quality_qa",)),
    )
    update = intent_node(make_state(user_input="那个东西"))
    assert update["decision"] == "accept"
    assert "reply" not in update


def test_decision_layer_off_for_rule_backend(monkeypatch):
    """回归保护：规则版的置信度没被校准过，决策层必须自动关闭，否则澄清会大面积误触发。

    实测规则版单关键词置信度 0.60，经温度缩放会掉到 0.54，低于本地模型的放行阈值 0.5888。
    """
    nodes_module._reset_decision_cache()
    monkeypatch.setattr(
        nodes_module,
        "classify",
        lambda text, context=None: IntentResult("query_order", 0.60, "", ("query_order",)),
    )
    update = intent_node(make_state(user_input="帮我查一下订单 2026093000123"))
    assert update["decision"] == "accept"     # 规则版下不启用拒答


def test_intent_node_marks_out_of_domain(monkeypatch):
    monkeypatch.setattr(
        nodes_module,
        "classify",
        lambda text, context=None: IntentResult("chitchat", 0.9, "", ("chitchat",)),
    )
    update = intent_node(make_state(user_input="讲个笑话吧"))
    assert update["out_of_domain"] is True
    assert update["ood_reason"]


def test_intent_node_skips_ood_while_filling_slots(monkeypatch):
    monkeypatch.setattr(
        nodes_module,
        "classify",
        lambda text, context=None: IntentResult("chitchat", 0.9, "", ("chitchat",)),
    )
    update = intent_node(make_state(user_input="2026093000123", active_flow="bad_fruit_claim"))
    assert update["out_of_domain"] is False
    assert update["decision"] == "accept"


# ---------------- multi_node：两条分支都执行并合并回复 ----------------


def test_multi_node_merges_two_branch_replies(monkeypatch):
    calls: list[str] = []

    def fake_branch(name: str, reply: str):
        def node(state):
            calls.append(name)
            return {"reply": reply, "active_flow": "", "tool_results": [{"name": name}]}

        return node

    monkeypatch.setattr(
        nodes_module,
        "MULTI_BRANCH_NODES",
        {
            "claim": fake_branch("claim", "已提交理赔，应赔 12.90 元。"),
            "order": fake_branch("order", "物流显示已到达网点。"),
        },
    )
    update = multi_node(make_state(branches=["claim", "order"], multi_strategy="parallel"))

    assert calls == ["claim", "order"]           # 按主分支在前执行
    assert update["reply"].startswith("已提交理赔")
    assert "另外" in update["reply"] and "物流显示已到达网点" in update["reply"]
    assert update["branches"] == ["claim", "order"]


def test_multi_node_keeps_pending_slot_from_any_branch(monkeypatch):
    monkeypatch.setattr(
        nodes_module,
        "MULTI_BRANCH_NODES",
        {
            "claim": lambda state: {
                "reply": "请提供订单号。",
                "active_flow": "bad_fruit_claim",
                "pending_slot": "order_id",
            },
            "faq": lambda state: {"reply": "车厘子冷藏可放 5 天。", "active_flow": "", "pending_slot": ""},
        },
    )
    update = multi_node(make_state(branches=["claim", "faq"]))
    # 有分支在等槽位时必须保住流程状态，否则用户下一句的订单号会被丢掉
    assert update["active_flow"] == "bad_fruit_claim"
    assert update["pending_slot"] == "order_id"


def test_multi_node_degrades_to_single_branch(monkeypatch):
    monkeypatch.setattr(
        nodes_module, "MULTI_BRANCH_NODES", {"faq": lambda state: {"reply": "单分支回复"}}
    )
    assert multi_node(make_state(branches=["faq"]))["reply"] == "单分支回复"
    # 没有可用分支时走兜底
    assert "售后服务的范围" in multi_node(make_state(branches=["不存在的分支"]))["reply"]


# ---------------- 模型缺失自动回落 + 整图跑通 ----------------


def test_local_backend_falls_back_to_rule_when_adapter_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("INTENT_BACKEND", "local")
    monkeypatch.setenv("INTENT_LOCAL_DIR", str(tmp_path / "不存在的adapter"))
    from agent.intent import classifier as classifier_module

    assert classifier_module.get_classifier().name == "rule-based-v1"
    # 图继续可用：分类入口照常产出标签
    assert classifier_module.classify("我的订单到哪了").all_labels


def test_graph_runs_one_turn_with_rule_backend(monkeypatch):
    """整图跑一轮：查单链路端到端可用（不依赖本地模型）。"""
    monkeypatch.setenv("INTENT_BACKEND", "rule")
    from agent.graph.workflow import build_graph

    graph = build_graph()
    result = graph.invoke(
        {"user_input": "帮我查一下订单 2026093000123", "session_id": "graph-test"},
        config={"configurable": {"thread_id": "graph-test"}},
    )
    assert "2026093000123" in (result.get("reply") or "")
    assert result.get("intent")


def test_graph_out_of_domain_turn_goes_to_fallback(monkeypatch):
    monkeypatch.setenv("INTENT_BACKEND", "rule")
    from agent.graph.workflow import build_graph

    graph = build_graph()
    result = graph.invoke(
        {"user_input": "讲个笑话吧", "session_id": "graph-test-ood"},
        config={"configurable": {"thread_id": "graph-test-ood"}},
    )
    assert "超出了售后服务的范围" in (result.get("reply") or "")
    assert result.get("out_of_domain") is True
