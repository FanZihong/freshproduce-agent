"""对话评测脚本的纯逻辑测试：跳过判定、判据求值、工具聚合、指标汇总、跑图收集。

`run_dialogue` 用一个假图对象注入（只需实现 invoke），所以这些用例不需要 langgraph。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.eval_dialogues import (
    aggregate_tools,
    dialogue_outcome,
    evaluate_checks,
    group_stats,
    run_dialogue,
    skip_reason,
    summarize,
    write_report,
)


def make_row(**overrides) -> dict:
    row = {
        "id": "dg-99",
        "scenario": "测试用例",
        "chain": "claim",
        "category": "core",
        "turns": [{"user": "东西烂了"}],
        "expected_goal_resolved": True,
        "expected_handoff": False,
        "resolution_criteria": "给出赔付结论",
        "check": {"tools": ["create_ticket"], "reply_contains": ["工单号"]},
        "requires_env": {},
        "note": "",
    }
    row.update(overrides)
    return row


def make_facts(**overrides) -> dict:
    facts = {
        "last_reply": "已提交理赔，工单号 T-1",
        "tools": ["query_order", "create_ticket"],
        "handoff": False,
        "handoff_summary": "",
        "branches": ["claim"],
    }
    facts.update(overrides)
    return facts


# ---------------- 跳过判定 ----------------


def test_skip_reason_passes_when_env_matches():
    row = make_row(requires_env={"INTENT_BACKEND": "local"})
    assert skip_reason(row, backend="local", strategy="priority") is None
    assert skip_reason(make_row(), backend="rule", strategy="priority") is None


def test_skip_reason_reports_each_mismatch():
    row = make_row(requires_env={"INTENT_BACKEND": "local", "INTENT_MULTI_STRATEGY": "parallel"})
    reason = skip_reason(row, backend="rule", strategy="priority")
    assert reason is not None
    assert "INTENT_BACKEND=local" in reason and "INTENT_MULTI_STRATEGY=parallel" in reason
    # 只差一个时只报那一个
    partial = skip_reason(row, backend="local", strategy="priority")
    assert partial and "INTENT_BACKEND" not in partial and "INTENT_MULTI_STRATEGY" in partial


# ---------------- 判据求值 ----------------


def test_evaluate_checks_all_types():
    row = make_row(
        check={
            "tools": ["create_ticket"],
            "reply_contains": ["工单号"],
            "handoff_summary_contains": ["诉求"],
            "branches": ["claim"],
        }
    )
    facts = make_facts(handoff_summary="订单 X；诉求：坏果理赔")
    results = evaluate_checks(row, facts)
    assert results == {
        "tools": True,
        "reply_contains": True,
        "handoff_summary_contains": True,
        "branches": True,
    }


def test_evaluate_checks_reports_what_is_missing():
    row = make_row(check={"tools": ["create_ticket", "query_policy"], "reply_contains": ["12.90"]})
    facts = make_facts(last_reply="请提供订单号", tools=["query_order"])
    results = evaluate_checks(row, facts)
    assert results["tools"] is False and results["reply_contains"] is False
    assert facts["missing_tools"] == ["create_ticket", "query_policy"]
    assert facts["missing_in_reply"] == ["12.90"]


def test_evaluate_checks_reply_contains_any_needs_one_hit():
    row = make_row(check={"reply_contains_any": ["没有查到", "订单号"]})
    assert evaluate_checks(row, make_facts(last_reply="没有查到这笔订单"))["reply_contains_any"] is True
    assert evaluate_checks(row, make_facts(last_reply="稍等"))["reply_contains_any"] is False


def test_evaluate_checks_without_check_returns_empty():
    assert evaluate_checks(make_row(check={}), make_facts()) == {}


# ---------------- 单条结论 ----------------


def test_dialogue_outcome_marks_resolved_and_handoff():
    row = make_row(expected_goal_resolved=True, expected_handoff=False)
    outcome = dialogue_outcome(row, make_facts())
    assert outcome["resolved_predicted"] is True
    assert outcome["resolved_match"] is True
    assert outcome["handoff_ok"] is True


def test_dialogue_outcome_detects_false_claim_and_wrong_handoff():
    row = make_row(expected_goal_resolved=False, expected_handoff=True)
    outcome = dialogue_outcome(row, make_facts(handoff=False))
    # 判据全过 = 行为正确（这里"预期不该解决"指的是用户目标没达成，不是行为不对）
    assert outcome["resolved_predicted"] is True
    assert outcome["resolved_match"] is False
    assert outcome["handoff_ok"] is False


# ---------------- 工具聚合 ----------------


def test_aggregate_tools_counts_success_retry_and_degradation():
    tool_results = [
        {"name": "query_order", "ok": True, "attempts": 1},
        {"name": "query_logistics", "ok": True, "attempts": 3},      # 重试后成功
        {"name": "query_policy", "ok": False, "action": "handoff", "attempts": 1},
        {"name": "create_ticket", "ok": False, "action": "ask_again", "attempts": 2},
    ]
    stats = aggregate_tools(tool_results)
    assert stats["calls"] == 4
    assert stats["success"] == 2 and stats["retried_success"] == 1
    assert stats["degraded_ask_again"] == 1 and stats["degraded_handoff"] == 1
    assert stats["failed"] == 0
    assert stats["degraded"] == 2
    assert stats["success_rate"] == 0.5
    assert stats["degrade_rate"] == 0.5
    assert stats["by_tool"]["query_order"] == 1


def test_aggregate_tools_handles_empty_and_unknown_action():
    assert aggregate_tools([])["success_rate"] == 0.0
    stats = aggregate_tools([{"name": "x", "ok": False, "action": None}])
    assert stats["failed"] == 1 and stats["degrade_rate"] == 0.0


# ---------------- 指标汇总 ----------------


def make_record(**overrides) -> dict:
    record = {
        "id": "dg-99",
        "chain": "claim",
        "category": "core",
        "turns": 2,
        "resolved_expected": True,
        "resolved_predicted": True,
        "handoff_expected": False,
        "handoff_actual": False,
        "checks": {},
        "tools": {"calls": 1, "success": 1, "retried_success": 0, "degraded": 0},
    }
    record.update(overrides)
    # 派生字段跟着 expected/predicted 走，避免夹具自己造出自相矛盾的数据
    record.setdefault("resolved_match", record["resolved_expected"] == record["resolved_predicted"])
    record.setdefault("handoff_ok", record["handoff_expected"] == record["handoff_actual"])
    return record


def test_summarize_separates_should_resolve_and_should_not():
    records = [
        make_record(id="a", resolved_expected=True, resolved_predicted=True),
        make_record(id="b", resolved_expected=True, resolved_predicted=False),
        make_record(id="c", resolved_expected=False, resolved_predicted=False),
        make_record(id="d", resolved_expected=False, resolved_predicted=True),
    ]
    metrics = summarize(records)
    assert metrics["dialogues"] == 4 and metrics["turns"] == 8
    assert metrics["behavior_ok_rate"] == 0.5         # 4 条里判据全过 2 条
    assert metrics["behavior_ok_total"] == 4
    assert metrics["resolution_rate"] == 0.5          # 该解决的 2 条里解决 1 条
    assert metrics["resolution_total"] == 2
    # 不该解决的 2 条里，行为正确的只有 1 条 —— 同样是"越高越好"
    assert metrics["expected_unresolved_ok_rate"] == 0.5
    assert metrics["expected_unresolved_total"] == 2
    assert metrics["label_agreement"] == 0.5          # 数据自检，不是模型指标


def test_summarize_handoff_recall_and_false_rate():
    records = [
        make_record(id="a", handoff_expected=True, handoff_actual=True, handoff_ok=True),
        make_record(id="b", handoff_expected=True, handoff_actual=False, handoff_ok=False),
        make_record(id="c", handoff_expected=False, handoff_actual=False, handoff_ok=True),
        make_record(id="d", handoff_expected=False, handoff_actual=True, handoff_ok=False),
    ]
    metrics = summarize(records)
    assert metrics["handoff_recall"] == 0.5
    assert metrics["handoff_total"] == 2
    assert metrics["handoff_false_rate"] == 0.5
    assert metrics["handoff_false_total"] == 2
    assert metrics["handoff_accuracy"] == 0.5


def test_summarize_handles_empty_and_all_zero_denominators():
    metrics = summarize([make_record(resolved_expected=False, handoff_expected=False)])
    assert metrics["resolution_total"] == 0 and metrics["resolution_rate"] == 0.0
    assert metrics["expected_unresolved_total"] == 1
    assert metrics["handoff_total"] == 0 and metrics["handoff_recall"] == 0.0


def test_group_stats_groups_by_chain_and_category():
    records = [
        make_record(id="a", chain="order", category="core"),
        make_record(id="b", chain="order", category="noise", resolved_predicted=False),
        make_record(id="c", chain="faq", category="core"),
    ]
    by_chain = dict(group_stats(records, "chain"))
    assert by_chain["order"]["dialogues"] == 2
    assert by_chain["faq"]["dialogues"] == 1
    by_category = dict(group_stats(records, "category"))
    assert by_category["core"]["dialogues"] == 2
    assert by_category["noise"]["dialogues"] == 1


# ---------------- 跑图收集（用假图，不需要 langgraph） ----------------


class FakeGraph:
    """按轮返回预设状态；只实现 invoke，模拟 LangGraph 编译后的图。"""

    def __init__(self, states: list[dict]):
        self._states = states
        self.calls: list[dict] = []

    def invoke(self, payload, config=None):
        self.calls.append({"payload": payload, "config": config})
        return self._states[min(len(self.calls) - 1, len(self._states) - 1)]


def test_run_dialogue_collects_facts_from_all_turns():
    row = make_row(
        turns=[{"user": "东西烂了"}, {"user": "照片已上传"}],
        expected_goal_resolved=True,
    )
    graph = FakeGraph(
        [
            {"reply": "请提供订单号", "intent": "bad_fruit_claim", "branches": ["claim"],
             "tool_results": []},
            {"reply": "已赔付 12.90 元，工单号 T-1", "intent": "bad_fruit_claim",
             "branches": ["claim"],
             "tool_results": [
                 {"name": "query_order", "ok": True, "attempts": 1},
                 {"name": "create_ticket", "ok": True, "attempts": 1},
             ]},
        ]
    )
    result = run_dialogue(graph, row)

    assert result["turns"] == 2
    assert result["facts"]["last_reply"].startswith("已赔付")
    assert set(result["facts"]["tools"]) == {"query_order", "create_ticket"}
    assert result["tools"]["success_rate"] == 1.0
    assert result["resolved_predicted"] is True
    assert result["resolved_match"] is True
    # 每轮都带上同一个 session_id，多轮记忆才能续接
    assert {call["payload"]["session_id"] for call in graph.calls} == {"eval-dlg-dg-99"}
    assert len(graph.calls) == 2


def test_run_dialogue_marks_unresolved_when_check_fails():
    row = make_row(turns=[{"user": "东西烂了"}], expected_goal_resolved=True)
    graph = FakeGraph([{"reply": "请提供订单号", "intent": "bad_fruit_claim",
                        "branches": ["claim"], "tool_results": []}])
    result = run_dialogue(graph, row)
    assert result["resolved_predicted"] is False
    assert result["resolved_match"] is False
    assert result["checks"]["tools"] is False


def test_run_dialogue_merges_branches_across_turns():
    row = make_row(turns=[{"user": "a"}, {"user": "b"}],
                   check={"branches": ["claim", "faq"]})
    graph = FakeGraph(
        [
            {"reply": "x", "branches": ["claim"], "tool_results": []},
            {"reply": "y", "branches": ["faq"], "tool_results": []},
        ]
    )
    result = run_dialogue(graph, row)
    assert result["facts"]["branches"] == ["claim", "faq"]
    assert result["checks"]["branches"] is True


# ---------------- 报告分节：真失败 vs 标注待复核 ----------------


def make_reported_record(**overrides) -> dict:
    """报告用例需要完整字段（回放、判据明细、工具聚合）。"""
    record = make_record(**overrides)
    record.setdefault("scenario", "测试用例")
    record.setdefault("resolution_criteria", "给出赔付结论")
    record.setdefault("note", "")
    record.setdefault("check_details", {})
    record.setdefault(
        "turn_log",
        [
            {
                "index": 1,
                "user": "东西烂了",
                "intent": "bad_fruit_claim",
                "branches": ["claim"],
                "reply": "请提供订单号",
            }
        ],
    )
    return record


def test_write_report_splits_failures_from_label_review(tmp_path):
    """判据没过才进第四节；判据全过但标注口径不一致的进第五节。

    这两类在旧版混在一张表里，8 段行为完全正确的对话被当成"未通过"，
    看报告的人会得出"模型有 11 处缺陷"的错误结论。
    """
    failed = make_reported_record(
        id="dg-11",
        scenario="理赔中途要求人工",
        resolved_expected=True,
        resolved_predicted=False,
        handoff_expected=True,
        handoff_actual=False,
        checks={"handoff_summary_contains": False},
        check_details={"missing_in_summary": ["2026093000123"]},
    )
    review = make_reported_record(
        id="dg-02",
        scenario="查单-缺订单号",
        resolved_expected=False,
        resolved_predicted=True,
        handoff_expected=False,
        handoff_actual=False,
        checks={"reply_contains": True},
    )

    path = tmp_path / "dialogues.md"
    write_report(path, "rule", "priority", [failed, review], [], Path("dialogues_golden.jsonl"))
    text = path.read_text(encoding="utf-8")

    assert "## 四、判据未通过（1 段" in text
    assert "## 五、标注待复核（1 段" in text
    # 真失败要带回放，并写清缺口在哪
    assert "2026093000123" in text
    section4, section5 = text.split("## 五、标注待复核")
    assert "dg-11" in section4 and "dg-02" not in section4
    assert "dg-02" in section5
