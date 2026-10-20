"""多意图路由策略测试：优先级重排、conflict_rule、并行分支、回复合并、覆盖口径。"""

from __future__ import annotations

import pytest

from agent.intent.multi_intent import (
    MULTI_STRATEGIES,
    coverage,
    merge_replies,
    order_by_priority,
    plan,
)
from agent.intent.taxonomy import load_taxonomy


def test_order_by_priority_follows_taxonomy():
    taxonomy = load_taxonomy()
    ordered = order_by_priority(["chitchat", "bad_fruit_claim", "human_handoff"], taxonomy)
    assert ordered == ("human_handoff", "bad_fruit_claim", "chitchat")


def test_priority_strategy_picks_single_branch():
    plan_result = plan(["urge_logistics", "bad_fruit_claim"], "priority")
    # 主意图按体系重排成理赔（优先级更高），只走一条分支
    assert plan_result.primary == "bad_fruit_claim"
    assert plan_result.secondary == "urge_logistics"
    assert plan_result.branches == ("claim",)
    assert plan_result.is_multi is True
    assert "催物流" in plan_result.reply_hint          # 次诉求被写进提示词，不是丢掉


def test_parallel_strategy_keeps_both_branches():
    plan_result = plan(["urge_logistics", "bad_fruit_claim"], "parallel")
    assert plan_result.branches == ("claim", "order")
    assert "并行" in plan_result.reply_hint


def test_parallel_dedupes_when_both_labels_share_route():
    # query_order 与 urge_logistics 都走 order 分支：不应该出现两条一样的
    plan_result = plan(["urge_logistics", "query_order"], "parallel")
    assert plan_result.branches == ("order",)


def test_single_label_behaves_like_before():
    plan_result = plan(["invoice"], "priority")
    assert plan_result.primary == "invoice"
    assert plan_result.secondary is None
    assert plan_result.is_multi is False
    assert plan_result.branches == ("faq",)
    assert plan_result.reply_hint == ""


def test_claim_handoff_pair_follows_conflict_rule():
    """taxonomy 的 conflict_rule：理赔 + 转人工时先走理赔分支。"""
    plan_result = plan(["human_handoff", "bad_fruit_claim"], "priority")
    assert plan_result.primary == "bad_fruit_claim"
    assert plan_result.escalated is True
    assert plan_result.primary_route == "claim"
    assert "conflict_rule" in plan_result.note


def test_unknown_labels_fall_back_to_empty_plan():
    plan_result = plan(["不存在的类别"], "priority")
    assert plan_result.primary is None
    assert plan_result.branches == ()
    assert "兜底" in plan_result.note


def test_max_labels_from_taxonomy_is_respected():
    plan_result = plan(["invoice", "complaint", "chitchat"], "parallel")
    assert plan_result.labels == ("complaint", "invoice")   # 只保留优先级最高的两个


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        plan(["invoice"], "shuffle")
    assert MULTI_STRATEGIES == ("priority", "parallel")


def test_merge_replies_orders_and_prefixes():
    merged = merge_replies(["已为你提交理赔。", "物流方面已加急。"])
    assert merged.startswith("已为你提交理赔。")
    assert "另外" in merged and "物流方面已加急。" in merged
    assert merge_replies(["只有一条"]) == "只有一条"
    assert merge_replies(["", "  ", "有效"]) == "有效"
    assert merge_replies([]) == ""


def test_coverage_counts_secondary_only_in_parallel():
    taxonomy = load_taxonomy()
    predicted = ["urge_logistics", "bad_fruit_claim"]
    gold = ["bad_fruit_claim", "urge_logistics"]
    priority_metrics = coverage(plan(predicted, "priority", taxonomy), gold, taxonomy)
    parallel_metrics = coverage(plan(predicted, "parallel", taxonomy), gold, taxonomy)

    assert priority_metrics["primary_hit"] is True
    assert priority_metrics["secondary_covered"] is False   # 只走 claim
    assert parallel_metrics["secondary_covered"] is True    # claim + order 都走
    assert priority_metrics["both_covered"] is False
    assert parallel_metrics["both_covered"] is True


def test_coverage_handles_single_label_gold():
    taxonomy = load_taxonomy()
    metrics = coverage(plan(["invoice"], "priority", taxonomy), ["invoice"], taxonomy)
    assert metrics["primary_hit"] is True
    assert metrics["secondary_covered"] is None
