"""规则匹配：触发词打分 + 条件判断 + 优先级裁决。

触发词打分和技能路由共用同一套口径（`2 + min(len(keyword), 4) * 0.5`），
这样团队只需要理解一种"关键词权重"。

和技能路由的一个**有意差异**：技能选错有代价，所以前两名接近时不选；
护栏漏一条代价更大，所以这里用 `priority` 决定，而不是"分数接近就放弃"。
"""

from __future__ import annotations

import re

from agent.rails.models import RailEvent, RailRule

OPS: tuple[str, ...] = ("==", "!=", ">", ">=", "<", "<=", "in", "contains", "exists", "not_exists")


def _keyword_score(keywords: tuple[str, ...], text: str) -> tuple[float, list[str]]:
    lowered = text.lower()
    hits = [keyword for keyword in keywords if keyword and keyword.lower() in lowered]
    score = sum(2.0 + min(len(keyword), 4) * 0.5 for keyword in hits)
    return score, hits


def check_condition(condition: dict, payload: dict) -> bool:
    field = str(condition.get("field") or "")
    op = str(condition.get("op") or "==")
    # value_field 用来比较两个 payload 字段（如 clarify_rounds >= max_clarify_rounds）
    value_field = str(condition.get("value_field") or "")
    expected = payload.get(value_field) if value_field else condition.get("value")
    actual = payload.get(field)
    if op == "exists":
        return actual is not None
    if op == "not_exists":
        return actual is None
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    if op == "in":
        return actual in (expected or [])
    if op == "contains":
        return str(expected or "") in str(actual or "")
    if actual is None or expected is None:
        return False
    try:
        left, right = float(actual), float(expected)
    except (TypeError, ValueError):
        return False
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    return False


def score_rule(rule: RailRule, event: RailEvent) -> tuple[float, list[str]]:
    """返回 (分数, 命中说明)；不匹配时分为 0。"""
    if rule.event not in {event.type, "*"}:
        return 0.0, []
    for condition in rule.conditions:
        if not check_condition(condition, event.payload):
            return 0.0, []

    text = str(event.payload.get("text") or event.payload.get("reply") or "")
    score, hits = _keyword_score(rule.keywords, text) if rule.keywords else (0.0, [])
    if rule.regex and re.search(rule.regex, text):
        score += 3.0
        hits = [*hits, f"regex:{rule.regex}"]
    if rule.not_regex and not re.search(rule.not_regex, text):
        score += 3.0
        hits = [*hits, f"not_regex:{rule.not_regex}"]

    # 只有条件、没有关键词的规则（如 loop_count >= 3）给到最低分，靠 priority 排序
    if not rule.keywords and not rule.regex and not rule.not_regex and rule.conditions:
        score = rule.min_score
        hits = [f"条件命中：{len(rule.conditions)} 条"]
    return score, hits


def select_rule(rules: list[RailRule], event: RailEvent) -> tuple[RailRule, float, list[str]] | None:
    """在候选规则里选优先级最高、同优先级下分数最高的一条。"""
    best: tuple[RailRule, float, list[str]] | None = None
    best_key: tuple[int, float, str] | None = None
    for rule in rules:
        score, hits = score_rule(rule, event)
        if score < rule.min_score:
            continue
        key = (-rule.priority, -score, rule.id)
        if best_key is None or key < best_key:
            best, best_key = (rule, score, hits), key
    return best
