"""多意图路由策略：两个标签都成立时，走"优先动作型"还是"并行执行两个分支"。

两种策略的差别只在**路由**，不影响识别指标——所以评测要看的不是 F1，而是
主分支命中率、次诉求覆盖率、双诉求全覆盖率（见 scripts/eval_multi_intent.py）。

- priority（默认）：只走优先级最高的那个分支；次诉求不丢，写进回复提示词，
  由主分支的回复末尾补一句处理入口。
- parallel：两个分支都执行，回复按"主分支在前"的顺序合并。

主意图的判定**不依赖模型输出的顺序**，而是按 taxonomy 的 routing_priority 重排
（human_handoff > complaint > bad_fruit_claim > short_weight > …）。
这样即使模型偶尔把顺序写反，路由也不会跟着乱。

一个例外来自 taxonomy 的 conflict_rule：主意图是 bad_fruit_claim 且辅助是 human_handoff 时，
**先走理赔分支**，遇到争议再转人工——赔付是用户最直接的动作型诉求，先给结论体验更好。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from agent.intent.target import normalize_labels
from agent.intent.taxonomy import load_taxonomy

MULTI_STRATEGIES: tuple[str, ...] = ("priority", "parallel")

# taxonomy.multi_intent_policy.conflict_rule 里写死的这一对：理赔优先于转人工
CLAIM_HANDOFF_PAIR = ("bad_fruit_claim", "human_handoff")


@dataclass
class MultiIntentPlan:
    """一次多意图路由的完整计划（可直接写进 trace，便于复盘）。"""

    labels: tuple[str, ...] = ()
    primary: str | None = None
    secondary: str | None = None
    primary_route: str | None = None
    secondary_route: str | None = None
    strategy: str = "priority"
    branches: tuple[str, ...] = ()
    escalated: bool = False
    reply_hint: str = ""
    note: str = ""

    @property
    def is_multi(self) -> bool:
        return self.secondary is not None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["labels"] = list(self.labels)
        payload["branches"] = list(self.branches)
        return payload


def order_by_priority(labels: Sequence[str], taxonomy) -> tuple[str, ...]:
    """按 routing_priority 重排标签（体系里没写的排最后，保持相对顺序）。"""
    priority = list(taxonomy.routing_priority)
    unique = list(dict.fromkeys(str(label) for label in labels if label))

    def rank(label: str) -> tuple[int, int]:
        return (priority.index(label) if label in priority else len(priority), unique.index(label))

    return tuple(sorted(unique, key=rank))


def plan(
    labels: Sequence[str],
    strategy: str = "priority",
    taxonomy: Any = None,
    max_labels: int | None = None,
) -> MultiIntentPlan:
    """把模型的标签序列变成路由计划。"""
    taxonomy = taxonomy or load_taxonomy()
    if strategy not in MULTI_STRATEGIES:
        raise ValueError(f"不支持的策略：{strategy}（可选 {'/'.join(MULTI_STRATEGIES)}）")
    limit = int(max_labels or taxonomy.multi_intent_policy.get("max_labels", 2))
    clean = list(normalize_labels(labels, set(taxonomy.keys)))[:limit]
    if not clean:
        return MultiIntentPlan(strategy=strategy, note="没有可用标签，交给兜底分支")

    ordered = list(order_by_priority(clean, taxonomy))

    # conflict_rule：理赔 + 转人工时，理赔先走（见 taxonomy.yaml 的 conflict_rule）
    escalated = False
    if set(ordered) == set(CLAIM_HANDOFF_PAIR):
        ordered = ["bad_fruit_claim", "human_handoff"]
        escalated = True

    primary = ordered[0]
    secondary = ordered[1] if len(ordered) > 1 else None
    primary_route = taxonomy.route_of(primary)
    secondary_route = taxonomy.route_of(secondary) if secondary else None

    branches: list[str] = [primary_route]
    if strategy == "parallel" and secondary_route and secondary_route not in branches:
        branches.append(secondary_route)

    hint = ""
    if secondary:
        if strategy == "priority":
            hint = (
                f"用户同时提到「{taxonomy.name_of(secondary)}」：先答主诉求"
                f"（{taxonomy.name_of(primary)}），末尾补一句次级诉求的处理入口。"
            )
        else:
            hint = (
                f"两条诉求并行处理（{taxonomy.name_of(primary)} → "
                f"{taxonomy.name_of(secondary)}），回复按此顺序合并。"
            )

    return MultiIntentPlan(
        labels=tuple(ordered),
        primary=primary,
        secondary=secondary,
        primary_route=primary_route,
        secondary_route=secondary_route,
        strategy=strategy,
        branches=tuple(branches),
        escalated=escalated,
        reply_hint=hint,
        note="命中 conflict_rule：赔付优先，遇争议再转人工" if escalated else "",
    )


def merge_replies(replies: Sequence[str]) -> str:
    """把多个分支的回复按顺序合成一条（parallel 策略用）。"""
    parts = [str(reply).strip() for reply in replies if reply and str(reply).strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    merged = parts[0]
    for part in parts[1:]:
        merged = f"{merged}\n\n另外，{part[0].lower() + part[1:] if len(part) > 1 else part}"
    return merged


def coverage(plan_result: MultiIntentPlan, gold_labels: Sequence[str], taxonomy: Any = None) -> dict:
    """按"分支覆盖"算策略效果：主分支是否命中、次诉求是否也被路由到。"""
    taxonomy = taxonomy or load_taxonomy()
    gold = list(order_by_priority(normalize_labels(gold_labels, set(taxonomy.keys)), taxonomy))
    if not gold:
        return {"primary_hit": None, "secondary_covered": None, "both_covered": None}
    gold_primary_route = taxonomy.route_of(gold[0])
    primary_hit = plan_result.primary_route == gold_primary_route
    if len(gold) < 2:
        return {"primary_hit": primary_hit, "secondary_covered": None, "both_covered": primary_hit}
    gold_secondary_route = taxonomy.route_of(gold[1])
    secondary_covered = gold_secondary_route in plan_result.branches
    return {
        "primary_hit": primary_hit,
        "secondary_covered": secondary_covered,
        "both_covered": bool(primary_hit and secondary_covered),
    }
