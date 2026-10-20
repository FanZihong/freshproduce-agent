"""路由判断：条件边的决策逻辑。

今天先把"读状态做判断"的函数写好并单独验证，明天在 workflow.py 里挂成条件边。
注意：路由不仅看本轮意图，还要看上一轮遗留的 active_flow，否则用户回答
「2026093000123」时会被误判为域外问题。
"""

from __future__ import annotations

from agent.graph.state import AgentState, claim_slots_complete
from agent.intent.taxonomy import route_of

# 本轮意图本身就要求人工介入：转人工 / 投诉。补槽位期间也必须让路。
EXPLICIT_HANDOFF_INTENTS: frozenset[str] = frozenset({"human_handoff", "complaint"})

# 只有这两条分支需要领域知识/话术手册；查单、理赔已经有自己的工具流程
SKILL_BRANCHES: frozenset[str] = frozenset({"faq", "fallback"})


def detect_dispute(text: str) -> str | None:
    """识别争议信号，返回原因；规则在 `data/rails/input.yaml`（tags: dispute）。"""
    try:
        from agent.rails.policy import dispute_reason

        return dispute_reason(text)
    except Exception as exc:  # noqa: BLE001 - 护栏失败不能挡住路由
        print(f"[warn] 争议护栏失败，按无争议继续：{type(exc).__name__}: {exc}")
        return None


def route_after_intent(state: AgentState) -> str:
    """意图识别之后的主路由，返回值对应分支名。

    判定顺序（每一步都有明确理由）：
    1. 争议/已标记转人工 → handoff；
    2. 本轮意图明确是转人工/投诉 → handoff：用户的诉求优先于流程完整；
    3. **延续未完成的流程优先**：用户在补槽位时，下一句可能是「2026093000123」这种
       信息碎片，先做域外/置信度判断会把它误判成域外问题或低置信度追问；
    4. 域外 → fallback（三路信号 + 领域词护栏，见 agent/intent/ood.py）；
    5. 置信度决策：clarify 先澄清、handoff 转人工（见 agent/intent/decision.py）；
    6. parallel 策略且需要两条分支 → multi（两条都执行）；
    7. 其余走单分支：意图 → 分支的映射统一来自 taxonomy.yaml。
    """
    if state.get("handoff"):
        return "handoff"

    dispute = detect_dispute(state.get("user_input", ""))
    if dispute:
        return "handoff"

    # dg-11（理赔中途要求人工）踩过的坑：补槽位时用户说「别问了，我要转人工」，
    # 旧顺序被下面的"续接流程"截胡，系统继续追问坏果比例，把用户诉求吞了。
    if (state.get("intent") or "") in EXPLICIT_HANDOFF_INTENTS:
        return "handoff"

    active_flow = state.get("active_flow")
    slots = state.get("slots") or {}

    # 延续未完成的流程：用户上一轮在补槽位，这一轮大概率还是在补槽位
    if active_flow == "bad_fruit_claim" and not claim_slots_complete(slots):
        return "claim"
    if active_flow == "query_order" and not slots.get("order_id"):
        return "order"

    if state.get("out_of_domain"):
        return "fallback"

    decision = state.get("decision") or "accept"
    if decision == "handoff":
        return "handoff"
    if decision == "clarify":
        return "clarify"

    branches = list(state.get("branches") or [])
    if state.get("multi_strategy") == "parallel" and len(branches) >= 2:
        return "multi"

    intent = state.get("intent") or "fallback"
    # 意图 -> 分支的映射统一来自 taxonomy.yaml（10 类意图映射到 5 个分支）
    return route_of(intent)


def skills_enabled() -> bool:
    from agent.config import get_env

    return (get_env("SKILLS_ENABLED", "on") or "on").strip().lower() not in {
        "off", "0", "false", "no",
    }


def route_with_skills(state: AgentState) -> str:
    """意图之后先经过技能节点（仅 FAQ/兜底）；关掉开关就退回原来的边。"""
    route = route_after_intent(state)
    if route in SKILL_BRANCHES and skills_enabled():
        return "skill"
    return route


def route_after_skill(state: AgentState) -> str:
    """技能节点只是旁路：不改路由状态，这里重算一次原分支。"""
    route = route_after_intent(state)
    return route if route in {"faq", "fallback", "handoff"} else "fallback"


def route_after_input(state: AgentState) -> str:
    """输入护栏拦截后直接生成回复，不再走意图识别。"""
    return "rails_output" if state.get("blocked") else "intent"


def route_after_execution(state: AgentState) -> str:
    """执行护栏命中就转人工，否则进入输出护栏。"""
    return "handoff" if state.get("handoff") else "rails_output"


def route_after_claim(state: AgentState) -> str:
    """理赔节点执行后的走向：连续多轮拿不到有效信息就转人工，否则本轮结束。"""
    if state.get("handoff"):
        return "handoff"
    loop_count = int(state.get("loop_count") or 0)
    try:
        from agent.rails.policy import loop_hit

        if loop_hit(loop_count):
            return "handoff"
    except Exception as exc:  # noqa: BLE001 - 回退旧阈值
        print(f"[warn] 循环护栏失败，回退旧阈值：{type(exc).__name__}: {exc}")
        if loop_count >= 3:
            return "handoff"
    return "end"


def route_after_node(state: AgentState) -> str:
    """通用节点出口：节点把 handoff 置为 True（工具降级）时走转人工分支。"""
    return "handoff" if state.get("handoff") else "end"
