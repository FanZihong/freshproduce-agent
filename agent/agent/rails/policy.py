"""策略层：把散落在图/工具/决策里的护栏判断统一收口到 rails。

设计原则是**行为零变化**：
- 规则文件存在时，由 YAML 里的阈值/关键词决策，结果和旧代码一致；
- 规则文件缺失或 `RAILS_ENABLED=off` 时，退回内置表（和旧代码逐字一致）；
- 新能力（PII 标记、输出承诺改写）只做"标记/改写"，不改变路由。
"""

from __future__ import annotations

from agent.config import get_env
from agent.memory.long_term import PHONE_PATTERN
from agent.rails.engine import RailEngine
from agent.rails.models import RailEvent, RailRunResult
from agent.rails.registry import RailRegistry, get_registry

# 规则文件缺失时的兜底：与迁移前 router.DISPUTE_RULES 完全一致（含顺序）
FALLBACK_DISPUTE: tuple[tuple[str, str], ...] = (
    ("全退", "用户要求全额退款"),
    ("全赔", "用户要求全额赔付"),
    ("投诉", "用户表达投诉意愿"),
    ("差评", "用户威胁给差评"),
    ("曝光", "用户威胁曝光"),
    ("太差", "用户强烈不满"),
    ("不行", "用户拒绝当前赔付方案"),
)
FALLBACK_MAX_AUTO_REFUND = 200.0


def enabled() -> bool:
    mode = (get_env("RAILS_ENABLED", "on") or "on").strip().lower()
    return mode not in {"off", "0", "false", "no"}


def _subset(
    registry: RailRegistry, tags: set[str] | None = None, *, scope: str | None = None
) -> RailRegistry:
    rules = [
        rule
        for rule in registry.rules(scope=scope)
        if not tags or set(rule.tags) & tags
    ]
    return RailRegistry(registry.directory, {rule.id: rule for rule in rules}, registry.errors)


def _run(
    tags: set[str] | None, event: RailEvent, *, scope: str | None = None
) -> RailRunResult | None:
    """返回 None 表示"没有可用规则"，调用方走旧逻辑兜底。"""
    if not enabled():
        return None
    try:
        registry = get_registry()
        subset = _subset(registry, tags, scope=scope)
        if not subset.rules():
            return None
        return RailEngine(subset).run([event])
    except Exception as exc:  # noqa: BLE001 - 护栏失败必须回退旧逻辑
        print(f"[warn] 护栏执行失败，回退旧逻辑：{type(exc).__name__}: {exc}")
        return None


def input_guard(text: str) -> RailRunResult | None:
    """跑一遍全部输入护栏（block/flag/handoff 都可能出现），结果供图节点决策。"""
    return _run(None, RailEvent("user_message", {"text": str(text or "")}), scope="input")


def dispute_hit(text: str) -> dict | None:
    """争议词：返回 {rule, reason}；没有命中返回 None。"""
    result = _run({"dispute"}, RailEvent("user_message", {"text": str(text or "")}))
    if result is None:
        lowered = str(text or "")
        for keyword, reason in FALLBACK_DISPUTE:
            if keyword in lowered:
                return {
                    "rule": f"fallback.dispute.{keyword}",
                    "scope": "input",
                    "action": "handoff",
                    "reason": reason,
                    "why": [keyword],
                }
        return None
    if not result.handoff:
        return None
    return {
        "rule": result.handoff["rule"],
        "scope": "input",
        "action": "handoff",
        "reason": result.handoff.get("reason") or result.handoff.get("message") or "",
        "why": [result.handoff.get("message") or ""],
    }


def dispute_reason(text: str) -> str | None:
    hit = dispute_hit(text)
    return hit["reason"] if hit else None


def pii_hit(text: str) -> dict | None:
    result = _run({"pii"}, RailEvent("user_message", {"text": str(text or "")}))
    if result and result.flags:
        return result.flags[0]
    return None


def ood_hit(out_of_domain: bool) -> dict | None:
    """OOD 信号只做标记：路由仍由 route_after_intent 决定，行为不变。"""
    result = _run({"ood"}, RailEvent("intent", {"out_of_domain": bool(out_of_domain)}))
    if result and result.flags:
        return result.flags[0]
    return None


def loop_hit(loop_count: int) -> dict | None:
    result = _run({"loop"}, RailEvent("intent", {"loop_count": int(loop_count or 0)}))
    if result is None:
        if int(loop_count or 0) >= 3:
            return {"rule": "fallback.loop_guard", "reason": "连续多轮未能获取有效信息"}
        return None
    return result.handoff


def clarify_hit(clarify_rounds: int, max_rounds: int) -> dict | None:
    result = _run(
        {"clarify"},
        RailEvent(
            "intent",
            {"clarify_rounds": int(clarify_rounds or 0), "max_clarify_rounds": int(max_rounds or 0)},
        ),
    )
    if result is None:
        if int(clarify_rounds or 0) >= int(max_rounds or 0):
            return {"rule": "fallback.clarify_cap", "reason": "澄清次数已达上限"}
        return None
    return result.handoff


def retrieval_hit(data: dict) -> dict | None:
    found = bool((data or {}).get("found"))
    result = _run(
        {"grounding"},
        RailEvent(
            "retrieval",
            {"found": found, "top_score": float((data or {}).get("top_score") or 0.0)},
        ),
    )
    if result is None:
        return None if found else {"rule": "fallback.no_evidence", "reason": "知识库无据"}
    return result.handoff


def tool_hit(tool: str, data: dict) -> dict | None:
    """工具成功后校验：金额超限、工单号格式异常。"""
    payload = {"tool": tool, "text": str((data or {}).get("ticket_id") or ""), **(data or {})}
    result = _run({"amount", "ticket_format"}, RailEvent("tool_result", payload))
    if result is None:
        try:
            amount = float((data or {}).get("refund_amount") or 0)
        except (TypeError, ValueError):
            amount = 0.0
        if tool == "create_ticket" and amount > FALLBACK_MAX_AUTO_REFUND:
            return {"rule": "fallback.high_refund", "reason": "赔付金额超过自动审核上限"}
        return None
    return result.handoff


def _mask_phone(text: str) -> str:
    return PHONE_PATTERN.sub(
        lambda match: f"{match.group(0)[:3]}****{match.group(0)[-4:]}", text
    )


def output_guard(reply: str) -> tuple[str, list[dict]]:
    """输出护栏：手机号脱敏 + 承诺/泄漏规则改写；返回 (文本, 命中的规则)。"""
    if not enabled():
        return reply, []
    result = output_run(reply)
    if result is None:
        return reply, []
    text = _mask_phone(str(reply or ""))
    if result.final_text:
        text = result.final_text
    return text, result.hits


def output_run(reply: str) -> RailRunResult | None:
    """输出护栏的原始事件结果（demo/trace 想看事件演化时用）。"""
    return _run({"promise", "leak"}, RailEvent("reply", {"text": str(reply or "")}))
