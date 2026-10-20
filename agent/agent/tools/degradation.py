"""降级策略表：每种工具失败类型都必须有明确去处，不允许静默吞掉。

三类动作：
- ask_again：向用户追问（参数不合法、订单号查不到）
- handoff：转人工（系统不可用、物流缺失、赔付规则未覆盖）
- fallback_reply：直接给兜底话术（保留给后续扩展）
"""

from __future__ import annotations

from dataclasses import dataclass

ASK_AGAIN = "ask_again"
HANDOFF = "handoff"
FALLBACK_REPLY = "fallback_reply"


@dataclass(frozen=True)
class Degradation:
    action: str
    reason: str
    reply: str


DEGRADATION_TABLE: dict[tuple[str, str], Degradation] = {
    ("query_order", "validation"): Degradation(
        ASK_AGAIN,
        "订单号格式不合法",
        "订单号看起来不对，应该是 10-20 位数字，麻烦您核对后重新发给我。",
    ),
    ("query_order", "not_found"): Degradation(
        ASK_AGAIN,
        "订单不存在",
        "没有查到这笔订单，请核对订单号后重新发给我；如果确认无误，我可以帮您转人工核实。",
    ),
    ("query_order", "error"): Degradation(
        HANDOFF,
        "订单系统不可用",
        "订单系统暂时不可用，我帮您转接人工客服核实。",
    ),
    ("query_logistics", "not_found"): Degradation(
        HANDOFF,
        "物流记录缺失",
        "这笔订单暂时没有物流记录，我帮您转人工核实。",
    ),
    ("query_logistics", "error"): Degradation(
        HANDOFF,
        "物流接口不可用",
        "物流查询暂时不可用，我帮您转人工核实。",
    ),
    ("query_policy", "validation"): Degradation(
        ASK_AGAIN,
        "赔付参数不合法",
        "赔付参数不完整，麻烦您再确认一下坏果比例。",
    ),
    ("query_policy", "not_found"): Degradation(
        HANDOFF,
        "赔付规则未覆盖",
        "这一单的赔付金额超出自动审核范围，需要人工确认，我帮您转接。",
    ),
    ("query_policy", "error"): Degradation(
        HANDOFF,
        "赔付规则服务异常",
        "赔付规则服务暂时不可用，我帮您转人工处理。",
    ),
    ("create_ticket", "validation"): Degradation(
        ASK_AGAIN,
        "工单参数缺失",
        "还差一些必要信息才能提交工单，麻烦您补充一下。",
    ),
    ("create_ticket", "error"): Degradation(
        HANDOFF,
        "工单系统不可用",
        "工单系统暂时不可用，已为您转接人工客服，请稍候。",
    ),
    ("search_faq", "validation"): Degradation(
        ASK_AGAIN,
        "问题描述不完整",
        "麻烦您把问题描述得再具体一点，我再帮您查知识库。",
    ),
    ("search_faq", "error"): Degradation(
        HANDOFF,
        "知识库检索不可用",
        "知识库检索暂时不可用，我帮您转接人工客服。",
    ),
}

DEFAULT_DEGRADATION = Degradation(HANDOFF, "未知工具异常", "系统出现异常，已为您转接人工客服。")


def resolve_degradation(tool: str, kind: str) -> Degradation:
    """查降级表，未登记的组合走默认转人工。"""
    return DEGRADATION_TABLE.get((tool, kind), DEFAULT_DEGRADATION)
