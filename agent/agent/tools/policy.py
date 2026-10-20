"""赔付规则工具。

今天先用一张写死的赔付档位表，明天迁到知识库（混合检索 + Reranker）后，
函数签名保持不变，只换内部实现。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyTier:
    min_ratio: float
    max_ratio: float
    refund_ratio: float
    note: str


# 生鲜坏果赔付标准（示例规则，按 PRD 假设整理）
POLICY_TIERS: tuple[PolicyTier, ...] = (
    PolicyTier(0.00, 0.10, 0.00, "轻微瑕疵，安抚为主，可补偿优惠券"),
    PolicyTier(0.10, 0.30, 0.20, "按订单金额 20% 赔付"),
    PolicyTier(0.30, 0.50, 0.50, "按订单金额 50% 赔付"),
    PolicyTier(0.50, 0.80, 0.80, "按订单金额 80% 赔付"),
    PolicyTier(0.80, 1.00, 1.00, "按订单金额全额赔付"),
)

# 超过这个金额不再自动赔付，需要人工确认（对应"规则未覆盖 -> 转人工"的降级路径）
MAX_AUTO_REFUND = 200.0


def match_tier(ratio: float) -> PolicyTier:
    """按坏果比例匹配档位，比例超出范围时收敛到边界档位。"""
    clamped = min(1.0, max(0.0, ratio))
    for tier in POLICY_TIERS:
        if clamped <= tier.max_ratio:
            return tier
    return POLICY_TIERS[-1]


def query_policy(amount: float, ratio: float, category: str = "浆果类") -> dict:
    """按订单金额与坏果比例计算赔付金额。"""
    tier = match_tier(ratio)
    refund_amount = round(amount * tier.refund_ratio, 2)
    matched = refund_amount <= MAX_AUTO_REFUND
    return {
        "matched": matched,
        "category": category,
        "ratio": round(min(1.0, max(0.0, ratio)), 2),
        "tier": f"{tier.min_ratio:.0%}-{tier.max_ratio:.0%}",
        "refund_ratio": tier.refund_ratio,
        "refund_amount": refund_amount,
        "note": tier.note,
        "reason": None if matched else f"赔付金额超过自动审核上限（{MAX_AUTO_REFUND:.0f} 元），需人工确认",
    }
