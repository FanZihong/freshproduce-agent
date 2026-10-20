"""意图识别入口：规则版 / few-shot 大模型 / 本地小模型三种后端共用一套接口。

接口约定（也是三种后端必须遵守的契约）：
    输入一段文本（+ 可选的多轮上下文），输出主标签、全部标签、置信度与理由。
后端通过环境变量切换（默认 rule，保证离线与单测稳定）：
    INTENT_BACKEND=rule | fewshot | local

- rule：关键词规则，只覆盖 4 类（查询/理赔/转人工/兜底），用作兜底与对照；
- fewshot：`agent/intent/baseline.py` 的大模型 few-shot 基线；
- local：`agent/intent/local_model.py` 的 LoRA 微调小模型（Day 2 落地）。

调用方（图节点 / 路由 / 评测脚本）只依赖 `IntentResult`，换后端不用改一行。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class IntentResult:
    """一次意图识别的结果。

    labels 是完整的标签序列（多意图时第 1 个是主意图，与 label 一致）；
    单标签后端可以不填，`all_labels` 会自动退化成 (label,)。
    """

    label: str
    confidence: float
    reason: str = ""
    labels: tuple[str, ...] = ()

    @property
    def all_labels(self) -> tuple[str, ...]:
        if self.labels:
            return self.labels
        return (self.label,) if self.label else ()


class IntentClassifier(Protocol):
    """三种后端都要满足的接口（新增后端时照这个写即可）。"""

    name: str

    def classify(self, text: str, context: Any = None) -> IntentResult: ...


# 优先级从上到下：先判断是否要人工，再看理赔，最后看查单
HANDOFF_KEYWORDS: tuple[str, ...] = (
    "转人工",
    "人工客服",
    "找人工",
    "投诉",
    "差评",
    "曝光",
    "太差",
    "全退",
    "全赔",
    "不行",
    "不满意",
)

CLAIM_KEYWORDS: tuple[str, ...] = (
    "坏",
    "烂",
    "变质",
    "发霉",
    "理赔",
    "赔偿",
    "赔",
    "缺斤少两",
    "少称",
    "重量不够",
)

ORDER_KEYWORDS: tuple[str, ...] = (
    "订单",
    "物流",
    "快递",
    "发货",
    "到哪",
    "签收",
    "配送",
    "什么时候到",
)

# 知识问答：储存、保鲜、政策、时效这类"需要查文档"的问题
KNOWLEDGE_KEYWORDS: tuple[str, ...] = (
    "保存",
    "储存",
    "冷藏",
    "冷冻",
    "保鲜",
    "保质",
    "怎么放",
    "能放多久",
    "能吃吗",
    "发芽",
    "霉",
    "怎么吃",
    "时效",
    "多久到",
    "多久发货",
    "催单",
    "补偿",
    "退款多久",
    "退款到账",
    "到账",
    "发票",
    "开票",
    "退货",
    "无理由",
    "签收",
    "地址",
    "改地址",
    "预约",
)


def _match(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [keyword for keyword in keywords if keyword in text]


class RuleBasedIntentClassifier:
    """规则分类器：命中关键词越多，置信度越高。"""

    name = "rule-based-v1"

    def classify(self, text: str, context: Any = None) -> IntentResult:
        normalized = (text or "").strip()
        if not normalized:
            return IntentResult("fallback", 0.30, "空输入", ("fallback",))

        handoff_hits = _match(normalized, HANDOFF_KEYWORDS)
        if handoff_hits:
            confidence = min(0.95, 0.62 + 0.08 * len(handoff_hits))
            return IntentResult(
                "human_handoff",
                round(confidence, 2),
                "命中关键词 " + "/".join(handoff_hits),
                ("human_handoff",),
            )

        claim_hits = _match(normalized, CLAIM_KEYWORDS)
        if claim_hits:
            confidence = min(0.95, 0.62 + 0.08 * len(claim_hits))
            return IntentResult(
                "bad_fruit_claim",
                round(confidence, 2),
                "命中关键词 " + "/".join(claim_hits),
                ("bad_fruit_claim",),
            )

        order_hits = _match(normalized, ORDER_KEYWORDS)
        if order_hits:
            confidence = min(0.95, 0.60 + 0.06 * len(order_hits))
            return IntentResult(
                "query_order",
                round(confidence, 2),
                "命中关键词 " + "/".join(order_hits),
                ("query_order",),
            )

        # 知识问答排在订单之后：先保证"查单/理赔"这类带工具动作的意图优先命中
        knowledge_hits = _match(normalized, KNOWLEDGE_KEYWORDS)
        if knowledge_hits:
            confidence = min(0.95, 0.60 + 0.08 * len(knowledge_hits))
            return IntentResult(
                "knowledge_qa",
                round(confidence, 2),
                "命中关键词 " + "/".join(knowledge_hits),
                ("knowledge_qa",),
            )

        return IntentResult("fallback", 0.35, "未命中任何意图关键词", ("fallback",))


_default_classifier = RuleBasedIntentClassifier()

# 后端名 -> 分类器实例（惰性创建并缓存，避免每轮都重建模型客户端）
_classifiers: dict[str, object] = {}
_warned: set[str] = set()

BACKENDS: tuple[str, ...] = ("rule", "fewshot", "local")
DEFAULT_BACKEND = "rule"


def get_backend() -> str:
    """读取 INTENT_BACKEND，非法值一律回到 rule（并只提醒一次）。"""
    from agent.config import get_env

    backend = (get_env("INTENT_BACKEND", DEFAULT_BACKEND) or DEFAULT_BACKEND).strip().lower()
    if backend not in BACKENDS:
        _warn_once(f"unknown:{backend}", f"[意图后端] 不认识的 INTENT_BACKEND={backend}，本次用 rule")
        return DEFAULT_BACKEND
    return backend


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(message, file=sys.stderr)


def get_classifier() -> IntentClassifier:
    """按 INTENT_BACKEND 返回分类器；不可用时降级到规则版，不让图上炸。"""
    backend = get_backend()
    if backend in _classifiers:
        return _classifiers[backend]  # type: ignore[return-value]

    classifier: IntentClassifier = _default_classifier
    if backend == "fewshot":
        from agent.intent.baseline import FewShotIntentClassifier  # 延迟导入，避免强依赖 openai

        candidate = FewShotIntentClassifier()
        if candidate.available:
            classifier = candidate
        else:
            _warn_once(
                "fewshot-unconfigured",
                "[意图后端] 选了 fewshot，但 agent/.env 里没配可用的 LLM_API_KEY，本次回落 rule",
            )
    elif backend == "local":
        try:
            from agent.intent.local_model import LocalIntentClassifier  # type: ignore

            candidate = LocalIntentClassifier()
            if candidate.available:
                classifier = candidate
            else:
                # 这里必须显式判 available：LocalIntentClassifier 是惰性加载的，
                # 构造它不会报错，"模型缺失"要到第一次 classify 才炸——图就崩了。
                _warn_once(
                    "local-missing",
                    f"[意图后端] 选了 local，但 {candidate.adapter_dir} 里没有 adapter，"
                    "本次回落 rule（可用 INTENT_LOCAL_DIR 指定 adapter 目录）",
                )
        except Exception as exc:  # noqa: BLE001 - 模型没训好也要能用
            _warn_once(
                "local-missing",
                f"[意图后端] 本地小模型加载失败（{type(exc).__name__}: {exc}），本次回落 rule",
            )

    _classifiers[backend] = classifier
    return classifier


def classify(text: str, context: Any = None) -> IntentResult:
    """统一分类入口：由 INTENT_BACKEND 决定用谁，调用方无需关心。"""
    return get_classifier().classify(text, context)
