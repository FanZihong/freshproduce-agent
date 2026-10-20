"""域外问题判定：三路信号 + 一条"别误杀"的护栏。

三路信号（计划里定的）：
  0. **模型的 chitchat 判定**（第一路，也是最有效的一路：本地模型域外召回 0.94）
  A. 模型 top1 置信度低（`confidence < low_confidence`）
  B. 与所有已知类别的语义相似度都低（可选，需要 embedder）
  C. 规则版判为兜底（没有任何意图关键词命中）

外加一条**护栏**：命中领域词表（订单/物流/赔付/发票/商品名…）时**强制判为域内**。
原因很实际——验收标准里有"已知类误杀率 < 5%"，而小模型偶尔会对很典型的业务问句给低置信度，
如果只靠投票，这些会被误判成域外、直接走兜底分支，用户体感极差。

判定规则：命中护栏 → 域内；否则**模型判 chitchat 就直接域外**（实测 0.94 召回），
其余信号按"可用信号里至少 min_votes 个投 OOD"兜底，用来捞模型漏判的域外问句。
再加一条**规则版否决**：规则版明确判成业务意图时，即使模型说 chitchat 也判域内——
实测这能把已知类误杀率从 6.8% 压到 5% 以下，而域外集里规则版基本都判兜底，不受影响。
注意 required_votes = min(min_votes, 可用信号数)，所以没有 embedder 时 A、C 必须一致才判域外。

为什么把"模型自己判 chitchat"单列成第一路：第一版实现里只有 A/B/C 三路投票，阈值又都按
原文默认值（置信度 0.5、相似度 0.35），而微调模型给什么都打 0.9+ 的置信度、bge 相似度也普遍
高于 0.35 —— 结果三路几乎都不投 OOD，域外召回从模型自身的 0.94 掉到 0.08。
教训：组合信号必须按"实际后端的输出分布"重新标定阈值，否则多路投票会变成多路都不敢说话。

为什么不用"生成式输出一个 out_of_domain 标签"：那要额外训练数据，而且小模型对"我没见过"的
自我认知很不可靠。用可解释的信号组合，更好调、也更好在报告里讲清楚。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

# 本体系里 chitchat（闲聊与域外）承担"域外兜底"职责
OOD_LABEL = "chitchat"

# 领域话题词：命中就认为"这是业务问题"，用于压制误杀
DOMAIN_TOPICS: tuple[str, ...] = (
    "订单", "物流", "快递", "发货", "派送", "签收", "收货", "到货", "配送",
    "退款", "退货", "赔付", "理赔", "赔偿", "补发", "差价", "退款到账", "到账",
    "坏果", "坏", "烂", "发霉", "变质", "异味", "异物", "压坏", "磕碰",
    "缺斤少两", "少称", "重量", "数量不对", "分量",
    "发票", "开票", "抬头", "税号", "电子发票",
    "保存", "储存", "冷藏", "冷冻", "保鲜", "保质期", "保质", "能放多久", "存放",
    "时效", "多久到", "多久发货", "催", "催单", "加急",
    "投诉", "差评", "曝光", "转人工", "人工客服", "客服", "售后",
    "优惠券", "运费", "邮费", "包邮", "无理由退货", "订单号", "运单号",
    "商品", "生鲜", "品质", "新鲜", "批次", "产地", "规格", "斤", "盒", "克",
)


def _products() -> tuple[str, ...]:
    """商品名复用检索层已有的词表，避免两处维护。"""
    try:
        from agent.rag.rewrite import KNOWN_PRODUCTS

        return tuple(KNOWN_PRODUCTS)
    except Exception:  # noqa: BLE001 - 检索层不可用时不影响域外判定
        return ()


@dataclass
class OodConfig:
    """判定阈值（默认值是先验，正式值应由域外测试集 + 已知类误杀率调出来）。"""

    low_confidence: float = 0.5
    min_similarity: float = 0.35
    min_votes: int = 2

    def to_dict(self) -> dict:
        return {
            "low_confidence": self.low_confidence,
            "min_similarity": self.min_similarity,
            "min_votes": self.min_votes,
        }


@dataclass
class OodSignals:
    """一路输入的信号快照，便于在报告里逐条复盘。"""

    text: str
    model_label: str | None = None
    model_says_ood: bool = False
    rule_business: bool = False
    confidence: float | None = None
    max_similarity: float | None = None
    rule_fallback: bool = False
    domain_hits: tuple[str, ...] = ()
    votes: list[str] = field(default_factory=list)
    available: int = 0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model_label": self.model_label,
            "model_says_ood": self.model_says_ood,
            "rule_business": self.rule_business,
            "confidence": self.confidence,
            "max_similarity": self.max_similarity,
            "rule_fallback": self.rule_fallback,
            "domain_hits": list(self.domain_hits),
            "votes": list(self.votes),
        }


def domain_hits(text: str) -> tuple[str, ...]:
    """命中哪些领域词（商品名也算）。"""
    normalized = (text or "").strip()
    if not normalized:
        return ()
    hits = [word for word in DOMAIN_TOPICS if word in normalized]
    hits += [product for product in _products() if product and product in normalized]
    return tuple(dict.fromkeys(hits))


def collect_signals(
    text: str,
    confidence: float | None = None,
    max_similarity: float | None = None,
    rule_label: str | None = None,
    model_label: str | None = None,
    config: OodConfig | None = None,
) -> OodSignals:
    """把各路原始输入折算成投票。

    model_label：当前后端自己判出的标签（判成 chitchat 就是"模型觉得这是域外"）；
    rule_label：规则版判出的标签（fallback / chitchat 即投 OOD）。
    """
    config = config or OodConfig()
    signals = OodSignals(
        text=text,
        model_label=model_label,
        model_says_ood=model_label == OOD_LABEL,
        rule_business=rule_label is not None and rule_label not in {"", "fallback", OOD_LABEL},
        confidence=confidence,
        max_similarity=max_similarity,
        rule_fallback=rule_label is not None and rule_label in {"fallback", OOD_LABEL},
        domain_hits=domain_hits(text),
    )

    if model_label is not None:
        signals.available += 1
        if signals.model_says_ood:
            signals.votes.append("model_chitchat")
    if confidence is not None:
        signals.available += 1
        if confidence < config.low_confidence:
            signals.votes.append("low_confidence")
    if max_similarity is not None:
        signals.available += 1
        if max_similarity < config.min_similarity:
            signals.votes.append("low_similarity")
    if rule_label is not None:
        signals.available += 1
        if signals.rule_fallback:
            signals.votes.append("rule_fallback")
    return signals


def required_votes(signals: OodSignals, config: OodConfig) -> int:
    """可用信号不足时自动降低要求，避免"没装 embedder 就永远判不出域外"。"""
    return max(1, min(config.min_votes, signals.available or 1))


def is_out_of_domain(
    signals: OodSignals,
    config: OodConfig | None = None,
    use_rule_veto: bool = True,
) -> tuple[bool, str]:
    """返回 (是否域外, 理由)。理由会写进 trace，方便排查误判。"""
    config = config or OodConfig()
    if signals.domain_hits:
        return False, f"命中领域词（{'、'.join(signals.domain_hits[:3])}），强制判域内"
    if use_rule_veto and signals.rule_business:
        return False, "规则版判为业务意图，否决域外判定"
    if signals.model_says_ood:
        return True, "模型自身判为 chitchat（第一路信号）"
    need = required_votes(signals, config)
    if len(signals.votes) >= need:
        return True, f"命中 {len(signals.votes)}/{signals.available} 路信号（{'、'.join(signals.votes)}）"
    return False, (
        f"仅命中 {len(signals.votes)}/{signals.available} 路信号（需要 {need} 路），判域内"
        if signals.votes
        else "没有信号命中，判域内"
    )


# ---------------- 语义相似度（可选信号） ----------------


def build_class_centroids(embedder, taxonomy) -> dict[str, list[float]]:
    """用类别定义与正例算每个意图的质心向量，作为"域内长什么样"的参照。"""
    centroids: dict[str, list[float]] = {}
    for spec in taxonomy.labels:
        texts = [f"{spec.name}：{spec.definition}", *spec.positives]
        vectors = embedder.encode(texts)
        if vectors is None or len(vectors) == 0:   # numpy 数组不能用真值判断
            continue
        centroids[spec.key] = _mean_vector(vectors)
    return centroids


def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    rows = [list(vector) for vector in vectors]
    if not rows:
        return []
    width = len(rows[0])
    sums = [0.0] * width
    for row in rows:
        for index in range(min(width, len(row))):
            sums[index] += row[index]
    return [value / len(rows) for value in sums]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def max_class_similarity(
    embedder, centroids: dict[str, list[float]], text: str
) -> float:
    """文本与"最像的那个意图"的相似度；越像业务问题，这个值越高。"""
    if not centroids:
        return 1.0  # 没有参照时不投票，避免误判
    vector = embedder.encode([text])
    if vector is None or len(vector) == 0:
        return 1.0
    query = list(vector[0])
    return round(max(cosine(query, centroid) for centroid in centroids.values()), 4)


def make_similarity_fn(embedder) -> Callable[[str], float]:
    """把 embedder 包成 (text) -> max_similarity 的函数，供评测脚本注入。"""
    from agent.intent.taxonomy import load_taxonomy

    centroids = build_class_centroids(embedder, load_taxonomy())

    def similarity(text: str) -> float:
        return max_class_similarity(embedder, centroids, text)

    return similarity
