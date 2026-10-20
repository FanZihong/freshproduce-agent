"""查询改写：把多轮追问补全成可独立检索的问题。

先用规则版做指代消解（利用上一轮问题、已识别商品与槽位），
接口固定为 rewrite(query, context) -> str，后续可直接替换成 LLM 改写。
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field, replace

# 生鲜售后场景里的高频商品词，用于识别"上文在说哪个商品"
KNOWN_PRODUCTS: tuple[str, ...] = (
    "草莓", "车厘子", "蓝莓", "树莓", "牛油果", "黄桃", "香蕉", "芒果",
    "火龙果", "榴莲", "土豆", "洋葱", "南瓜", "胡萝卜", "白萝卜", "叶菜",
    "鸡蛋", "冷鲜肉", "冷冻肉", "牛油果", "水产", "蓝莓", "草莓",
)

# 指代词：出现它们通常说明这句话依赖上文
ANAPHORA_WORDS: tuple[str, ...] = ("这个", "那个", "这款", "此商品", "该商品", "这件", "它")

# 主题词：出现它们说明问题已经自带话题，不需要补商品
DOMAIN_TOPICS: tuple[str, ...] = (
    "赔付", "退款", "退货", "发票", "物流", "发货", "签收", "保存", "储存",
    "保质", "重量", "缺斤少两", "坏果", "变质", "保鲜", "食用", "冷冻", "冷藏",
    "时效", "催单", "理赔", "补发", "优惠券", "到账",
)

# 从上一轮问题里抽取"问的是什么"
QUESTION_TOPICS: tuple[str, ...] = (
    "怎么保存", "怎么赔", "能退吗", "多久到账", "怎么开发票", "怎么开票",
    "多久发货", "怎么处理", "怎么吃", "怎么保鲜", "多少钱", "多久能到",
)


@dataclass
class RewriteContext:
    """改写所需的上下文（来自会话状态）。"""

    last_question: str = ""
    last_answer: str = ""
    product: str = ""
    slots: dict = field(default_factory=dict)
    # 检索召回到的长期事实（intent_node 注入；few-shot 分类会把它当上下文读）
    memory: tuple[str, ...] = ()


@dataclass(frozen=True)
class RewriteResult:
    original: str
    query: str
    rewritten: bool
    reason: str
    strategy: str = "rule"
    model: str = ""
    latency_ms: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _as_context(context) -> RewriteContext:
    """容忍 None / dict / RewriteContext 三种入参。"""
    if context is None:
        return RewriteContext()
    if isinstance(context, RewriteContext):
        return context
    if isinstance(context, dict):
        return RewriteContext(
            last_question=str(context.get("last_question") or ""),
            last_answer=str(context.get("last_answer") or ""),
            product=str(context.get("product") or ""),
            slots=dict(context.get("slots") or {}),
        )
    raise TypeError(f"不支持的上下文类型：{type(context)!r}")


def _find_product(text: str) -> str:
    for product in KNOWN_PRODUCTS:
        if product and product in text:
            return product
    return ""


def _detect_product(context: RewriteContext) -> str:
    """按优先级找"上文在说哪个商品"：显式商品 > 上一轮问题 > 槽位 > 上一轮回答。"""
    if context.product:
        return context.product
    for source in (context.last_question, " ".join(str(v) for v in context.slots.values()), context.last_answer):
        found = _find_product(source or "")
        if found:
            return found
    return ""


def _topic_from(question: str) -> str:
    for topic in QUESTION_TOPICS:
        if topic in question:
            return topic
    return ""


def _has_topic(text: str) -> bool:
    if _find_product(text):
        return True
    return any(topic in text for topic in DOMAIN_TOPICS)


def _clean(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned.strip("，,。？?！! ")


def _rule_rewrite(text: str, ctx: RewriteContext) -> RewriteResult:
    """规则版改写：利用商品与上一轮话题做指代消解。"""
    if not text:
        return RewriteResult(original="", query="", rewritten=False, reason="空查询")

    product = _detect_product(ctx)
    topic = _topic_from(ctx.last_question)
    has_anaphora = any(word in text for word in ANAPHORA_WORDS)

    # 情况一：句子里有指代词，且上文能确定商品 -> 用商品替换指代词
    if has_anaphora and product:
        replaced = text
        for word in ANAPHORA_WORDS:
            replaced = replaced.replace(word, product)
        # 判断"替换后是否只剩语气词"：只有「那它呢」「那这个呢」这类才需要补上一轮主题，
        # 「这个能退吗」这种自带问题词的必须保留原问题。
        remainder = text
        for word in ANAPHORA_WORDS:
            remainder = remainder.replace(word, "")
        remainder = _clean(remainder).strip("呢吗的么啊那")
        if topic and len(remainder) <= 1:
            replaced = f"{product}{topic}"
        return RewriteResult(
            original=text,
            query=_clean(replaced),
            rewritten=True,
            reason=f"用上文商品「{product}」替换指代词",
        )

    # 情况二：没有指代词但缺少话题（如「能退吗」「怎么保存」）-> 补商品
    if product and not _has_topic(text):
        return RewriteResult(
            original=text,
            query=_clean(f"{product}{text}"),
            rewritten=True,
            reason=f"补全商品主题「{product}」",
        )

    return RewriteResult(original=text, query=text, rewritten=False, reason="问题自包含，无需改写")


SYSTEM_PROMPT = (
    "你是检索查询改写器。请结合对话上下文，把用户最后一句问题改写成一条可以独立检索的问题。"
    "只输出改写后的问题本身，不要解释、不要引号、不要加任何前缀。"
    "如果最后一句本身已经是完整问题，就原样输出。"
)


@dataclass
class LLMRewriteConfig:
    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-flash"
    timeout: float = 20.0
    temperature: float = 0.0

    @property
    def configured(self) -> bool:
        """真正可用的条件：已启用 + 填了真实 key（占位符不算）。"""
        return self.enabled and bool(self.api_key) and "替换" not in self.api_key


def load_llm_config() -> LLMRewriteConfig:
    from agent.config import get_bool, get_env, get_float

    return LLMRewriteConfig(
        enabled=get_bool("LLM_REWRITE_ENABLED", False),
        api_key=get_env("LLM_API_KEY", ""),
        base_url=get_env("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        model=get_env("LLM_REWRITE_MODEL", "deepseek-flash"),
        timeout=get_float("LLM_REWRITE_TIMEOUT", 20.0),
    )


def build_rewrite_messages(query: str, context: RewriteContext) -> list[dict]:
    """把最近几轮对话与当前问题拼成 prompt。"""
    lines: list[str] = []
    if context.last_question:
        lines.append(f"上一轮用户问题：{context.last_question}")
    if context.last_answer:
        lines.append(f"上一轮客服回答：{context.last_answer[:120]}")
    if context.product:
        lines.append(f"上文识别到的商品：{context.product}")
    filled_slots = {key: value for key, value in (context.slots or {}).items() if value}
    if filled_slots:
        lines.append(f"已填槽位：{filled_slots}")
    lines.append(f"用户最后一句：{query}")
    lines.append("请输出改写后的检索问题：")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def postprocess_rewrite(raw: str) -> str:
    """清理模型输出：只取第一行，去掉引号与常见前缀。"""
    text = (raw or "").strip()
    if not text:
        return ""
    text = text.splitlines()[0].strip()
    for prefix in ("改写后：", "改写：", "检索问题：", "问题："):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text.strip("\"'“”‘’「」『』`。 ").strip()


class LLMRewriter:
    """LLM 改写：默认走 DeepSeek（OpenAI 兼容协议）。

    complete_fn 用于注入假实现，便于单测与离线验证。
    """

    def __init__(self, config: LLMRewriteConfig | None = None, complete_fn=None) -> None:
        self.config = config or load_llm_config()
        self._complete_fn = complete_fn

    @property
    def available(self) -> bool:
        return self._complete_fn is not None or self.config.configured

    def _complete(self, messages: list[dict]) -> str:
        if self._complete_fn is not None:
            return str(self._complete_fn(messages))
        from openai import OpenAI  # 延迟导入

        client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content or ""

    def rewrite(self, query: str, context: RewriteContext) -> RewriteResult | None:
        """返回改写结果；不可用返回 None，调用失败返回带 error 的结果。"""
        if not self.available:
            return None

        started = time.perf_counter()
        try:
            candidate = postprocess_rewrite(self._complete(build_rewrite_messages(query, context)))
        except Exception as exc:  # noqa: BLE001 - 外部依赖失败要退化为原文
            return RewriteResult(
                original=query,
                query=query,
                rewritten=False,
                reason="LLM 调用失败，已退回原文",
                strategy="llm-error",
                model=self.config.model,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        if not candidate:
            return RewriteResult(
                original=query,
                query=query,
                rewritten=False,
                reason="LLM 返回为空，保留原文",
                strategy="llm",
                model=self.config.model,
                latency_ms=latency_ms,
            )
        if candidate == query:
            return RewriteResult(
                original=query,
                query=query,
                rewritten=False,
                reason="LLM 判断无需改写",
                strategy="llm",
                model=self.config.model,
                latency_ms=latency_ms,
            )
        return RewriteResult(
            original=query,
            query=candidate,
            rewritten=True,
            reason="LLM 结合上下文改写",
            strategy="llm",
            model=self.config.model,
            latency_ms=latency_ms,
        )


_DEFAULT_REWRITER: LLMRewriter | None = None


def get_llm_rewriter() -> LLMRewriter:
    """进程内单例，避免每次改写都重建客户端。"""
    global _DEFAULT_REWRITER
    if _DEFAULT_REWRITER is None:
        _DEFAULT_REWRITER = LLMRewriter()
    return _DEFAULT_REWRITER


def reset_llm_rewriter() -> None:
    """配置变更（例如测试里改环境变量）后重建。"""
    global _DEFAULT_REWRITER
    _DEFAULT_REWRITER = None


def rewrite_detailed(
    query: str,
    context=None,
    *,
    mode: str = "auto",
    rewriter: LLMRewriter | None = None,
) -> RewriteResult:
    """带解释的改写。

    mode：rule 只用规则；llm 优先 LLM（不可用时退化为规则并标注）；auto 有配置就用 LLM。
    """
    text = _clean(query or "")
    ctx = _as_context(context)
    if mode == "rule":
        return _rule_rewrite(text, ctx)
    if mode not in {"auto", "llm"}:
        raise ValueError(f"未知改写模式：{mode}")

    llm = rewriter or get_llm_rewriter()
    if llm.available:
        result = llm.rewrite(text, ctx)
        if result is not None and result.strategy != "llm-error":
            return result
        if result is not None and mode == "llm":
            # 明确要求 LLM 但调用失败：保留规则兜底，并把错误暴露出来
            return replace(
                _rule_rewrite(text, ctx),
                strategy="rule-fallback",
                model=result.model,
                error=result.error,
            )

    if mode == "llm":
        return replace(
            _rule_rewrite(text, ctx),
            strategy="rule-fallback",
            error="LLM 未配置（未启用或缺少 API key）",
        )
    return _rule_rewrite(text, ctx)


def rewrite(query: str, context=None, *, mode: str = "auto", rewriter=None) -> str:
    """对外主接口：返回改写后的查询语句。"""
    return rewrite_detailed(query, context, mode=mode, rewriter=rewriter).query


def rewrite_cache_key(query: str, context=None, *, mode: str = "auto", model: str = "") -> str:
    ctx = _as_context(context)
    return "|".join([mode, model, ctx.last_question, ctx.product, query])


def rewrite_with_cache(
    query: str,
    context=None,
    *,
    mode: str = "auto",
    cache: dict | None = None,
    model: str = "",
    rewriter: LLMRewriter | None = None,
) -> RewriteResult:
    """带缓存的改写：LLM 调用要花钱，同一用例重复跑不重复付费。

    只有真正由模型产出的结果才入缓存；回落（rule-fallback）与调用失败（llm-error）
    不缓存，否则配好 key 之后会一直命中旧的兜底结果。
    """
    key = rewrite_cache_key(query, context, mode=mode, model=model)
    if cache is not None and key in cache:
        return RewriteResult(**cache[key])

    result = rewrite_detailed(query, context, mode=mode, rewriter=rewriter)
    if cache is not None and result.strategy not in {"rule-fallback", "llm-error"}:
        cache[key] = result.to_dict()
    return result


def context_from_state(state: dict) -> RewriteContext:
    """从 Agent 状态里构造改写上下文（接图时使用）。"""
    messages = state.get("messages") or []
    last_question = ""
    last_answer = ""
    for message in reversed(messages):
        content = str(getattr(message, "content", "") or "")
        if not content:
            continue
        if not last_answer and getattr(message, "type", "") == "ai":
            last_answer = content
        if not last_question and getattr(message, "type", "") == "human":
            last_question = content
        if last_question and last_answer:
            break

    slots = state.get("slots") or {}
    product = str(slots.get("product") or "") or _find_product(str(slots.get("order_id") or ""))
    return RewriteContext(
        last_question=last_question,
        last_answer=last_answer,
        product=product,
        slots=slots,
    )
