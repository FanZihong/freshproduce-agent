"""few-shot 意图分类基线：用大模型 + 体系定义做 10 类（含多意图、域外）识别。

为什么先做这个基线：
1. 它给出"不训练也能达到的水平"，是后面 LoRA 微调小模型的对照物——
   如果微调打不过它，答案就应该是"基线已够用"，这本身是有价值的结论；
2. 它同时暴露真实成本与 P95 时延，这两项才是本地小模型真正要赢的地方。

接口与规则版对齐（`classify(text) -> IntentResult`），所以图里的意图节点、
路由、评测脚本都不用关心背后是谁在分类：

    from agent.intent.baseline import FewShotIntentClassifier
    clf = FewShotIntentClassifier()
    result = clf.classify("草莓坏了一半怎么赔")
    detail = clf.classify_with_stats("草莓坏了一半怎么赔")   # 带时延 / token / 成本

提示词直接由 `data/intent/taxonomy.yaml` 渲染而来（定义、边界、易混提示、
多意图策略、域外信号都在里面），改体系只要改 yaml，不会出现"prompt 与标注
规范各写一份"的漂移。
"""

from __future__ import annotations

import json
import hashlib
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from eval.metrics import estimate_cost

from agent.intent.classifier import IntentResult, RuleBasedIntentClassifier
from agent.intent.target import output_contract, parse_target
from agent.intent.taxonomy import Taxonomy, load_taxonomy

# 注入式补全函数的签名：(messages) -> 字符串，或 (messages) -> (字符串, usage 字典)
CompleteFn = Callable[[list[dict]], Any]

# few-shot 基线的输出契约：JSON（带 confidence，便于后续做拒答阈值）
OUTPUT_CONTRACT = (
    "只输出一个 JSON 对象，不要解释、不要 Markdown 代码块，格式严格为："
    '{"labels": ["类别key"], "confidence": 0.0, "reason": "一句话理由"}'
)


@dataclass
class LLMIntentConfig:
    """few-shot 分类的模型配置（默认走 DeepSeek 的 OpenAI 兼容协议）。"""

    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    timeout: float = 30.0
    temperature: float = 0.0
    max_tokens: int = 256
    # 计价单位：元 / 百万 token。默认值是 DeepSeek 官网 deepseek-chat 档位的假设值，
    # 换模型或官方调价后请用 INTENT_PRICE_IN / INTENT_PRICE_OUT 覆盖，报告会写明口径。
    price_in_per_m: float = 2.0
    price_out_per_m: float = 8.0
    shots_per_class: int = 1

    @property
    def configured(self) -> bool:
        """真正可用：填了真实 key（占位符不算）。"""
        return bool(self.api_key) and "替换" not in self.api_key


def load_intent_config() -> LLMIntentConfig:
    from agent.config import get_env, get_float

    return LLMIntentConfig(
        api_key=get_env("LLM_API_KEY", ""),
        base_url=get_env("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        # 没单独配 INTENT_MODEL 时沿用预标注用的 LABEL_MODEL，避免两处口径不一致
        model=get_env("INTENT_MODEL", "") or get_env("LABEL_MODEL", "deepseek-chat"),
        timeout=get_float("INTENT_TIMEOUT", get_float("LLM_REWRITE_TIMEOUT", 30.0)),
        price_in_per_m=get_float("INTENT_PRICE_IN", 2.0),
        price_out_per_m=get_float("INTENT_PRICE_OUT", 8.0),
    )


@dataclass(frozen=True)
class IntentCall:
    """一次分类调用的完整记录：结果 + 时延 + token + 成本。"""

    label: str
    labels: tuple[str, ...]
    confidence: float
    reason: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    model: str = ""
    strategy: str = "fewshot"
    cached: bool = False
    error: str = ""
    raw: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["labels"] = list(self.labels)
        return payload


# 人工写的"难例"：taxonomy 的正例是干净样本，这几条专门覆盖最容易混的边界，
# 让提示里既有体系定义、也有对照。
HARD_EXAMPLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("我的订单到哪了", ("query_order",)),
    ("都三天了怎么还不发货，快点", ("urge_logistics",)),
    ("退款一般几天到账", ("refund_consult",)),
    ("车厘子坏了一半能赔多少", ("bad_fruit_claim",)),
    ("说好两斤只有一斤半", ("short_weight",)),
    ("车厘子坏了而且到现在还没发货", ("bad_fruit_claim", "urge_logistics")),
    ("你们客服太差了，我要投诉，顺便转人工", ("human_handoff", "complaint")),
    ("今天天气怎么样", ("chitchat",)),
)


def render_label_catalog(taxonomy: Taxonomy, detail: str = "full") -> list[str]:
    """把类别体系渲染成"可选类别"段落，供 few-shot 提示词与训练提示词共用。

    detail：
    - full    ：定义 + 包含/不含 + 一条易混提示（few-shot 基线用，信息最全）；
    - catalog ：key（名称）：定义（训练用，够模型建立类间边界）；
    - compact ：key（名称）；
    - minimal ：一行列出全部 key（训练提示词最短的形态）。
    """
    if detail == "minimal":
        return ["可选类别（只能从这里选，key 必须原样输出）：" + "、".join(taxonomy.keys)]

    lines = ["可选类别（只能从这里选，key 必须原样输出）："]
    for spec in taxonomy.labels:
        if detail == "compact":
            lines.append(f"- {spec.key}（{spec.name}）")
            continue
        lines.append(f"- {spec.key}（{spec.name}）：{spec.definition}")
        if detail == "full":
            if spec.includes:
                lines.append(f"    包含：{'、'.join(spec.includes)}")
            if spec.excludes:
                lines.append(f"    不含：{'、'.join(spec.excludes)}")
            for confusion in spec.confusions[:1]:
                lines.append(f"    与 {confusion.with_label} 的区别：{confusion.tip}")
    return lines


def _render_examples(taxonomy: Taxonomy, shots_per_class: int) -> list[str]:
    """样例 = 体系正例（每类前 N 条）+ 人工难例。"""
    lines = ["参考样例（输入 -> 输出）："]
    seen: set[str] = set()
    for spec in taxonomy.labels:
        for text in spec.positives[: max(1, shots_per_class)]:
            if text in seen:
                continue
            seen.add(text)
            payload = {
                "labels": [spec.key],
                "confidence": 0.95,
                "reason": f"典型{spec.name}问法",
            }
            lines.append(f"- 输入：{text} -> " + json.dumps(payload, ensure_ascii=False))
    for text, labels in HARD_EXAMPLES:
        if not text or text in seen:
            continue
        seen.add(text)
        payload = {"labels": list(labels), "confidence": 0.9, "reason": "边界样例"}
        lines.append(f"- 输入：{text} -> " + json.dumps(payload, ensure_ascii=False))
    return lines


def build_system_prompt(taxonomy: Taxonomy | None = None, shots_per_class: int = 1) -> str:
    """把 taxonomy 渲染成分类提示词（定义 + 边界 + 多意图策略 + 域外信号 + 样例）。"""
    taxonomy = taxonomy or load_taxonomy()
    policy = taxonomy.multi_intent_policy or {}
    max_labels = int(policy.get("max_labels", 2))

    lines = [
        "你是生鲜电商售后客服的意图识别器，你的输出会被程序直接解析。",
        "任务：判断用户这句话属于哪些意图类别，用于把会话路由到"
        "查单 / 理赔 / 知识问答 / 转人工 / 兜底分支。",
        "",
        *render_label_catalog(taxonomy, "full"),
        "",
        "多意图规则：",
        f"- 最多输出 {max_labels} 个标签，按重要程度排序（第 1 个是主意图）。",
        f"- 排序依据（越靠前越优先）：{' > '.join(taxonomy.routing_priority)}",
    ]
    if policy.get("example"):
        lines.append(f"- 示例：{policy['example']}")
    if policy.get("conflict_rule"):
        lines.append(f"- 冲突处理：{policy['conflict_rule']}")

    lines += ["", "域外与低置信处理："]
    if taxonomy.out_of_domain.get("note"):
        lines.append(f"- {taxonomy.out_of_domain['note']}")
    lines.append("- 与售后完全无关的寒暄或无关话题，标签用 chitchat。")
    lines.append("- 不要勉强猜一个最像的：判断不了就降低 confidence，并说明理由。")

    guidelines = taxonomy.labeling_guidelines[:4]
    if guidelines:
        lines += ["", "判断原则："]
        lines.extend(f"- {rule}" for rule in guidelines)

    lines += ["", *_render_examples(taxonomy, shots_per_class), ""]
    lines.append("confidence 是你对本次判断的把握（0-1），系统会用它做拒答与转人工阈值。")
    lines.append(OUTPUT_CONTRACT)
    return "\n".join(lines)


def build_messages(
    text: str,
    context: Any = None,
    taxonomy: Taxonomy | None = None,
    shots_per_class: int = 1,
    system_prompt: str | None = None,
    output_hint: str = "请输出 JSON：",
) -> list[dict]:
    """拼多轮上下文 + 当前问句；上下文缺省时退化为单轮分类。

    output_hint 是用户轮的最后一句提示，要和 system prompt 里的输出契约一致
    （few-shot 基线用 JSON；SFT 训练数据用 `主:/辅:` 文本格式时不能写"请输出 JSON"）。
    """
    prompt = system_prompt or build_system_prompt(taxonomy, shots_per_class)
    parts: list[str] = []
    if isinstance(context, dict):
        if context.get("last_question"):
            parts.append(f"上一轮用户问题：{context['last_question']}")
        if context.get("last_answer"):
            parts.append(f"上一轮客服回答：{str(context['last_answer'])[:120]}")
        if context.get("product"):
            parts.append(f"上文商品：{context['product']}")
        memory = context.get("memory") or []
        if memory:
            # 长期事实作为"已知信息"给分类器：追问句（"那笔赔付呢"）会因此更好判断
            parts.append("已知记忆：" + "；".join(str(item)[:80] for item in memory[:3]))
    parts.append(f"用户当前问句：{text}")
    if output_hint:
        parts.append(output_hint)
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "\n".join(parts)},
    ]


def build_training_system_prompt(
    taxonomy: Taxonomy | None = None,
    target_style: str = "key",
    max_labels: int = 2,
    detail: str = "catalog",
) -> str:
    """SFT 训练用的 system prompt：和 few-shot 基线同源，但**不带例句**。

    训练时带例句会让模型去记整段提示词；推理时我们要的是"给定体系就能判类"，
    所以只保留类别目录 + 多意图规则 + 输出格式（格式用纯格式示例说明）。
    """
    taxonomy = taxonomy or load_taxonomy()
    policy = taxonomy.multi_intent_policy or {}
    max_labels = int(policy.get("max_labels", max_labels))
    lines = [
        "你是生鲜电商售后客服的意图识别器。",
        "任务：判断用户这句话属于哪些意图类别，用于把会话路由到"
        "查单 / 理赔 / 知识问答 / 转人工 / 兜底分支。",
        "",
        *render_label_catalog(taxonomy, detail),
        "",
        "多意图规则：",
        f"- 最多 {max_labels} 个标签，第 1 个是主意图。",
        f"- 排序依据（越靠前越优先）：{' > '.join(taxonomy.routing_priority)}",
        "- 与售后无关的寒暄或无关话题用 chitchat；判断不了时不要硬猜最像的类别。",
        "",
        output_contract(target_style, max_labels, taxonomy),
    ]
    return "\n".join(lines)


def parse_intent_response(
    raw: str,
    valid_labels: Sequence[str],
    max_labels: int = 2,
    taxonomy: Taxonomy | None = None,
) -> tuple[tuple[str, ...], float, str]:
    """解析模型输出（宽容版），返回 (labels, confidence, reason)。

    具体容错规则见 `agent.intent.target.parse_target`：JSON 优先，失败则扫 token，
    容忍 `主:`/`辅:` 前缀、逗号/竖线分隔、中文名与旧别名。解析不出合法标签时抛 ValueError。
    """
    return parse_target(raw, valid_labels, taxonomy=taxonomy, max_labels=max_labels)


class FewShotIntentClassifier:
    """few-shot（大模型）意图分类器，接口与规则版一致。"""

    name = "fewshot-llm"

    def __init__(
        self,
        config: LLMIntentConfig | None = None,
        complete_fn: CompleteFn | None = None,
        taxonomy: Taxonomy | None = None,
        shots_per_class: int | None = None,
        fallback: RuleBasedIntentClassifier | None = None,
        response_cache: dict | None = None,
        cache_namespace: str = "intent",
    ) -> None:
        self.config = config or load_intent_config()
        self.taxonomy = taxonomy or load_taxonomy()
        self.shots_per_class = (
            shots_per_class if shots_per_class is not None else self.config.shots_per_class
        )
        self._complete_fn = complete_fn
        self._fallback = fallback or RuleBasedIntentClassifier()
        # 评测重跑很烧钱也很慢：同一 (模型, 提示词, 输入) 命中缓存就直接返回，
        # 缓存条目里带着 token 数，所以成本口径不会因为走缓存而失真。
        self._response_cache = response_cache
        self.cache_namespace = cache_namespace
        self._cache_lock = threading.Lock()
        self._client = None
        self._system_prompt: str | None = None

    @property
    def available(self) -> bool:
        return self._complete_fn is not None or self.config.configured

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = build_system_prompt(self.taxonomy, self.shots_per_class)
        return self._system_prompt

    def build_messages(self, text: str, context: Any = None) -> list[dict]:
        return build_messages(
            text,
            context,
            taxonomy=self.taxonomy,
            shots_per_class=self.shots_per_class,
            system_prompt=self.system_prompt,
        )

    def _client_or_create(self):
        if self._client is None:
            from openai import OpenAI  # 延迟导入：没配 key 时不必引入依赖

            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        return self._client

    def cache_key(self, messages: list[dict]) -> str:
        """缓存键：模型 + 温度 + 样例数 + 完整 messages（含系统提示词）。"""
        payload = json.dumps(
            [self.config.model, self.config.temperature, self.shots_per_class, messages],
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        return f"{self.cache_namespace}:{digest}"

    def _complete_raw(self, messages: list[dict]) -> tuple[str, dict]:
        """返回 (原始文本, usage 字典)。usage 里带 cached=True 表示命中缓存。"""
        key: str | None = None
        if self._response_cache is not None:
            key = self.cache_key(messages)
            with self._cache_lock:
                hit = self._response_cache.get(key)
            if hit:
                return str(hit.get("response", "")), {
                    "prompt_tokens": int(hit.get("prompt_tokens") or 0),
                    "completion_tokens": int(hit.get("completion_tokens") or 0),
                    "cached": True,
                }

        if self._complete_fn is not None:
            output = self._complete_fn(messages)
            if isinstance(output, tuple):
                text, usage = str(output[0]), dict(output[1] or {})
            else:
                text, usage = str(output), {}
        else:
            response = self._client_or_create().chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            raw_usage = getattr(response, "usage", None)
            text = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(raw_usage, "completion_tokens", 0) or 0),
            }

        record = {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
        }
        if key is not None:
            with self._cache_lock:
                self._response_cache[key] = {"response": text, **record}
        return text, record

    def classify_with_stats(self, text: str, context: Any = None) -> IntentCall:
        """带时延 / token / 成本的分类；失败不抛异常，错误信息放在 error 字段。"""
        normalized = (text or "").strip()
        if not normalized:
            return IntentCall(
                label="chitchat",
                labels=("chitchat",),
                confidence=0.3,
                reason="空输入，直接兜底，不调用模型",
                latency_ms=0.0,
                model=self.config.model,
                strategy="short-circuit",
            )

        started = time.perf_counter()
        try:
            raw, usage = self._complete_raw(self.build_messages(normalized, context))
        except Exception as exc:  # noqa: BLE001 - 外部依赖失败要能继续跑完评测
            return IntentCall(
                label="",
                labels=(),
                confidence=0.0,
                reason="调用失败",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                model=self.config.model,
                strategy="fewshot-error",
                error=f"{type(exc).__name__}: {exc}"[:200],
            )

        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cost = estimate_cost(
            prompt_tokens,
            completion_tokens,
            self.config.price_in_per_m,
            self.config.price_out_per_m,
        )
        try:
            labels, confidence, reason = parse_intent_response(
                raw,
                self.taxonomy.keys,
                max_labels=int(self.taxonomy.multi_intent_policy.get("max_labels", 2)),
                taxonomy=self.taxonomy,
            )
        except Exception as exc:  # noqa: BLE001 - 解析失败按"没答对"计入评测
            return IntentCall(
                label="",
                labels=(),
                confidence=0.0,
                reason="输出解析失败",
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
                model=self.config.model,
                strategy="fewshot-parse-error",
                cached=bool(usage.get("cached")),
                error=f"{type(exc).__name__}: {exc}"[:200],
                raw=raw[:400],
            )

        return IntentCall(
            label=labels[0],
            labels=labels,
            confidence=confidence,
            reason=reason,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            model=self.config.model,
            cached=bool(usage.get("cached")),
            raw=raw[:400],
        )

    def classify(self, text: str, context: Any = None) -> IntentResult:
        """对外统一接口；失败时回落规则版，保证图不会因为模型抖动而断掉。"""
        call = self.classify_with_stats(text, context)
        if call.labels:
            return IntentResult(call.label, call.confidence, call.reason, call.labels)
        mapped = self._fallback.classify(text)
        return IntentResult(
            mapped.label,
            mapped.confidence,
            f"few-shot 失败，回落规则版：{call.error or call.reason}",
            (mapped.label,),
        )


_DEFAULT_CLASSIFIER: FewShotIntentClassifier | None = None


def get_fewshot_classifier() -> FewShotIntentClassifier:
    """进程内单例，避免每次分类都重建客户端与提示词。"""
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = FewShotIntentClassifier()
    return _DEFAULT_CLASSIFIER


def classify(text: str, context: Any = None) -> IntentResult:
    return get_fewshot_classifier().classify(text, context)
