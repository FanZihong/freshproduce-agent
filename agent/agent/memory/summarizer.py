"""对话摘要器：规则折叠（默认）与 LLM 摘要（可选）。

为什么要两种：规则折叠（每条消息压一行再截断）零成本、可复现，但只保留"说了什么"，
不保留"事情办到哪一步"；LLM 摘要能把订单号、赔付比例、未完成事项提炼出来，
代价是每次压缩一次调用、有延迟也有费用。默认用规则，`MEMORY_SUMMARY=llm` 才走模型。

和改写共用同一套 LLM 配置（`LLM_API_KEY` / `LLM_BASE_URL` / `LLM_REWRITE_MODEL`），
不额外引入一套凭证；开关独立（`MEMORY_SUMMARY`），避免"想开改写就顺手开了摘要"。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.messages import BaseMessage

from agent.config import get_env, get_float
from agent.rag.rewrite import LLMRewriteConfig

SUMMARY_LIMIT = 200  # 规则摘要的字数上限（沿用历史口径）
LLM_SUMMARY_LIMIT = 120  # LLM 摘要要求更短：给后续轮次看的，不是给人看的

_ROLE_LABEL = {
    "human": "用户",
    "ai": "Agent",
    "system": "系统",
    "tool": "工具",
}

LLM_SYSTEM_PROMPT = """你是客服对话摘要器。把给你的多轮对话压成一段给后续轮次参考的摘要。

要求：
1. 保留：订单号、商品、坏果比例、赔付金额、用户的核心诉求、以及**还没办完的事**；
2. 不要编造对话里没出现的数字或结论；拿不准就写"未确认"；
3. 只输出摘要正文，不要前缀、不要解释，不超过 120 字。
"""

# 注入点：单测与 --fake-llm 演练用，避免测试依赖网络
_COMPLETER: Callable[[list[dict]], object] | None = None


def set_completer(fn: Callable[[list[dict]], object] | None) -> None:
    """替换真实模型调用（测试/演练用）。fn(messages) -> str 或 (str, usage)。"""
    global _COMPLETER
    _COMPLETER = fn


def load_summary_config() -> LLMRewriteConfig:
    """复用改写的 LLM 配置（同一套 key/base_url），模型可单独覆盖。"""
    return LLMRewriteConfig(
        enabled=True,  # 是否真的走 LLM 由 MEMORY_SUMMARY 决定
        api_key=get_env("LLM_API_KEY", ""),
        base_url=get_env("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        model=get_env("MEMORY_SUMMARY_MODEL", "") or get_env("LLM_REWRITE_MODEL", "deepseek-flash"),
        timeout=get_float("MEMORY_SUMMARY_TIMEOUT", 20.0),
    )


def choose_mode(config: LLMRewriteConfig | None = None) -> str:
    """返回 "rule" 或 "llm"。

    `MEMORY_SUMMARY`：rule（默认）| llm | auto。
    auto 的语义是"配了可用 key 就走 LLM，否则规则"；配错值按 rule 处理并提示。
    """
    mode = (get_env("MEMORY_SUMMARY", "rule") or "rule").strip().lower()
    if mode not in {"rule", "llm", "auto"}:
        print(f"[warn] MEMORY_SUMMARY 只能是 rule/llm/auto，收到 {mode!r}，按 rule 处理")
        return "rule"
    if mode == "rule":
        return "rule"
    if _COMPLETER is not None:
        return "llm"
    config = config or load_summary_config()
    if config.configured:
        return "llm"
    if mode == "llm":
        print("[warn] MEMORY_SUMMARY=llm 但没有可用的 LLM_API_KEY，本次按规则折叠")
    return "rule"


# ---------------- 规则折叠（原实现，行为不变） ----------------


def _one_line(message: BaseMessage, limit: int = 40) -> str:
    content = str(getattr(message, "content", "")).replace("\n", " ").strip()
    if len(content) > limit:
        content = content[:limit] + "..."
    label = _ROLE_LABEL.get(getattr(message, "type", ""), "消息")
    return f"{label}: {content}"


def summarize_messages(messages: list[BaseMessage], limit: int = SUMMARY_LIMIT) -> str:
    """规则版摘要：每条消息压成一行，整体再截断。"""
    if not messages:
        return ""
    digest = " | ".join(_one_line(message) for message in messages)
    if len(digest) > limit:
        digest = digest[:limit] + "..."
    return digest


# ---------------- LLM 摘要 ----------------


def build_summary_messages(messages: list[BaseMessage], previous: str = "") -> list[dict]:
    """把"历史摘要 + 本轮要折叠的消息"拼成 prompt。"""
    lines: list[str] = []
    if previous:
        lines.append(f"已有摘要（可以融合、可以改写，但要保留仍然有效的信息）：{previous}")
    lines.append("新增对话：")
    for message in messages[-20:]:
        label = _ROLE_LABEL.get(getattr(message, "type", ""), "消息")
        content = str(getattr(message, "content", "")).replace("\n", " ").strip()
        if len(content) > 120:
            content = content[:120] + "..."
        lines.append(f"{label}：{content}")
    return [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def postprocess_summary(raw: str, limit: int = LLM_SUMMARY_LIMIT) -> str:
    """清理模型输出：取第一段、去掉常见前缀与引号、按字数截断。"""
    text = (raw or "").strip()
    if not text:
        return ""
    text = text.splitlines()[0].strip()
    for prefix in ("摘要：", "摘要:", "总结：", "总结:", "对话摘要：", "压缩结果："):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    text = text.strip("\"'“”‘’「」『』`。 ")
    return text[:limit] + ("…" if len(text) > limit else "")


@dataclass(frozen=True)
class SummaryResult:
    """一次压缩的结果。

    `replaces=True` 表示这段文字就是完整摘要（LLM 已把历史融合进去），调用方直接替换；
    `False` 表示只是本段折叠（规则版），调用方把它拼在旧摘要后面。
    """

    text: str
    mode: str = "rule"
    replaces: bool = False
    meta: dict = field(default_factory=dict)


_STATS: dict = {
    "calls": 0,
    "ok": 0,
    "failed": 0,
    "fallback": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cost": 0.0,
    "latency_ms": 0.0,
}


def stats() -> dict:
    """累积的调用统计（进程内），给消融脚本和排障用。"""
    return dict(_STATS)


def reset_stats() -> None:
    for key in _STATS:
        _STATS[key] = 0.0 if key in {"cost", "latency_ms"} else 0


class LLMSummarizer:
    """LLM 摘要器：可用性判断、失败回退、token/成本记账都在这里。"""

    def __init__(self, config: LLMRewriteConfig | None = None) -> None:
        self.config = config or load_summary_config()

    @property
    def available(self) -> bool:
        return _COMPLETER is not None or self.config.configured

    def _complete(self, messages: list[dict]) -> tuple[str, dict]:
        if _COMPLETER is not None:
            result = _COMPLETER(messages)
            if isinstance(result, tuple):
                return str(result[0]), dict(result[1] or {})
            return str(result), {}
        from openai import OpenAI  # 延迟导入：没配 key 的机器不装也能跑

        client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=0.0,
        )
        usage = getattr(response, "usage", None)
        return response.choices[0].message.content or "", {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }

    def summarize(self, messages: list[BaseMessage], previous: str = "") -> SummaryResult | None:
        """成功返回 SummaryResult；失败返回 None（由调用方回退规则版）。"""
        started = time.perf_counter()
        _STATS["calls"] += 1
        try:
            raw, usage = self._complete(build_summary_messages(messages, previous))
            text = postprocess_summary(raw)
            if not text:
                raise ValueError("模型返回空摘要")
        except Exception as exc:  # noqa: BLE001 - 摘要失败不能拖垮对话
            _STATS["failed"] += 1
            print(f"[warn] LLM 摘要失败，回退规则折叠：{type(exc).__name__}: {exc}")
            return None

        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        latency_ms = (time.perf_counter() - started) * 1000
        cost = _estimate_cost(prompt_tokens, completion_tokens)
        _STATS["ok"] += 1
        _STATS["prompt_tokens"] += prompt_tokens
        _STATS["completion_tokens"] += completion_tokens
        _STATS["cost"] += cost
        _STATS["latency_ms"] += latency_ms
        return SummaryResult(
            text=text,
            mode="llm",
            replaces=True,
            meta={
                "model": self.config.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": round(latency_ms, 1),
                "cost": cost,
            },
        )


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    price_in = get_float("MEMORY_SUMMARY_PRICE_IN", get_float("INTENT_PRICE_IN", 2.0))
    price_out = get_float("MEMORY_SUMMARY_PRICE_OUT", get_float("INTENT_PRICE_OUT", 8.0))
    return round((prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000, 8)


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """对外暴露的成本估算（抽记忆、摘要共用同一套计价口径）。"""
    return _estimate_cost(prompt_tokens, completion_tokens)


def summarize_dialogue(
    messages: list[BaseMessage],
    previous: str = "",
    *,
    limit: int = SUMMARY_LIMIT,
) -> SummaryResult:
    """压缩入口：按配置选后端，LLM 失败自动回退规则。"""
    if choose_mode() == "llm":
        result = LLMSummarizer().summarize(messages, previous)
        if result is not None:
            return result
        _STATS["fallback"] += 1
        return SummaryResult(
            text=summarize_messages(messages, limit=limit),
            mode="rule",
            replaces=False,
            meta={"fallback": True},
        )
    digest = summarize_messages(messages, limit=limit)
    return SummaryResult(text=digest, mode="rule", replaces=False)
