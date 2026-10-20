"""技能条件作答：把技能块注入 system prompt，由大模型生成回复（可选）。

默认 `SKILL_ANSWER=off`：技能只被"选中 + 记录"，回复仍走原来的规则模板，
保证改动不改变现有评测口径。打开后 FAQ / 兜底分支在有技能时改走这里。
"""

from __future__ import annotations

from typing import Callable

from agent.config import get_env, get_float
from agent.rag.rewrite import LLMRewriteConfig

_COMPLETER: Callable[[list[dict]], object] | None = None
VALID_MODES = ("off", "llm")


def set_completer(fn: Callable[[list[dict]], object] | None) -> None:
    """测试/演示注入：fn(messages) -> str。"""
    global _COMPLETER
    _COMPLETER = fn


def answer_mode() -> str:
    mode = (get_env("SKILL_ANSWER", "off") or "off").strip().lower()
    if mode not in VALID_MODES:
        print(f"[warn] SKILL_ANSWER 只能是 off/llm，收到 {mode!r}，按 off 处理")
        return "off"
    return mode


def load_config() -> LLMRewriteConfig:
    return LLMRewriteConfig(
        enabled=True,
        api_key=get_env("LLM_API_KEY", ""),
        base_url=get_env("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        model=get_env("SKILL_ANSWER_MODEL", "") or get_env("LLM_REWRITE_MODEL", "deepseek-flash"),
        timeout=get_float("SKILL_ANSWER_TIMEOUT", 20.0),
    )


def build_messages(user_text: str, skill_prompt: str, hits: list[dict] | None = None) -> list[dict]:
    system = (
        "你是生鲜售后客服。严格按下面的技能手册处理本轮问题：\n\n"
        f"{skill_prompt}\n\n"
        "要求：只使用手册和知识库片段里的信息，不要编造规则或金额；"
        "涉及金额争议、异物理赔、用户要求人工时，说明会转人工。"
    )
    lines = [f"用户问题：{user_text}"]
    if hits:
        lines.append("")
        lines.append("知识库片段：")
        for index, hit in enumerate(hits[:3], start=1):
            lines.append(
                f"{index}. 《{hit.get('doc_title') or ''}》{hit.get('section') or ''}：{hit.get('text') or ''}"
            )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def compose_answer(
    user_text: str,
    skill_prompt: str,
    hits: list[dict] | None = None,
    *,
    completer: Callable[[list[dict]], object] | None = None,
) -> str | None:
    """按技能作答；关闭、没配 key 或调用失败时返回 None（调用方走原逻辑）。"""
    if not skill_prompt or answer_mode() != "llm":
        return None
    runner = completer or _COMPLETER
    if runner is None and not load_config().configured:
        return None
    messages = build_messages(user_text, skill_prompt, hits)
    try:
        if runner is not None:
            result = runner(messages)
        else:
            from openai import OpenAI

            config = load_config()
            client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=config.timeout)
            response = client.chat.completions.create(
                model=config.model, messages=messages, temperature=0.2
            )
            result = response.choices[0].message.content
        text = str(result or "").strip()
        return text or None
    except Exception as exc:  # noqa: BLE001 - 技能作答失败回退原逻辑
        print(f"[warn] 技能作答失败，回退规则模板：{type(exc).__name__}: {exc}")
        return None
