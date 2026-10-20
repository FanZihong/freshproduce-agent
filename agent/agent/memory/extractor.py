"""事实抽取器：把"新对话"变成"旧记忆里没有的新事实"。

Mem0 的关键不在"存"，而在**只存增量**：先把已有记忆喂给模型，让它只输出
新增/更正的部分，否则每次都会把旧信息重新抽一遍，库越长越脏。

三种动作：add（全新）、update（旧事实被更正，带 target_id）、skip（已经知道）。
LLM 不可用时退回规则版：拿现有 summary 当一条事实，保证链路能跑通、可复现。
"""

from __future__ import annotations

import json
import time
from typing import Callable

from agent.config import get_env
from agent.memory.summarizer import estimate_cost, load_summary_config

ACTIONS = ("add", "update", "skip")
CATEGORIES = ("claim", "order", "policy", "preference", "summary", "other")

EXTRACT_SYSTEM_PROMPT = (
    "你是客服对话的记忆抽取器。给你「已有记忆」和「新增对话」，"
    "只输出**已有记忆里没有的新信息**；已有记忆里已经有的不要再输出。\n\n"
    "每条事实一句话，聚焦：用户是谁、买了什么、发生了什么、诉求是什么、事情办到哪一步。\n"
    "不要臆造对话里没出现的数字或结论；拿不准就不写。\n\n"
    "动作只能是三种：add（全新）、update（旧事实被更正或补充，需给出 target_id）、"
    "skip（已经知道，不必重复）。\n\n"
    "只输出 JSON 数组，不要解释、不要 Markdown。格式：\n"
    '[{"fact": "一句话事实", "category": "claim|order|policy|preference|other", '
    '"entities": ["订单号或商品名"], "action": "add|update|skip", '
    '"target_id": "更新时填旧事实 id"}]'
)

_COMPLETER: Callable[[list[dict]], object] | None = None


def set_extractor_completer(fn: Callable[[list[dict]], object] | None) -> None:
    """替换真实模型调用（测试与 --fake-llm 演练用）。fn(messages) -> str 或 (str, usage)。"""
    global _COMPLETER
    _COMPLETER = fn


def turns_from_messages(messages) -> list[dict]:
    """把 langchain 消息折成 [{"user": ..., "reply": ...}]，忽略工具/系统消息。"""
    turns: list[dict] = []
    for message in messages or []:
        kind = str(getattr(message, "type", "") or "")
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        if kind == "human":
            turns.append({"user": content, "reply": ""})
        elif kind == "ai" and turns and not turns[-1].get("reply"):
            turns[-1]["reply"] = content
    return turns


def build_extract_messages(turns: list[dict], old_facts: list[dict]) -> list[dict]:
    lines: list[str] = []
    if old_facts:
        lines.append("已有记忆：")
        for item in old_facts[-30:]:
            lines.append(f"- id={item['id']}｜{item['fact']}")
    else:
        lines.append("已有记忆：（空）")
    lines.append("")
    lines.append("新增对话：")
    for turn in turns[-10:]:
        lines.append(f"用户：{turn.get('user') or ''}")
        if turn.get("reply"):
            lines.append(f"客服：{turn['reply']}")
    return [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def postprocess_extraction(raw: str) -> list[dict]:
    """解析模型输出：容错 Markdown 包裹，只保留合法项。"""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"模型未返回 JSON 数组：{text[:80]}")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, list):
        raise ValueError("抽取结果必须是数组")

    candidates: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact") or "").strip()
        if not fact:
            continue
        action = str(item.get("action") or "add").strip().lower()
        if action not in ACTIONS:
            action = "add"
        category = str(item.get("category") or "other").strip().lower()
        if category not in CATEGORIES:
            category = "other"
        entities = [
            str(entity).strip() for entity in (item.get("entities") or []) if str(entity).strip()
        ]
        candidates.append(
            {
                "fact": fact,
                "category": category,
                "entities": entities[:6],
                "action": action,
                "target_id": str(item.get("target_id") or "").strip(),
            }
        )
    return candidates


def _complete(messages: list[dict]) -> tuple[str, dict]:
    if _COMPLETER is not None:
        result = _COMPLETER(messages)
        if isinstance(result, tuple):
            return str(result[0]), dict(result[1] or {})
        return str(result), {}
    from openai import OpenAI  # 延迟导入：没配 key 的机器不装也能跑

    config = load_summary_config()
    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=config.timeout)
    response = client.chat.completions.create(
        model=get_env("MEMORY_EXTRACT_MODEL", "") or config.model,
        messages=messages,
        temperature=0.0,
    )
    usage = getattr(response, "usage", None)
    return response.choices[0].message.content or "", {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def extract_facts(
    turns: list[dict],
    old_facts: list[dict],
    *,
    mode: str = "llm",
    summary: str = "",
) -> tuple[list[dict], dict]:
    """抽取候选事实；返回 (candidates, meta)，meta 里有耗时、成本与回退标记。"""
    meta = {"mode": mode, "calls": 0, "cost": 0.0, "latency_ms": 0.0, "fallback": False}
    if mode == "llm" and (_COMPLETER is not None or load_summary_config().configured):
        started = time.perf_counter()
        meta["calls"] = 1
        try:
            raw, usage = _complete(build_extract_messages(turns, old_facts))
            candidates = postprocess_extraction(raw)
            meta["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            meta["cost"] = estimate_cost(
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
            )
            return candidates, meta
        except Exception as exc:  # noqa: BLE001 - 抽取失败按规则版继续
            print(f"[warn] 记忆抽取失败，回退规则版：{type(exc).__name__}: {exc}")
            meta["fallback"] = True

    text = str(summary or "").strip()
    if text:
        return (
            [
                {
                    "fact": text[:160],
                    "category": "summary",
                    "entities": [],
                    "action": "add",
                    "target_id": "",
                }
            ],
            meta,
        )
    return [], meta
