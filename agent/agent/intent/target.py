"""意图标签的"目标串"：训练时怎么序列化，推理时怎么解析回标签。

三种序列化风格（训练前选定，只影响训练数据长什么样）：

    key ： 主:bad_fruit_claim|辅:urge_logistics
    name： 主:坏果理赔|辅:催物流
    json： {"labels": ["bad_fruit_claim", "urge_logistics"], "confidence": 0.95}

解析侧故意写得比序列化侧宽松：小模型实际输出会漂移（多空格、用逗号、漏前缀、
偶尔吐 JSON、把 key 写成中文名），能认出来就该算它答对，而不是记成解析失败。
所以 `parse_target` 不按风格严格校验，风格只用来生成训练数据。
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

TARGET_STYLES: tuple[str, ...] = ("key", "name", "json")
PRIMARY_PREFIX = "主:"
AUX_PREFIX = "辅:"

# 解析时要去掉的前缀与包裹字符
_PREFIXES: tuple[str, ...] = ("主意图", "辅助意图", "主", "辅", "意图", "标签", "label", "labels")
_TRIM_CHARS = " \t\r\n\"'“”‘’`{}[]（）()：:。，,、|/;；"
_SEPARATORS = re.compile(r"[|/;；\n\t]+|,(?![^{}]*\})")


def _aliases() -> dict[str, str]:
    from agent.intent.taxonomy import ALIASES

    return ALIASES


def normalize_labels(labels: Sequence[str], valid: set[str] | None = None) -> tuple[str, ...]:
    """去别名、去重、保持顺序；给了 valid 就顺带剔除体系外的标签。"""
    aliases = _aliases()
    result: list[str] = []
    for item in labels or ():
        candidate = aliases.get(str(item).strip(), str(item).strip())
        if not candidate or candidate in result:
            continue
        if valid is not None and candidate not in valid:
            continue
        result.append(candidate)
    return tuple(result)


def _name_to_key(taxonomy: Any) -> dict[str, str]:
    if taxonomy is None:
        return {}
    return {spec.name: spec.key for spec in taxonomy.labels}


def format_target(
    labels: Sequence[str],
    style: str = "key",
    taxonomy: Any = None,
    max_labels: int = 2,
    confidence: float | None = None,
) -> str:
    """把标签序列渲染成训练目标串。"""
    if style not in TARGET_STYLES:
        raise ValueError(f"不支持的风格：{style}（可选 {'/'.join(TARGET_STYLES)}）")
    clean = normalize_labels(labels)[:max_labels]
    if not clean:
        raise ValueError("没有可写入目标的标签")

    if style == "json":
        payload: dict[str, Any] = {"labels": list(clean)}
        if confidence is not None:
            payload["confidence"] = round(float(confidence), 2)
        return json.dumps(payload, ensure_ascii=False)

    if style == "name":
        if taxonomy is None:
            raise ValueError("name 风格需要 taxonomy 才能把 key 翻成中文名")
        show = [taxonomy.name_of(label) for label in clean]
    else:
        show = list(clean)

    parts = [f"{PRIMARY_PREFIX}{show[0]}"]
    parts.extend(f"{AUX_PREFIX}{item}" for item in show[1:])
    return "|".join(parts)


def output_contract(style: str = "key", max_labels: int = 2, taxonomy: Any = None) -> str:
    """指令里说明"该输出什么格式"，并用纯格式示例（不涉及语义）说清楚。"""
    if style == "json":
        body = '{"labels": ["类别key"], "confidence": 0.0}'
        return (
            f"输出：只输出一个 JSON 对象，不要解释、不要代码块，最多 {max_labels} 个标签，"
            f"格式为 {body}"
        )
    if style == "name":
        example_single = f"{PRIMARY_PREFIX}坏果理赔"
        example_multi = f"{PRIMARY_PREFIX}坏果理赔|{AUX_PREFIX}催物流"
        unit = "类别中文名"
    else:
        example_single = f"{PRIMARY_PREFIX}bad_fruit_claim"
        example_multi = f"{PRIMARY_PREFIX}bad_fruit_claim|{AUX_PREFIX}urge_logistics"
        unit = "类别 key"

    lines = [
        f"输出：不要解释、不要引号、不要代码块，直接给出 {unit}。",
        f"- 单意图：{example_single}",
        f"- 多意图（最多 {max_labels} 个，第 1 个是主意图）：{example_multi}",
    ]
    return "\n".join(lines)


def _resolve(token: str, valid: set[str], names: dict[str, str]) -> str | None:
    candidate = token.strip().strip(_TRIM_CHARS).strip()
    if not candidate:
        return None
    aliases = _aliases()
    for prefix in _PREFIXES:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :].strip().strip(_TRIM_CHARS).strip()
    candidate = aliases.get(candidate, candidate)
    if candidate in valid:
        return candidate
    if candidate in names:  # 中文名 -> key
        return names[candidate]
    return None


def _parse_json(raw: str, valid: set[str], max_labels: int) -> tuple[tuple[str, ...], float, str]:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("没有 JSON 对象")
    payload = json.loads(raw[start : end + 1])
    raw_labels = payload.get("labels")
    if raw_labels is None:
        raw_labels = [payload["label"]] if payload.get("label") else []
    if isinstance(raw_labels, str):
        raw_labels = [raw_labels]
    labels = normalize_labels(raw_labels, valid)[:max_labels]
    if not labels:
        raise ValueError(f"JSON 里没有合法标签：{raw_labels}")
    confidence = max(0.0, min(1.0, float(payload.get("confidence") or 0.0)))
    reason = str(payload.get("reason") or "")[:120]
    return labels, confidence, reason


def parse_target(
    text: str,
    valid_labels: Sequence[str] | set[str],
    taxonomy: Any = None,
    max_labels: int = 2,
) -> tuple[tuple[str, ...], float, str]:
    """把模型输出解析成 (labels, confidence, reason)；解析不出合法标签时抛 ValueError。

    优先按 JSON 解析，失败就退化成"扫 token"：容忍 `主:`/`辅:` 前缀、逗号/斜杠/竖线
    分隔、中文名、旧别名。confidence 只有 JSON 风格才可能非 0。
    """
    valid = set(valid_labels)
    raw = (text or "").strip()
    if not raw:
        raise ValueError("模型返回为空")
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:]

    if "{" in raw and "}" in raw:
        try:
            return _parse_json(raw, valid, max_labels)
        except (ValueError, json.JSONDecodeError):
            pass  # 落回文本扫描

    names = _name_to_key(taxonomy)
    labels: list[str] = []
    for chunk in _SEPARATORS.split(raw):
        for token in re.split(r"[，,、\s]+", chunk):
            if not token.strip():
                continue
            resolved = _resolve(token, valid, names)
            if resolved and resolved not in labels:
                labels.append(resolved)
    if not labels:
        raise ValueError(f"没有合法标签：{raw[:60]}")
    return tuple(labels[:max_labels]), 0.0, ""
