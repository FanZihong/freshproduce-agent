"""技能正文加载：按 token 预算截断，保证一个包里再长也不会吃光上下文。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.config import get_env
from agent.skills.registry import Skill, SkillRegistry, get_registry

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
DEFAULT_MAX_TOKENS = 600
TRUNCATED_MARK = "\n…（技能正文已截断）"


@dataclass(frozen=True)
class LoadedSkill:
    """加载结果：截断后的正文 + 估算 token，trace/报告直接用。"""

    skill: Skill
    body: str
    truncated: bool
    tokens: int


def estimate_tokens(text: str) -> int:
    """粗略估算：中文按 0.75 token/字，其他字符按 4 字符 ≈ 1 token。

    不引 tokenizer 是有意的：只用来做预算和截断，不需要精确到个位；
    真接模型时再换成对应 tokenizer。
    """
    if not text:
        return 0
    cjk = len(CJK_RE.findall(text))
    other = len(text) - cjk
    return int(round(cjk * 0.75 + other / 4))


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """二分找最长前缀，再尽量切在段落/行边界上。"""
    if max_tokens <= 0 or estimate_tokens(text) <= max_tokens:
        return text, False

    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    cut = text[:low]
    boundary = max(cut.rfind("\n\n"), cut.rfind("\n"))
    if boundary >= len(cut) * 0.6:
        cut = cut[:boundary]
    return cut.rstrip() + TRUNCATED_MARK, True


def default_max_tokens() -> int:
    raw = get_env("SKILL_MAX_TOKENS", "") or str(DEFAULT_MAX_TOKENS)
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_MAX_TOKENS
    return value if value > 0 else DEFAULT_MAX_TOKENS


def load_skill(
    skill: Skill | str,
    *,
    max_tokens: int | None = None,
    registry: SkillRegistry | None = None,
) -> LoadedSkill:
    """按名字或 `Skill` 对象加载正文；超过预算按 token 截断。"""
    resolved = skill if isinstance(skill, Skill) else (registry or get_registry()).require(skill)
    budget = default_max_tokens() if max_tokens is None else int(max_tokens)
    body, truncated = truncate_to_tokens(resolved.body, budget)
    return LoadedSkill(
        skill=resolved,
        body=body,
        truncated=truncated,
        tokens=estimate_tokens(body),
    )
