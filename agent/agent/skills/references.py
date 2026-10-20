"""references 二层加载：只有 SKILL.md 声明过的文件才允许读。

这是渐进式披露的第三层：

- 常驻上下文：技能目录（name/description）；
- 命中技能：`SKILL.md` 正文；
- 用户追问细节（"具体第几条""原文怎么规定"）：才读 `references/`。

安全边界（三个都必须过）：
1. **白名单**：文件名必须出现在 `Skill.references` 里，不能现场拼路径；
2. **路径校验**：拒绝绝对路径和 `../`，`resolve()` 后必须仍在技能目录内；
3. **大小限制**：单文件 ≤ 256 KB，技能内合计 ≤ 512 KB，注入时再按 token 截断。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.config import get_env
from agent.rag.hybrid_search import tokenize
from agent.skills.loader import estimate_tokens, truncate_to_tokens
from agent.skills.registry import (
    MAX_REFERENCE_BYTES,
    Skill,
    SkillError,
)

DETAIL_KEYWORDS: tuple[str, ...] = (
    "具体",
    "第几条",
    "细则",
    "原文",
    "条款",
    "规定",
    "标准表",
    "依据",
    "详细",
    "怎么写的",
)
DEFAULT_DETAIL_TOKENS = 800
DEFAULT_MAX_REFERENCES = 2


class SkillReferenceError(SkillError):
    """细则加载被拒绝（未声明 / 越界 / 超限）。"""


@dataclass(frozen=True)
class ReferenceDoc:
    name: str
    text: str
    tokens: int
    truncated: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tokens": self.tokens,
            "truncated": self.truncated,
        }


def wants_details(text: str) -> bool:
    """用户是否在追问细节——只有这时才值得多花一次文件读取和上下文预算。"""
    content = str(text or "")
    return any(keyword in content for keyword in DETAIL_KEYWORDS)


def _resolve_reference(skill: Skill, name: str) -> Path:
    entry = str(name or "").strip()
    if not entry:
        raise SkillReferenceError("细则文件名不能为空")
    if entry not in skill.references:
        raise SkillReferenceError(
            f"细则未在 SKILL.md 中声明，拒绝读取：{entry}"
        )
    candidate = Path(entry)
    if candidate.is_absolute():
        raise SkillReferenceError(f"细则不能是绝对路径：{entry}")
    root = skill.directory.resolve()
    references_root = (skill.directory / "references").resolve()
    resolved = (skill.directory / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise SkillReferenceError(f"细则越界（禁止 ../）：{entry}")
    if resolved != references_root and references_root not in resolved.parents:
        raise SkillReferenceError(f"细则必须放在 references/ 目录下：{entry}")
    if not resolved.is_file():
        raise SkillReferenceError(f"细则文件不存在：{entry}")
    size = resolved.stat().st_size
    if size > MAX_REFERENCE_BYTES:
        raise SkillReferenceError(
            f"细则文件超过 {MAX_REFERENCE_BYTES} 字节：{entry}（{size}）"
        )
    return resolved


def default_detail_tokens() -> int:
    raw = get_env("SKILL_DETAIL_TOKENS", "") or str(DEFAULT_DETAIL_TOKENS)
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_DETAIL_TOKENS
    return value if value > 0 else DEFAULT_DETAIL_TOKENS


def default_max_references() -> int:
    raw = get_env("SKILL_MAX_REFERENCES", "") or str(DEFAULT_MAX_REFERENCES)
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_MAX_REFERENCES
    return max(1, value)


def select_references(
    skill: Skill,
    query: str,
    *,
    limit: int | None = None,
) -> list[str]:
    """在已声明的细则里挑最相关的几条；不追问细节就返回空。"""
    if not wants_details(query) or not skill.references:
        return []
    size = default_max_references() if limit is None else max(1, int(limit))
    query_tokens = set(tokenize(query))
    scored: list[tuple[int, int, str]] = []
    for order, entry in enumerate(skill.references):
        try:
            # 即使注册表已经校验过，这里也走同一条白名单 + 越界检查，防"二次拼路径"
            path = _resolve_reference(skill, entry)
            snippet = path.read_text(encoding="utf-8-sig")[:400]
        except (SkillReferenceError, OSError):
            snippet = ""
        overlap = len(query_tokens & set(tokenize(f"{entry} {snippet}")))
        scored.append((-overlap, order, entry))
    scored.sort()
    return [entry for _, _, entry in scored[:size]]


def load_reference(
    skill: Skill,
    name: str,
    *,
    max_tokens: int | None = None,
) -> ReferenceDoc:
    path = _resolve_reference(skill, name)
    text = path.read_text(encoding="utf-8-sig")
    budget = default_detail_tokens() if max_tokens is None else int(max_tokens)
    body, truncated = truncate_to_tokens(text, budget)
    return ReferenceDoc(
        name=str(name).strip(),
        text=body,
        tokens=estimate_tokens(body),
        truncated=truncated,
    )


def load_references(
    skill: Skill,
    names: list[str],
    *,
    max_tokens: int | None = None,
) -> list[ReferenceDoc]:
    return [load_reference(skill, name, max_tokens=max_tokens) for name in names]


def build_reference_prompt(skill: Skill, docs: list[ReferenceDoc]) -> str:
    if not docs:
        return ""
    lines = [f"【技能细则：{skill.name}】"]
    for doc in docs:
        lines.extend(["", f"### {doc.name}", doc.text.strip()])
        if doc.truncated:
            lines.append("（细则超出预算，已截断；需要完整条款请转人工核实。）")
    return "\n".join(lines).strip()
