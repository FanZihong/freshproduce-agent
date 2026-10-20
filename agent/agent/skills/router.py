"""技能路由：用触发词 + 描述重合度选技能，纯规则、可复现。

为什么不全交给模型：
- 只有几个技能时，关键词比一次模型调用更快、更便宜、更好解释；
- trace 里能写清"为什么选它"，出问题能直接改 triggers；
- 等技能多到几十个，再把这一层换成向量召回，接口不变。
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.config import get_float
from agent.rag.hybrid_search import tokenize
from agent.skills.registry import Skill, SkillRegistry, get_registry

DEFAULT_MIN_SCORE = 3.0
DEFAULT_MARGIN = 1.0


@dataclass(frozen=True)
class SkillMatch:
    skill: Skill
    score: float
    reason: str


def _score(text: str, skill: Skill) -> tuple[float, list[str], int]:
    lowered = text.lower()
    hits = [trigger for trigger in skill.triggers if trigger and trigger.lower() in lowered]
    # 长触发词更具体，权重更高（"怎么保存" 比 "保存" 更有辨识度）
    score = sum(2.0 + min(len(trigger), 4) * 0.5 for trigger in hits)
    overlap = len(set(tokenize(text)) & set(tokenize(f"{skill.when_to_use} {skill.description}")))
    score += min(overlap, 6) * 0.5
    return score, hits, overlap


def select_skill(
    text: str,
    *,
    registry: SkillRegistry | None = None,
    min_score: float | None = None,
    margin: float | None = None,
) -> SkillMatch | None:
    """返回最匹配的技能；分数不够或前两名太接近时返回 None（宁可不注入）。"""
    query = str(text or "").strip()
    if not query:
        return None
    threshold = get_float("SKILL_MIN_SCORE", DEFAULT_MIN_SCORE) if min_score is None else min_score
    gap = get_float("SKILL_MARGIN", DEFAULT_MARGIN) if margin is None else margin
    skills = (registry or get_registry()).skills()
    if not skills:
        return None

    scored: list[tuple[float, list[str], int, Skill]] = []
    for skill in skills:
        score, hits, overlap = _score(query, skill)
        scored.append((score, hits, overlap, skill))
    scored.sort(key=lambda item: (-item[0], item[3].name))
    best_score, best_hits, best_overlap, best = scored[0]
    if best_score < threshold:
        return None
    if len(scored) > 1 and best_score - scored[1][0] < gap:
        return None

    if best_hits:
        reason = "命中触发词：" + "/".join(best_hits[:4])
    else:
        reason = f"描述关键词重合 {best_overlap} 个"
    return SkillMatch(skill=best, score=round(best_score, 2), reason=reason)
