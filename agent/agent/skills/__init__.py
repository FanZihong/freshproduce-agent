"""技能包：目录常驻、正文按需加载（渐进式披露）。"""

from agent.skills.compose import answer_mode, compose_answer
from agent.skills.loader import LoadedSkill, estimate_tokens, load_skill, truncate_to_tokens
from agent.skills.prompt import build_skill_prompt
from agent.skills.references import (
    ReferenceDoc,
    SkillReferenceError,
    build_reference_prompt,
    load_reference,
    load_references,
    select_references,
    wants_details,
)
from agent.skills.registry import (
    Skill,
    SkillError,
    SkillRegistry,
    default_skills_dir,
    get_registry,
    load_registry,
    parse_skill,
    reset_registry,
)
from agent.skills.router import SkillMatch, select_skill

__all__ = [
    "LoadedSkill",
    "ReferenceDoc",
    "SkillMatch",
    "Skill",
    "SkillError",
    "SkillReferenceError",
    "SkillRegistry",
    "answer_mode",
    "build_skill_prompt",
    "compose_answer",
    "build_reference_prompt",
    "default_skills_dir",
    "estimate_tokens",
    "get_registry",
    "load_skill",
    "load_registry",
    "load_reference",
    "load_references",
    "parse_skill",
    "reset_registry",
    "select_skill",
    "select_references",
    "truncate_to_tokens",
    "wants_details",
]
