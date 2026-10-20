"""把技能正文拼成可注入的上下文块（不含用户可见文案）。"""

from __future__ import annotations

from agent.skills.loader import LoadedSkill


def build_skill_prompt(loaded: LoadedSkill) -> str:
    """技能块：头部给模型看名字/版本，正文是处理手册。"""
    skill = loaded.skill
    lines = [
        f"【技能：{skill.name} v{skill.version}】",
        f"适用：{skill.when_to_use}",
        "",
        loaded.body.strip(),
    ]
    if loaded.truncated:
        lines.append("\n（正文超出预算，已截断；需要细节请转人工核实。）")
    return "\n".join(lines).strip()
