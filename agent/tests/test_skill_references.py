"""references 二层加载：白名单、路径逃逸、大小限制、按需触发。"""

from __future__ import annotations

from dataclasses import replace

import pytest
import yaml

from agent.graph import nodes as nodes_module
from agent.skills import references as refs_module
from agent.skills import registry as registry_module
from agent.skills.references import (
    SkillReferenceError,
    build_reference_prompt,
    load_reference,
    select_references,
    wants_details,
)
from agent.skills.registry import get_registry, load_registry
from agent.trace import build_turn_record


def _write_skill(root, dirname: str, refs: list[tuple[str, str]], *, name: str | None = None):
    directory = root / dirname
    directory.mkdir(parents=True)
    meta = {
        "name": name or dirname,
        "description": "用于测试细则加载的技能说明",
        "when_to_use": "测试时使用",
        "triggers": ["测试"],
        "references": [entry for entry, _ in refs],
    }
    text = "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + "---\n\n## 正文\n按手册处理。"
    (directory / "SKILL.md").write_text(text, encoding="utf-8")
    for entry, content in refs:
        target = directory / entry
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return directory


def test_declared_reference_loads_and_builds_prompt():
    skill = get_registry().require("refund-policy")
    doc = load_reference(skill, "references/refund-policy.md")
    assert doc.tokens > 0
    assert "1 至 3 个工作日" in doc.text
    assert doc.truncated is False

    block = build_reference_prompt(skill, [doc])
    assert block.startswith("【技能细则：refund-policy】")
    assert "### references/refund-policy.md" in block


def test_undeclared_reference_is_rejected():
    skill = get_registry().require("refund-policy")
    with pytest.raises(SkillReferenceError, match="未在 SKILL.md 中声明"):
        load_reference(skill, "references/not-declared.md")


def test_reference_outside_skill_dir_is_rejected():
    skill = get_registry().require("refund-policy")
    bad = replace(skill, references=("../../secret.md",))
    with pytest.raises(SkillReferenceError, match="越界"):
        load_reference(bad, "../../secret.md")


def test_reference_must_live_under_references_dir():
    skill = get_registry().require("refund-policy")
    bad = replace(skill, references=("references/../secret.md",))
    with pytest.raises(SkillReferenceError, match="references/ 目录"):
        load_reference(bad, "references/../secret.md")


def test_absolute_reference_is_rejected():
    skill = get_registry().require("refund-policy")
    bad = replace(skill, references=("/etc/passwd",))
    with pytest.raises(SkillReferenceError, match="绝对路径"):
        load_reference(bad, "/etc/passwd")


def test_oversized_reference_is_rejected(monkeypatch):
    monkeypatch.setattr(refs_module, "MAX_REFERENCE_BYTES", 10)
    skill = get_registry().require("refund-policy")
    with pytest.raises(SkillReferenceError, match="超过"):
        load_reference(skill, "references/refund-policy.md")


def test_registry_rejects_too_many_references(tmp_path):
    refs = [(f"references/r{i}.md", "内容") for i in range(registry_module.MAX_REFERENCES + 1)]
    _write_skill(tmp_path, "demo-skill", refs)
    registry = load_registry(tmp_path)
    assert "最多" in registry.errors[0]


def test_registry_rejects_oversized_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(registry_module, "MAX_REFERENCE_BYTES", 4)
    _write_skill(tmp_path, "demo-skill", [("references/big.md", "123456789")])
    registry = load_registry(tmp_path)
    assert "超过" in registry.errors[0]


def test_wants_details_only_for_follow_up_questions():
    assert wants_details("退款具体第几条")
    assert wants_details("政策原文怎么规定的")
    assert not wants_details("退款一般几天到账")
    assert not wants_details("车厘子怎么保存")


def test_select_references_respects_details_gate():
    skill = get_registry().require("refund-policy")
    assert select_references(skill, "退款多久到账") == []
    assert select_references(skill, "退款具体第几条") == ["references/refund-policy.md"]


def test_skill_node_loads_details_only_when_asked():
    plain = nodes_module.skill_node({"user_input": "退款一般几天到账"})
    assert plain["skill_name"] == "refund-policy"
    assert plain["skill_references"] == []
    assert "【技能细则" not in plain["skill_prompt"]

    detailed = nodes_module.skill_node({"user_input": "退款一般几天到账，具体第几条"})
    assert detailed["skill_name"] == "refund-policy"
    assert detailed["skill_references"][0]["name"] == "references/refund-policy.md"
    assert "【技能细则：refund-policy】" in detailed["skill_prompt"]


def test_trace_records_reference_details():
    update = nodes_module.skill_node({"user_input": "退款一般几天到账，具体第几条"})
    record = build_turn_record(
        {**update, "reply": "1 至 3 个工作日"},
        session_id="s1",
        user_input="退款一般几天到账，具体第几条",
        turn=1,
    )
    assert record["skill"]["references"][0]["name"] == "references/refund-policy.md"
    assert record["skill"]["references"][0]["tokens"] > 0
