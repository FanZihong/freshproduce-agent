"""Skill 包与注册表的测试：真实目录能加载、坏包被拒、路径越界被挡。"""

from __future__ import annotations

import pytest
import yaml

from agent.skills import SkillError, get_registry, load_registry, reset_registry
from agent.skills import registry as registry_module

DEFAULT_META = {
    "name": "demo-skill",
    "description": "用于单元测试的技能包说明",
    "when_to_use": "测试时使用",
    "version": "0.1.0",
    "tags": ["test"],
    "references": [],
}


def write_skill(
    root,
    dirname: str = "demo-skill",
    *,
    meta: dict | None = None,
    raw_meta: str | None = None,
    body: str = "## 处理步骤\n1. 先确认订单号。",
    files: dict[str, str] | None = None,
):
    directory = root / dirname
    directory.mkdir(parents=True)
    values = dict(DEFAULT_META if meta is None else meta)
    values["name"] = values.get("name", dirname)
    if raw_meta is not None:
        text = f"---\n{raw_meta}\n---\n\n{body}"
    else:
        text = "---\n" + yaml.safe_dump(values, allow_unicode=True, sort_keys=False) + "---\n\n" + body
    (directory / "SKILL.md").write_text(text, encoding="utf-8")
    for relative, content in (files or {}).items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return directory


def test_real_registry_loads_metadata_only():
    registry = load_registry()
    assert registry.errors == []
    assert registry.names() == [
        "compensation",
        "logistics-sla",
        "refund-policy",
        "storage-guide",
    ]
    entry = registry.index()[0]
    assert "body" not in entry                  # 目录里不塞正文
    assert entry["chars"] > 0
    assert set(entry) >= {"name", "description", "when_to_use", "version", "tags"}
    skill = registry.require("refund-policy")
    assert "1 至 3 个工作日" in skill.body
    assert "退款" in skill.triggers
    assert skill.directory.name == "refund-policy"


def test_missing_frontmatter_is_rejected(tmp_path):
    directory = tmp_path / "demo-skill"
    directory.mkdir()
    (directory / "SKILL.md").write_text("# 只有正文\n没有元数据", encoding="utf-8")
    registry = load_registry(tmp_path)
    assert len(registry) == 0
    assert "frontmatter" in registry.errors[0]


def test_missing_required_field_is_rejected(tmp_path):
    meta = {key: value for key, value in DEFAULT_META.items() if key != "when_to_use"}
    write_skill(tmp_path, meta=meta)
    registry = load_registry(tmp_path)
    assert len(registry) == 0
    assert "when_to_use" in registry.errors[0]


def test_directory_name_must_match_frontmatter(tmp_path):
    write_skill(tmp_path, "wrong-dir", meta={**DEFAULT_META, "name": "demo-skill"})
    registry = load_registry(tmp_path)
    assert "必须和 frontmatter 的 name" in registry.errors[0]


def test_invalid_name_is_rejected(tmp_path):
    meta = {**DEFAULT_META, "name": "Refund Policy"}
    write_skill(tmp_path, "Refund Policy", meta=meta)
    registry = load_registry(tmp_path)
    assert "kebab-case" in registry.errors[0]


def test_empty_body_is_rejected(tmp_path):
    write_skill(tmp_path, body="   \n  ")
    registry = load_registry(tmp_path)
    assert "正文为空" in registry.errors[0]


def test_oversize_body_is_rejected(tmp_path):
    write_skill(tmp_path, body="x" * (registry_module.MAX_BODY_CHARS + 1))
    registry = load_registry(tmp_path)
    assert "正文超过" in registry.errors[0]


def test_reference_outside_skill_dir_is_rejected(tmp_path):
    (tmp_path / "secret.md").write_text("不该被读到", encoding="utf-8")
    write_skill(tmp_path, meta={**DEFAULT_META, "references": ["../secret.md"]})
    registry = load_registry(tmp_path)
    assert "越界" in registry.errors[0]


def test_missing_reference_is_rejected(tmp_path):
    write_skill(tmp_path, meta={**DEFAULT_META, "references": ["references/none.md"]})
    registry = load_registry(tmp_path)
    assert "不存在" in registry.errors[0]


def test_reference_must_live_under_references_dir(tmp_path):
    write_skill(
        tmp_path,
        meta={**DEFAULT_META, "references": ["policy.md"]},
        files={"policy.md": "放在根目录的细则"},
    )
    registry = load_registry(tmp_path)
    assert "references/ 目录" in registry.errors[0]


def test_valid_reference_is_accepted(tmp_path):
    write_skill(
        tmp_path,
        meta={**DEFAULT_META, "references": ["references/policy.md"]},
        files={"references/policy.md": "政策原文"},
    )
    registry = load_registry(tmp_path)
    assert registry.errors == []
    assert registry.require("demo-skill").references == ("references/policy.md",)


def test_triggers_must_be_string_list(tmp_path):
    write_skill(tmp_path, meta={**DEFAULT_META, "triggers": "退款"})
    registry = load_registry(tmp_path)
    assert "triggers 必须是字符串列表" in registry.errors[0]


def test_strict_mode_raises_on_any_bad_package(tmp_path):
    write_skill(tmp_path, body="")
    with pytest.raises(SkillError):
        load_registry(tmp_path, strict=True)


def test_skills_dir_env_override(tmp_path, monkeypatch):
    write_skill(tmp_path)
    monkeypatch.setenv("SKILLS_DIR", str(tmp_path))
    registry = load_registry()
    assert registry.names() == ["demo-skill"]


def test_registry_is_cached_and_can_reset(tmp_path, monkeypatch):
    write_skill(tmp_path)
    monkeypatch.setenv("SKILLS_DIR", str(tmp_path))
    reset_registry()
    first = get_registry()
    assert get_registry() is first
    reset_registry()
    assert get_registry() is not first
    reset_registry()
