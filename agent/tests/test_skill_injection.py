"""技能路由 / 加载 / 注入 / 图节点的测试。"""

from __future__ import annotations

import yaml

from agent.graph import nodes as nodes_module
from agent.graph.router import route_after_skill, route_with_skills
from agent.skills.compose import build_messages, compose_answer
from agent.skills.loader import (
    DEFAULT_MAX_TOKENS,
    TRUNCATED_MARK,
    estimate_tokens,
    load_skill,
)
from agent.skills.prompt import build_skill_prompt
from agent.skills.registry import get_registry, load_registry
from agent.skills.router import select_skill
from agent.trace import build_turn_record


def test_selects_expected_skill_for_domain_questions():
    cases = {
        "车厘子怎么保存": "storage-guide",
        "退款一般几天到账": "refund-policy",
        "坏了一半怎么赔": "compensation",
        "三天了怎么还不发货": "logistics-sla",
        "香蕉放冰箱可以吗": "storage-guide",
    }
    for query, expected in cases.items():
        match = select_skill(query)
        assert match is not None, query
        assert match.skill.name == expected, f"{query} -> {match.skill.name}: {match.reason}"
        assert match.score >= 3.0


def test_unrelated_or_empty_text_selects_nothing():
    assert select_skill("") is None
    assert select_skill("你好") is None
    assert select_skill("今天天气不错") is None


def _write_skill(root, dirname: str, meta: dict) -> None:
    directory = root / dirname
    directory.mkdir(parents=True)
    text = "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + "---\n\n## 正文\n按手册处理。"
    (directory / "SKILL.md").write_text(text, encoding="utf-8")


def test_ambiguous_query_is_skipped(tmp_path):
    """两个技能同分时不注入：宁可没有技能，也不要把错的手册塞进去。"""
    _write_skill(
        tmp_path,
        "alpha-skill",
        meta={"name": "alpha-skill", "description": "alpha 测试技能说明",
              "when_to_use": "测试时", "triggers": ["退款"], "references": [], "tags": ["t"]},
    )
    _write_skill(
        tmp_path,
        "beta-skill",
        meta={"name": "beta-skill", "description": "beta 测试技能说明",
              "when_to_use": "测试时", "triggers": ["退款"], "references": [], "tags": ["t"]},
    )
    registry = load_registry(tmp_path)
    assert select_skill("我要退款", registry=registry) is None


def test_load_skill_truncates_to_token_budget():
    skill = get_registry().require("storage-guide")
    full = load_skill(skill, max_tokens=DEFAULT_MAX_TOKENS)
    assert full.truncated is False
    assert full.tokens > 0

    small = load_skill(skill, max_tokens=40)
    assert small.truncated is True
    assert small.body.endswith(TRUNCATED_MARK)
    assert small.tokens < full.tokens
    assert estimate_tokens("") == 0


def test_build_skill_prompt_has_header_and_body():
    loaded = load_skill("storage-guide")
    prompt = build_skill_prompt(loaded)
    assert prompt.startswith("【技能：storage-guide v0.1.0】")
    assert "适用：" in prompt
    assert "冷藏" in prompt


def test_skill_node_sets_state_and_trace():
    state = {
        "user_input": "车厘子怎么保存",
        "intent": "quality_qa",
        "out_of_domain": False,
        "decision": "accept",
        "branches": [],
    }
    update = nodes_module.skill_node(state)
    assert update["skill_name"] == "storage-guide"
    assert update["skill_version"] == "0.1.0"
    assert update["skill_tokens"] > 0
    assert update["skill_reason"].startswith("命中触发词")
    assert "【技能：storage-guide" in update["skill_prompt"]

    record = build_turn_record(
        {**state, **update, "reply": "冷藏 0-4 度"},
        session_id="s1",
        user_input=state["user_input"],
        turn=1,
    )
    assert record["skill"]["name"] == "storage-guide"
    assert record["skill"]["tokens"] > 0
    assert record["skill"]["prompt_chars"] > 0


def test_skill_node_never_breaks_the_turn(monkeypatch):
    import agent.skills.router as router_module

    def boom(*args, **kwargs):
        raise RuntimeError("注册表坏了")

    monkeypatch.setattr(router_module, "select_skill", boom)
    assert nodes_module.skill_node({"user_input": "车厘子怎么保存"}) == {}


def test_route_goes_through_skill_only_for_faq_and_fallback(monkeypatch):
    monkeypatch.setenv("SKILLS_ENABLED", "on")
    assert route_with_skills({"intent": "quality_qa"}) == "skill"
    assert route_with_skills({"intent": "chitchat"}) == "skill"
    assert route_with_skills({"intent": "query_order"}) == "order"
    assert route_after_skill({"intent": "quality_qa"}) == "faq"
    assert route_after_skill({"intent": "chitchat"}) == "fallback"

    monkeypatch.setenv("SKILLS_ENABLED", "off")
    assert route_with_skills({"intent": "quality_qa"}) == "faq"
    assert route_with_skills({"intent": "chitchat"}) == "fallback"


def test_compose_answer_respects_mode_and_completer(monkeypatch):
    messages = build_messages("车厘子怎么保存", "【技能：x】正文", [])
    assert messages[0]["role"] == "system"
    assert "【技能：x】正文" in messages[0]["content"]
    assert "用户问题：车厘子怎么保存" in messages[1]["content"]

    monkeypatch.setenv("SKILL_ANSWER", "off")
    assert compose_answer("q", "prompt", [], completer=lambda _: "答案") is None

    monkeypatch.setenv("SKILL_ANSWER", "llm")
    assert compose_answer("q", "prompt", [], completer=lambda _: "  答案  ") == "答案"
    assert compose_answer("q", "", [], completer=lambda _: "答案") is None
