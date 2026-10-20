"""Memory 压缩升级的测试：规则折叠 / LLM 摘要 / 失败回退 / 消融脚本的纯逻辑。"""

from __future__ import annotations

import json
from pathlib import Path

from agent.memory import summarizer as summarizer_module
from agent.memory.summarizer import (
    build_summary_messages,
    choose_mode,
    postprocess_summary,
    reset_stats,
    set_completer,
    stats,
    summarize_dialogue,
)
from agent.memory.short_term import compress_messages
from scripts.ablate_memory import (
    aggregate,
    build_padded_row,
    recommendation,
    run_row,
    write_report,
)


class Msg:
    def __init__(self, content: str, *, id: str = "", type: str = "human") -> None:
        self.content = content
        self.id = id
        self.type = type


class RemoveMsg:
    def __init__(self, id: str = "") -> None:
        self.id = id


def patch_env(monkeypatch, **values) -> None:
    """隔离本机 .env：summarizer 里的 get_env 换成受控字典。"""

    def fake_get_env(key: str, default: str = "") -> str:
        return values.get(key, default)

    monkeypatch.setattr(summarizer_module, "get_env", fake_get_env)


def history(count: int = 10) -> list[Msg]:
    return [
        Msg(f"第 {index} 句对话内容", id=f"m{index}", type="human" if index % 2 else "ai")
        for index in range(count)
    ]


# ---------------- 输出清理与 prompt ----------------


def test_postprocess_summary_strips_prefix_and_truncates():
    assert postprocess_summary("摘要：用户要查订单 2026093000123") == "用户要查订单 2026093000123"
    assert postprocess_summary("  「已填订单，待补凭证。」  ") == "已填订单，待补凭证"
    assert postprocess_summary("第一行\n第二行") == "第一行"
    assert postprocess_summary("啊" * 200).endswith("…")
    assert len(postprocess_summary("啊" * 200)) == 121
    assert postprocess_summary("   ") == ""


def test_build_summary_messages_includes_previous_and_turns():
    messages = build_summary_messages(
        [Msg("帮我查订单", type="human"), Msg("已建工单 TK-1", type="ai")],
        "旧摘要：已填订单",
    )
    assert messages[0]["role"] == "system" and "摘要" in messages[0]["content"]
    user = messages[1]["content"]
    assert "旧摘要：已填订单" in user
    assert "用户：帮我查订单" in user
    assert "Agent：已建工单 TK-1" in user


def test_build_summary_messages_truncates_long_content():
    messages = build_summary_messages([Msg("超" * 500, type="human")])
    assert "超" * 120 + "..." in messages[1]["content"]


# ---------------- 模式选择 ----------------


def test_choose_mode_defaults_to_rule(monkeypatch):
    patch_env(monkeypatch)
    set_completer(None)
    assert choose_mode() == "rule"


def test_choose_mode_llm_requires_key(monkeypatch):
    patch_env(monkeypatch, MEMORY_SUMMARY="llm")
    set_completer(None)
    assert choose_mode() == "rule"          # 没 key -> 回落规则，不静默失败


def test_choose_mode_llm_with_key_or_completer(monkeypatch):
    patch_env(monkeypatch, MEMORY_SUMMARY="llm", LLM_API_KEY="sk-test")
    set_completer(None)
    assert choose_mode() == "llm"

    patch_env(monkeypatch)                  # 模式没配，但注入了 completer（单测/演练）
    set_completer(lambda messages: "ok")
    try:
        assert choose_mode() == "rule"      # 只有 completer 不改默认模式
    finally:
        set_completer(None)


def test_choose_mode_auto_and_invalid(monkeypatch):
    patch_env(monkeypatch, MEMORY_SUMMARY="auto", LLM_API_KEY="sk-test")
    set_completer(None)
    assert choose_mode() == "llm"
    patch_env(monkeypatch, MEMORY_SUMMARY="auto")
    assert choose_mode() == "rule"
    patch_env(monkeypatch, MEMORY_SUMMARY="随便写的")
    assert choose_mode() == "rule"


# ---------------- 摘要与回退 ----------------


def test_summarize_dialogue_rule_mode_does_not_replace(monkeypatch):
    patch_env(monkeypatch, MEMORY_SUMMARY="rule")
    set_completer(None)
    result = summarize_dialogue(history(4), "旧摘要")
    assert result.mode == "rule" and result.replaces is False
    assert "第 0 句对话内容" in result.text


def test_summarize_dialogue_llm_mode_replaces_and_records_cost(monkeypatch):
    patch_env(monkeypatch, MEMORY_SUMMARY="llm", LLM_API_KEY="sk-test")
    set_completer(lambda messages: ("LLM 摘要正文", {"prompt_tokens": 200, "completion_tokens": 40}))
    reset_stats()
    try:
        result = summarize_dialogue(history(6), "旧摘要")
    finally:
        set_completer(None)
    assert result.mode == "llm" and result.replaces is True
    assert result.text == "LLM 摘要正文"
    assert result.meta["completion_tokens"] == 40
    counters = stats()
    assert counters["calls"] == 1 and counters["ok"] == 1
    assert counters["cost"] > 0 and counters["latency_ms"] >= 0


def test_summarize_dialogue_falls_back_when_llm_fails(monkeypatch):
    patch_env(monkeypatch, MEMORY_SUMMARY="llm", LLM_API_KEY="sk-test")

    def broken(messages):
        raise RuntimeError("模型超时")

    set_completer(broken)
    reset_stats()
    try:
        result = summarize_dialogue(history(6), "")
    finally:
        set_completer(None)
    assert result.mode == "rule" and result.meta.get("fallback") is True
    assert "第 0 句对话内容" in result.text
    counters = stats()
    assert counters["failed"] == 1 and counters["fallback"] == 1


# ---------------- 压缩节点 ----------------


def test_compress_skips_short_history(monkeypatch):
    patch_env(monkeypatch, MEMORY_SUMMARY="rule")
    assert compress_messages({"messages": history(8)}) == {}


def test_compress_rule_appends_to_previous_summary(monkeypatch):
    patch_env(monkeypatch, MEMORY_SUMMARY="rule")
    set_completer(None)
    update = compress_messages({"messages": history(10), "summary": "旧摘要"})
    assert update["summary"].startswith("旧摘要")
    assert "第 0 句对话内容" in update["summary"]
    assert len(update["messages"]) == 6        # 10 - 保留 4 条


def test_compress_llm_replaces_summary(monkeypatch):
    patch_env(monkeypatch, MEMORY_SUMMARY="llm", LLM_API_KEY="sk-test")
    set_completer(lambda messages: "新摘要（已融合历史）")
    try:
        update = compress_messages({"messages": history(10), "summary": "旧摘要"})
    finally:
        set_completer(None)
    assert update["summary"] == "新摘要（已融合历史）"   # 不再拼接旧摘要，避免无限增长


# ---------------- 消融脚本的纯逻辑 ----------------


def test_build_padded_row_prefixes_and_keeps_turns():
    row = {"id": "dg-06", "scenario": "理赔", "turns": [{"user": "东西坏了"}]}
    padded = build_padded_row(row, 5)
    assert padded["id"] == "mem-dg-06"
    assert len(padded["turns"]) == 6
    assert padded["turns"][-1] == {"user": "东西坏了"}
    assert row["turns"] == [{"user": "东西坏了"}]        # 不改原行


def make_record(**overrides) -> dict:
    record = {
        "id": "mem-dg-06",
        "scenario": "理赔",
        "checks": {"tools": True},
        "passed": True,
        "tools": {"calls": 2, "success": 2},
        "handoff": False,
        "summary": "摘要",
        "summary_len": 2,
        "last_reply": "已建工单",
    }
    record.update(overrides)
    return record


def test_aggregate_computes_rates():
    records = [
        make_record(),
        make_record(id="b", passed=False, tools={"calls": 2, "success": 1}, handoff=True),
    ]
    summary = aggregate(records, "rule", {})
    assert summary["dialogues"] == 2
    assert summary["pass_rate"] == 0.5
    assert summary["tool_calls"] == 4 and summary["tool_success_rate"] == 0.75
    assert summary["handoff_rate"] == 0.5
    assert aggregate([], "rule", {})["pass_rate"] == 0.0    # 空集不炸


def test_recommendation_branches():
    rule = {"pass_rate": 1.0, "tool_success_rate": 1.0, "handoff_rate": 0.0, "llm": {}}
    llm_better = {"pass_rate": 1.0, "tool_success_rate": 1.0, "handoff_rate": 0.0,
                  "llm": {"calls": 3}}
    assert "打平" in recommendation(rule, llm_better)
    assert "没有真正调用" in recommendation(rule, {**llm_better, "llm": {"calls": 0}})
    worse = {**llm_better, "pass_rate": 0.5}
    assert "保持 rule" in recommendation(rule, worse)


def test_run_row_uses_graph_states():
    class FakeGraph:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, payload, config=None):
            self.calls += 1
            if self.calls < 3:
                return {"reply": "在的", "branches": ["fallback"], "tool_results": []}
            return {
                "reply": "已建工单 TK-1",
                "branches": ["claim"],
                "handoff": False,
                "summary": "摘要",
                "tool_results": [{"name": "create_ticket", "ok": True}],
            }

    row = {
        "id": "mem-dg-06",
        "turns": [{"user": "a"}, {"user": "b"}, {"user": "c"}],
        "check": {"tools": ["create_ticket"], "reply_contains": ["工单"]},
    }
    record = run_row(FakeGraph(), row, mode="rule")
    assert record["passed"] is True
    assert record["tools"] == {"calls": 1, "success": 1}
    assert record["handoff"] is False


def test_write_report_renders_variants(tmp_path):
    rows = [{"id": "dg-06"}]
    variants = {
        "rule": {"mode": "rule", "dialogues": 1, "pass_rate": 1.0, "tool_calls": 2,
                 "tool_success_rate": 1.0, "handoff_rate": 0.0, "avg_summary_len": 80.0,
                 "llm": {}},
        "llm": {"mode": "llm", "dialogues": 1, "pass_rate": 1.0, "tool_calls": 2,
                "tool_success_rate": 1.0, "handoff_rate": 0.0, "avg_summary_len": 60.0,
                "llm": {"calls": 2, "latency_ms": 800.0, "cost": 0.0004}},
    }
    path = tmp_path / "memory_summary.md"
    write_report(path, rows=rows, variants=variants, padding=5)
    text = path.read_text(encoding="utf-8")
    assert "规则折叠 vs LLM 摘要" in text
    assert "| 工具成功率 | 1.0000 | 1.0000 |" in text
    assert "¥0.000400" in text
    assert "## 三、怎么复现" in text

    fake_path = tmp_path / "fake.md"
    write_report(fake_path, rows=rows, variants=variants, padding=5, fake_llm=True)
    assert "--fake-llm" in fake_path.read_text(encoding="utf-8")
