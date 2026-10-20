"""事实记忆注入意图上下文的测试：开关、阈值、补槽位不召回、trace、消融脚本的纯逻辑。"""

from __future__ import annotations

import numpy as np

from agent.graph import nodes as nodes_module
from agent.graph.nodes import intent_node
from agent.intent.classifier import IntentResult
from agent.memory import facts as facts_store
from agent.trace import build_turn_record
from scripts.ablate_memory_recall import (
    aggregate,
    recommendation,
    run_dialogue,
    seed_prefix,
    write_report,
)


def make_state(**overrides) -> dict:
    state = {
        "user_input": "那笔赔付怎么样了",
        "slots": {},
        "tool_results": [],
        "handoff": False,
        "active_flow": "",
        "pending_slot": "",
        "loop_count": 0,
        "clarify_rounds": 0,
    }
    state.update(overrides)
    return state


HITS = [
    {"id": "f1", "fact": "用户草莓订单 2026093000123 坏果一半，已赔 12.90 元",
     "score": 0.03, "vector_score": 0.91},
    {"id": "f2", "fact": "用户要求开具电子发票", "score": 0.02, "vector_score": 0.30},
]


def patch_search(monkeypatch, hits=HITS):
    import agent.memory.search as search_module

    monkeypatch.setattr(search_module, "search_facts", lambda *a, **k: list(hits))


def patch_classify(monkeypatch, captured: list):
    def fake(text, context=None):
        captured.append(context)
        return IntentResult("bad_fruit_claim", 0.9, "test", ("bad_fruit_claim",))

    monkeypatch.setattr(nodes_module, "classify", fake)


def test_recall_off_by_default(monkeypatch):
    # 显式设 off：本机 .env 可能被人手动改成 on，别让用例跟着环境漂
    monkeypatch.setenv("MEMORY_RECALL", "off")
    called = {"n": 0}
    import agent.memory.search as search_module

    monkeypatch.setattr(search_module, "search_facts", lambda *a, **k: called.__setitem__("n", 1) or [])
    assert nodes_module._recall_memory(make_state(), "那笔赔付怎么样了", "s1") == []
    assert called["n"] == 0                       # 关着就根本不去检索


def test_recall_filters_by_similarity_and_caps(monkeypatch):
    monkeypatch.setenv("MEMORY_RECALL", "on")
    patch_search(monkeypatch)
    hits = nodes_module._recall_memory(make_state(), "那笔赔付怎么样了", "s1")
    assert [hit["id"] for hit in hits] == ["f1"]   # f2 相似度 0.30 低于 0.55，被挡掉
    assert hits[0]["fact"].startswith("用户草莓订单")


def test_recall_keeps_entity_hit_even_below_similarity(monkeypatch):
    """订单号精确命中是硬信号：向量分不够也要放行，否则 BM25 那一路白做。"""
    monkeypatch.setenv("MEMORY_RECALL", "on")
    patch_search(
        monkeypatch,
        [{"id": "f3", "fact": "用户订单 2026093000123 已签收",
          "score": 0.02, "vector_score": 0.31, "entity_hits": 1}],
    )
    hits = nodes_module._recall_memory(make_state(), "2026093000123", "s1")
    assert [hit["id"] for hit in hits] == ["f3"]
    assert hits[0]["entity_hits"] == 1


def test_recall_skips_slot_filling_turns(monkeypatch):
    monkeypatch.setenv("MEMORY_RECALL", "on")
    patch_search(monkeypatch)
    assert nodes_module._recall_memory(make_state(active_flow="bad_fruit_claim"), "订单 2026093000123", "s1") == []


def test_intent_node_injects_memory_into_classifier_context(monkeypatch):
    monkeypatch.setenv("MEMORY_RECALL", "on")
    patch_search(monkeypatch)
    captured: list = []
    patch_classify(monkeypatch, captured)

    update = intent_node(make_state(session_id="s1"), {"configurable": {"thread_id": "s1"}})
    assert [hit["id"] for hit in update["memory_hits"]] == ["f1"]
    # few-shot 分类器拿到的是 dict 上下文，且里面有事实文本
    assert captured[0]["memory"] == ["用户草莓订单 2026093000123 坏果一半，已赔 12.90 元"]


def test_intent_node_falls_back_when_classifier_raises(monkeypatch):
    """模型跑飞（解析失败/显存不足）不能让整轮对话崩：回退规则版。"""
    monkeypatch.setenv("MEMORY_RECALL", "off")

    def boom(text, context=None):
        raise ValueError("本地模型输出无法解析")

    monkeypatch.setattr(nodes_module, "classify", boom)
    update = intent_node(make_state(session_id="s1"), {"configurable": {"thread_id": "s1"}})
    assert update["intent"]
    assert update["confidence"] > 0


def test_local_model_context_is_off_by_default(monkeypatch):
    """SFT 模型是单轮训练的：默认不拼上下文，避免把提示词当答案复读。"""
    from agent.intent.local_model import local_context

    monkeypatch.setenv("LOCAL_INTENT_CONTEXT", "off")
    assert local_context({"memory": ["x"]}) is None
    monkeypatch.setenv("LOCAL_INTENT_CONTEXT", "on")
    assert local_context({"memory": ["x"]}) == {"memory": ["x"]}
    assert local_context({"memory": ["x"]}, enabled=False) is None


def test_trace_records_injected_memory():
    record = build_turn_record(
        make_state(memory_hits=HITS[:1], reply="已赔付 12.90 元"),
        session_id="s1",
        user_input="那笔赔付怎么样了",
        turn=2,
    )
    assert record["memory_hits"][0]["id"] == "f1"
    assert len(record["memory_hits"][0]["fact"]) <= 60


# ---------- 消融脚本：纯逻辑（不跑图） ----------


def make_record(**overrides) -> dict:
    record = {
        "id": "dg-06",
        "passed": True,
        "tools": 2,
        "tool_success": 2,
        "handoff": False,
        "recalls": [0, 2],
        "turn_log": [],
        "tool_hits": [True],
        "asks": 1,
        "elapsed_ms": 100.0,
    }
    record.update(overrides)
    return record


def test_aggregate_counts_injected_turns_not_dialogue_averages():
    records = [
        make_record(recalls=[0, 2], elapsed_ms=100.0),
        make_record(id="dg-07", passed=False, tools=0, tool_success=0,
                    handoff=True, recalls=[0, 0], asks=2, elapsed_ms=300.0),
    ]
    stats = aggregate(records, "on")
    assert stats["avg_recall"] == 0.5          # 2 条 ÷ 4 轮，不是"每段平均再平均"
    assert stats["recall_turns"] == 1
    assert stats["injected_dialogues"] == 1
    assert stats["pass_rate"] == 0.5
    assert stats["tool_success_rate"] == 1.0
    assert stats["tool_hit_rate"] == 1.0       # 2 段各 1 个期望工具，都调到了
    assert stats["tool_expect"] == 2
    assert stats["handoff_rate"] == 0.5
    assert aggregate([], "off")["avg_recall"] == 0.0   # 空集不炸


def test_recommendation_branches():
    base = {"dialogues": 6, "avg_recall": 1.2, "pass_rate": 1.0,
            "tool_success_rate": 1.0, "tool_hit_rate": 1.0, "asks_total": 3}
    assert "没有跑出" in recommendation(base, {**base, "dialogues": 0})
    assert "一次都没召回" in recommendation(base, {**base, "avg_recall": 0.0})
    assert "收益" in recommendation({**base, "pass_rate": 0.5}, base)
    assert "变差" in recommendation(base, {**base, "pass_rate": 0.5})
    assert "追问更少" in recommendation(base, {**base, "asks_total": 1})
    assert "rule 意图后端不读上下文" in recommendation(base, base, backend="rule")
    assert "打平" in recommendation(base, base, backend="local")


def test_seed_prefix_skips_last_turn(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_FACTS_DB", str(tmp_path / "facts.db"))
    monkeypatch.setenv("MEMORY_INDEX_PATH", str(tmp_path / "index.json"))

    class FakeEmbedder:
        def encode(self, texts, *, is_query: bool = False):
            return np.ones((len(texts), 4), dtype="float32")

    monkeypatch.setattr("agent.memory.writer._default_embedder", lambda: FakeEmbedder())

    row = {"id": "dg-06", "turns": [{"user": "a"}, {"user": "b"}, {"user": "c"}]}
    assert seed_prefix("s1", row) == 1
    stored = facts_store.list_facts("s1")
    assert len(stored) == 1
    # 最后一轮不播种，防泄漏
    assert stored[0]["fact"] == "用户此前反馈：a；b"
    assert stored[0]["source"] == "seed"

    # 单轮场景没有"上一次会话"，不写事实
    assert seed_prefix("s2", {"id": "dg-01", "turns": [{"user": "查单"}]}) == 0
    assert facts_store.list_facts("s2") == []


def test_run_dialogue_records_injected_facts_per_turn():
    class FakeGraph:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, payload, config=None):
            self.calls += 1
            if self.calls == 1:
                return {
                    "reply": "请提供订单号",
                    "memory_hits": [
                        {"id": "f1", "fact": "x" * 80, "vector_score": 0.8, "entity_hits": 0}
                    ],
                }
            return {
                "reply": "已建工单 TK-1",
                "tool_results": [{"name": "create_ticket", "ok": True}],
            }

    row = {
        "id": "dg-06",
        "turns": [{"user": "a"}, {"user": "b"}],
        "check": {"tools": ["create_ticket"], "reply_contains": ["工单"]},
    }
    record = run_dialogue(FakeGraph(), row, "s1")
    assert record["recalls"] == [1, 0]
    assert record["turn_log"][0]["recall"][0]["fact"] == "x" * 40   # trace 里只留摘要
    assert record["asks"] == 1
    assert record["passed"] is True


def test_run_dialogue_separates_thread_from_session():
    """三次跑要换 thread（checkpoint 别串），但保持同一 session（事实才看得到）。"""
    seen: list[tuple] = []

    class FakeGraph:
        def invoke(self, payload, config=None):
            seen.append((payload["session_id"], config["configurable"]["thread_id"]))
            return {"reply": "好的"}

    row = {"id": "dg-06", "turns": [{"user": "在吗"}]}
    run_dialogue(FakeGraph(), row, "sess-1", thread_id="sess-1-off")
    assert seen == [("sess-1", "sess-1-off")]


def test_write_report_renders_mechanism_check_and_details(tmp_path):
    off = {"dialogues": 1, "turns": 2, "pass_rate": 1.0, "tool_calls": 1,
           "tool_success_rate": 1.0, "tool_hit_rate": 1.0, "handoff_rate": 0.0, "avg_recall": 0.0,
           "recall_turns": 0, "injected_dialogues": 0, "asks_total": 1,
           "avg_elapsed_ms": 120.0}
    on = {**off, "avg_recall": 0.5, "recall_turns": 1, "injected_dialogues": 1}
    details = {"on": [make_record(turn_log=[{"user": "a", "recall": [
        {"id": "f1", "fact": "用户订单 2026093000123 坏果一半",
         "vector_score": 0.81, "entity_hits": 1}]}])]}
    path = tmp_path / "memory_recall.md"
    write_report(
        path, rows=[{"id": "dg-06"}], off=off, on=on, details=details,
        padding=5, seed_mode="prefix", backend="rule",
    )
    text = path.read_text(encoding="utf-8")
    assert "记忆注入消融" in text
    assert "| 平均每轮注入事实数 | 0.0 | 0.5 |" in text
    assert "实验组发生注入的轮次：1/2" in text
    assert "rule 意图后端按关键词判定" in text
    assert "坏果一半（0.81）" in text


def test_write_report_flags_zero_recall(tmp_path):
    stats = {"dialogues": 1, "turns": 2, "pass_rate": 1.0, "tool_calls": 0,
             "tool_success_rate": 0.0, "tool_hit_rate": 0.0, "handoff_rate": 0.0, "avg_recall": 0.0,
             "recall_turns": 0, "injected_dialogues": 0, "asks_total": 0,
             "avg_elapsed_ms": 10.0}
    path = tmp_path / "zero.md"
    write_report(path, rows=[{"id": "dg-06"}], off=stats, on=stats, padding=5)
    text = path.read_text(encoding="utf-8")
    assert "没有任何一轮发生注入" in text
