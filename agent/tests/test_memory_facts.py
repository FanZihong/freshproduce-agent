"""事实记忆（增量写入）的测试：哈希去重、相似去重、只抽增量、索引落盘。"""

from __future__ import annotations

import json

import numpy as np

from agent.memory import facts as facts_store
from agent.memory.extractor import set_extractor_completer, turns_from_messages
from agent.memory.index import MemoryIndex
from agent.memory.writer import remember


class FakeEmbedder:
    """按文本映射到固定向量：让"同义改写"和"新事实"在测试里可复现。"""

    name = "fake"
    dim = 4

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def encode(self, texts, *, is_query: bool = False):
        return np.asarray(
            [self.mapping.get(str(text), [0.0, 0.0, 0.0, 1.0]) for text in texts],
            dtype="float32",
        )


OLD_FACT = "用户草莓订单 2026093000123 坏果一半，按 50% 赔付 12.90 元"
PARAPHRASE = "草莓订单 2026093000123 坏果约一半，已赔 12.9 元"
NEW_FACT = "用户要求开具电子发票，抬头为公司"


def fake_llm(payload: str, captured: list | None = None):
    def complete(messages):
        if captured is not None:
            captured.append(messages)
        return payload, {"prompt_tokens": 200, "completion_tokens": 60}

    return complete


def make_embedder() -> FakeEmbedder:
    # OLD_FACT 与 PARAPHRASE 映射到同一方向 -> 相似度 1.0；NEW_FACT 正交
    return FakeEmbedder(
        {OLD_FACT: [1, 0, 0, 0], PARAPHRASE: [1, 0, 0, 0], NEW_FACT: [0, 1, 0, 0]}
    )


# ---------------- 事实库：哈希与增删改查 ----------------


def test_fact_hash_ignores_punctuation_and_spaces():
    assert facts_store.fact_hash("用户 喜欢 草莓。") == facts_store.fact_hash("用户喜欢草莓")
    assert facts_store.fact_hash("用户喜欢草莓") != facts_store.fact_hash("用户喜欢车厘子")
    assert len(facts_store.fact_hash("x")) == 32


def test_insert_fact_dedupes_by_hash_within_session(tmp_path):
    db = tmp_path / "facts.db"
    first, created = facts_store.insert_fact("s1", OLD_FACT, db_path=db)
    second, created_again = facts_store.insert_fact("s1", OLD_FACT + "。", db_path=db)
    third, other_session = facts_store.insert_fact("s2", OLD_FACT, db_path=db)
    assert created is True and created_again is False and other_session is True
    assert first["id"] == second["id"]
    assert third["id"] != first["id"]
    assert facts_store.count_facts("s1", db_path=db) == 1


def test_list_and_update_fact(tmp_path):
    db = tmp_path / "facts.db"
    record, _ = facts_store.insert_fact(
        "s1", OLD_FACT, entities=["2026093000123", "草莓"], db_path=db
    )
    listed = facts_store.list_facts("s1", db_path=db)
    assert listed[0]["entities"] == ["2026093000123", "草莓"]

    updated = facts_store.update_fact(
        record["id"], fact=NEW_FACT, entities=["发票"], category="policy", db_path=db
    )
    assert updated["fact"] == NEW_FACT
    assert updated["content_hash"] == facts_store.fact_hash(NEW_FACT)
    assert updated["entities"] == ["发票"]
    assert facts_store.get_fact(record["id"], db_path=db)["category"] == "policy"


# ---------------- 增量写入管线 ----------------


def test_remember_extracts_only_new_facts_and_passes_old_ones(tmp_path):
    """验收主线：旧记忆里有的事不该再抽一遍；prompt 里必须带上旧事实。"""
    db = tmp_path / "facts.db"
    facts_store.insert_fact("s1", OLD_FACT, db_path=db)
    captured: list[list[dict]] = []
    set_extractor_completer(
        fake_llm(
            json.dumps(
                [{"fact": NEW_FACT, "category": "policy", "entities": ["发票"], "action": "add"}],
                ensure_ascii=False,
            ),
            captured,
        )
    )
    try:
        stats = remember(
            "s1",
            [{"user": "再帮我开个发票", "reply": "好的，请提供抬头"}],
            mode="llm",
            db_path=db,
            index_path=tmp_path / "index.json",
            embedder=make_embedder(),
        )
    finally:
        set_extractor_completer(None)

    assert stats.candidates == 1 and stats.added == 1 and stats.updated == 0
    assert stats.calls == 1 and stats.cost > 0
    assert facts_store.count_facts("s1", db_path=db) == 2
    # 模型拿到的是"旧事实 + 新对话"，它才有机会只输出增量
    prompt = captured[0][1]["content"]
    assert "已有记忆" in prompt and OLD_FACT in prompt
    assert "再帮我开个发票" in prompt


def test_remember_skips_identical_fact_by_hash(tmp_path):
    db = tmp_path / "facts.db"
    facts_store.insert_fact("s1", OLD_FACT, db_path=db)
    set_extractor_completer(
        fake_llm(json.dumps([{"fact": OLD_FACT, "action": "add"}], ensure_ascii=False))
    )
    try:
        stats = remember(
            "s1",
            [{"user": "草莓坏了一半", "reply": "已赔付"}],
            mode="llm",
            db_path=db,
            index_path=tmp_path / "index.json",
            embedder=make_embedder(),
        )
    finally:
        set_extractor_completer(None)
    assert stats.skipped == 1 and stats.added == 0
    assert facts_store.count_facts("s1", db_path=db) == 1


def test_remember_updates_when_paraphrase_is_similar(tmp_path):
    """第二层去重：换说法命中相似度阈值 -> 更新旧事实，而不是新增一条。"""
    db = tmp_path / "facts.db"
    record, _ = facts_store.insert_fact("s1", OLD_FACT, db_path=db)
    set_extractor_completer(
        fake_llm(json.dumps([{"fact": PARAPHRASE, "action": "add"}], ensure_ascii=False))
    )
    try:
        stats = remember(
            "s1",
            [{"user": "赔付到底多少来着", "reply": "12.9 元"}],
            mode="llm",
            db_path=db,
            index_path=tmp_path / "index.json",
            embedder=make_embedder(),
        )
    finally:
        set_extractor_completer(None)
    assert stats.updated == 1 and stats.added == 0
    assert facts_store.count_facts("s1", db_path=db) == 1
    assert facts_store.get_fact(record["id"], db_path=db)["fact"] == PARAPHRASE


def test_remember_honours_update_and_skip_actions(tmp_path):
    db = tmp_path / "facts.db"
    record, _ = facts_store.insert_fact("s1", OLD_FACT, db_path=db)
    set_extractor_completer(
        fake_llm(
            json.dumps(
                [
                    {"fact": NEW_FACT, "action": "update", "target_id": record["id"]},
                    {"fact": "无关紧要的一句话", "action": "skip"},
                ],
                ensure_ascii=False,
            )
        )
    )
    try:
        stats = remember(
            "s1",
            [{"user": "改成开发票", "reply": "好的"}],
            mode="llm",
            db_path=db,
            index_path=tmp_path / "index.json",
            embedder=make_embedder(),
        )
    finally:
        set_extractor_completer(None)
    assert stats.updated == 1 and stats.skipped == 1 and stats.added == 0
    assert facts_store.get_fact(record["id"], db_path=db)["fact"] == NEW_FACT


def test_remember_falls_back_to_rule_when_llm_fails(tmp_path):
    db = tmp_path / "facts.db"

    def broken(messages):
        raise RuntimeError("模型超时")

    set_extractor_completer(broken)
    try:
        stats = remember(
            "s1",
            [{"user": "你好", "reply": "在的"}],
            summary="用户咨询过草莓理赔，工单已建",
            mode="llm",
            db_path=db,
            index_path=tmp_path / "index.json",
            embedder=make_embedder(),
        )
    finally:
        set_extractor_completer(None)
    assert stats.added == 1 and stats.calls == 1
    assert facts_store.count_facts("s1", db_path=db) == 1


def test_remember_off_mode_does_nothing(tmp_path, monkeypatch):
    db = tmp_path / "facts.db"
    monkeypatch.setenv("MEMORY_FACTS", "off")
    stats = remember(
        "s1",
        [{"user": "你好", "reply": "在的"}],
        summary="摘要",
        db_path=db,
        index_path=tmp_path / "index.json",
        embedder=make_embedder(),
    )
    assert stats.mode == "off" and stats.candidates == 0
    assert facts_store.count_facts("s1", db_path=db) == 0


def test_index_file_persists_vectors(tmp_path):
    db = tmp_path / "facts.db"
    index_path = tmp_path / "memory_index.json"
    set_extractor_completer(
        fake_llm(json.dumps([{"fact": NEW_FACT, "action": "add"}], ensure_ascii=False))
    )
    try:
        remember(
            "s1",
            [{"user": "开发票", "reply": "好的"}],
            mode="llm",
            db_path=db,
            index_path=index_path,
            embedder=make_embedder(),
        )
    finally:
        set_extractor_completer(None)

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["facts"] and payload["facts"][0]["vector"]
    reloaded = MemoryIndex(index_path).load()
    ids, matrix = reloaded.vectors_for("s1")
    assert len(ids) == 1 and matrix.shape == (1, 4)


def test_turns_from_messages_pairs_user_and_reply():
    class Msg:
        def __init__(self, type, content):
            self.type = type
            self.content = content

    turns = turns_from_messages(
        [Msg("human", "你好"), Msg("ai", "在的"), Msg("human", "查订单"), Msg("tool", "x")]
    )
    assert turns == [{"user": "你好", "reply": "在的"}, {"user": "查订单", "reply": ""}]
