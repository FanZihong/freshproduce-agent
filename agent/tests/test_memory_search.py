"""事实记忆混合召回的测试：向量管语义、BM25 管字面、实体标签加分。"""

from __future__ import annotations

import numpy as np

from agent.memory import facts as facts_store
from agent.memory.search import query_entities, search_facts

F1 = "用户草莓订单 2026093000123 坏果一半，赔付 12.90 元"
F2 = "用户车厘子订单 2026092800451 还没发货"
F3 = "用户在咨询无理由退货政策"
F4 = "草莓订单 2026093000123 坏果约一半，已赔 12.9 元"
# 实体被打了标签，但事实文本里没有出现订单号（抽取时常这样"改写"）
F5 = "用户有一笔售后正在退款流程中"

SEMANTIC_QUERY = "果子烂了一半怎么赔"
LITERAL_QUERY = "2026093000123"
ENTITY_QUERY = "那个 2026093000123 到底怎么样了"


class FakeEmbedder:
    """固定向量：让"语义命中"和"字面命中"在测试里可以分别构造。"""

    name = "fake"
    dim = 4

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def encode(self, texts, *, is_query: bool = False):
        return np.asarray(
            [self.mapping.get(str(text), [0.0, 0.0, 0.0, 1.0]) for text in texts],
            dtype="float32",
        )


def seed(tmp_path):
    db = tmp_path / "facts.db"
    records = {}
    for text, category, entities in (
        (F1, "claim", ["2026093000123", "草莓"]),
        (F2, "order", ["2026092800451", "车厘子"]),
        (F3, "policy", []),
        (F4, "claim", ["2026093000123"]),
        (F5, "claim", ["2026093000123"]),
    ):
        record, _ = facts_store.insert_fact(
            "s1", text, category=category, entities=entities, db_path=db
        )
        records[text] = record["id"]
    return db, records


def make_embedder() -> FakeEmbedder:
    return FakeEmbedder(
        {
            F1: [1, 0, 0, 0],
            F4: [1, 0, 0, 0],
            F2: [0, 1, 0, 0],
            F3: [0, 0, 1, 0],
            F5: [0, 0, 0, 1],
            SEMANTIC_QUERY: [1, 0, 0, 0],
            # 故意让"字面查询"的向量指向另一条事实：证明 BM25 能把它救回来
            LITERAL_QUERY: [0, 1, 0, 0],
            ENTITY_QUERY: [0, 1, 0, 0],
        }
    )


def test_query_entities_picks_order_id_and_product():
    assert query_entities("那个 2026093000123 怎么样") == {"2026093000123"}
    assert "草莓" in query_entities("草莓怎么存")
    assert query_entities("你好") == set()


def test_semantic_query_hits_paraphrase_via_vector(tmp_path):
    db, records = seed(tmp_path)
    results = search_facts(
        SEMANTIC_QUERY,
        "s1",
        db_path=db,
        index_path=tmp_path / "index.json",
        embedder=make_embedder(),
    )
    assert results
    assert results[0]["id"] in {records[F1], records[F4]}     # 换说法也能召回
    assert results[0]["vector_score"] > 0


def test_literal_query_is_rescued_by_bm25(tmp_path):
    """向量路指向别的事实，靠 BM25 的订单号字面匹配把正确答案拉回前列。"""
    db, records = seed(tmp_path)
    results = search_facts(
        LITERAL_QUERY,
        "s1",
        db_path=db,
        index_path=tmp_path / "index.json",
        embedder=make_embedder(),
        top_k=3,
    )
    ids = [row["id"] for row in results]
    assert records[F1] in ids[:2] or records[F4] in ids[:2]
    assert any(row["bm25_score"] > 0 for row in results)


def test_entity_tag_boost_promotes_fact(tmp_path):
    """事实文本里没写订单号、但打了实体标签时，实体加成把这条提前。

    这里单独造一个小会话：三条事实的文本都不含订单号，BM25 两路都无分，
    只有"打了实体标签"的那条能靠加成胜出——把加成的作用单独隔出来。
    """
    db = tmp_path / "facts.db"
    tagged, _ = facts_store.insert_fact(
        "s-entity", F5, category="claim", entities=["2026093000123"], db_path=db
    )
    other_a, _ = facts_store.insert_fact("s-entity", F3, category="policy", db_path=db)
    other_b, _ = facts_store.insert_fact("s-entity", "用户问过发票怎么开", db_path=db)

    boosted = search_facts(
        ENTITY_QUERY,
        "s-entity",
        db_path=db,
        index_path=tmp_path / "index.json",
        embedder=FakeEmbedder({F5: [0, 0, 0, 1], F3: [0, 1, 0, 0],
                               "用户问过发票怎么开": [0, 0, 1, 0],
                               ENTITY_QUERY: [0, 1, 0, 0]}),
        top_k=3,
    )
    assert boosted[0]["id"] == tagged["id"]
    assert boosted[0]["entity_hits"] >= 1

    plain = search_facts(
        ENTITY_QUERY,
        "s-entity",
        db_path=db,
        index_path=tmp_path / "index.json",
        embedder=FakeEmbedder({F5: [0, 0, 0, 1], F3: [0, 1, 0, 0],
                               "用户问过发票怎么开": [0, 0, 1, 0],
                               ENTITY_QUERY: [0, 1, 0, 0]}),
        top_k=3,
        entity_boost=0.0,
    )
    assert plain[0]["id"] != tagged["id"]       # 关掉加成，它就不该在第一位
    assert plain[0]["id"] in {other_a["id"], other_b["id"]}


def test_search_handles_empty_session_and_top_k(tmp_path):
    db, _ = seed(tmp_path)
    assert search_facts("随便问问", "没有这个会话", db_path=db) == []
    results = search_facts(
        SEMANTIC_QUERY,
        "s1",
        db_path=db,
        index_path=tmp_path / "index.json",
        embedder=make_embedder(),
        top_k=2,
    )
    assert len(results) == 2
