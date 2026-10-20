"""查询改写与检索入口测试。"""

from __future__ import annotations

from agent.rag.embedder import HashingEmbedder
from agent.rag.retriever import Retriever
from agent.rag.rewrite import rewrite, rewrite_detailed


def test_rewrite_resolves_anaphora_with_product_and_topic():
    result = rewrite_detailed("那它呢", {"last_question": "草莓怎么保存", "product": "草莓"})
    assert result.rewritten is True
    assert "草莓" in result.query
    assert "保存" in result.query


def test_rewrite_replaces_anaphora_with_product():
    result = rewrite_detailed("这个能退吗", {"product": "车厘子"})
    assert result.rewritten is True
    assert result.query == "车厘子能退吗"


def test_rewrite_keeps_question_word_when_previous_topic_exists():
    """有上一轮主题时，「这个能退吗」不能被他上一轮的主题覆盖。"""
    result = rewrite_detailed(
        "这个能退吗", {"last_question": "车厘子怎么保存", "product": "车厘子"}
    )
    assert result.query == "车厘子能退吗"


def test_rewrite_keeps_self_contained_query():
    result = rewrite_detailed(
        "坏果比例 50% 赔多少", {"last_question": "草莓怎么保存", "product": "草莓"}
    )
    assert result.rewritten is False
    assert result.query == "坏果比例 50% 赔多少"


def test_rewrite_without_context_returns_original():
    assert rewrite("那它呢", None) == "那它呢"


def test_retriever_returns_fused_hits_sorted_with_ranks():
    retriever = Retriever(embedder=HashingEmbedder())
    hits = retriever.retrieve("车厘子怎么保存", top_k=3)
    assert len(hits) == 3
    assert all(hit.ranks for hit in hits)
    # 默认重排是 0 成本基线，retriever 字段反映实际生效的重排方式
    assert all(hit.retriever == "fusion-score" for hit in hits)
    scores = [hit.fused_score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_retriever_debug_exposes_rewrite_and_both_paths():
    retriever = Retriever(embedder=HashingEmbedder())
    debug = retriever.retrieve_debug(
        "那它呢",
        {"last_question": "草莓怎么保存", "product": "草莓"},
        top_k=3,
    )
    assert debug["rewrite"]["rewritten"] is True
    assert "草莓" in debug["rewrite"]["query"]
    assert debug["bm25"] and debug["vector"] and debug["fused"]
    assert debug["candidate_k"] >= 20
    assert debug["fused"][0]["ranks"]


def test_retriever_uses_rewritten_query_for_recall():
    retriever = Retriever(embedder=HashingEmbedder())
    hits = retriever.retrieve(
        "那它呢",
        {"last_question": "土豆发芽了还能吃吗", "product": "土豆"},
        top_k=3,
    )
    assert hits
    assert any("土豆" in hit.text for hit in hits)
