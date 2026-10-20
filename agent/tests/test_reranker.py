"""重排测试：分数基线、CrossEncoder 注入打分、管线里把正确项提到第一位。"""

from __future__ import annotations

from agent.rag.embedder import HashingEmbedder
from agent.rag.fusion import FusedHit
from agent.rag.reranker import CrossEncoderReranker, ScoreReranker, build_reranker
from agent.rag.retriever import Retriever


def _candidate(chunk_id: str, fused_score: float, text: str = "") -> FusedHit:
    return FusedHit(
        chunk_id=chunk_id,
        doc_id="doc",
        doc_title="文档",
        section="小节",
        text=text or chunk_id,
        source="示例来源",
        fused_score=fused_score,
        ranks={"bm25": 1, "vector": 2},
        scores={},
    )


def test_score_reranker_orders_by_fused_score_and_tracks_movement():
    candidates = [
        _candidate("a", 0.030),
        _candidate("b", 0.010),
        _candidate("c", 0.020),
    ]
    reranked = ScoreReranker().rerank("任意查询", candidates)

    assert [hit.chunk_id for hit in reranked] == ["a", "c", "b"]
    assert [hit.before_rank for hit in reranked] == [1, 3, 2]
    assert [hit.after_rank for hit in reranked] == [1, 2, 3]
    assert reranked[1].movement == 1  # c 从第 3 提到第 2
    assert reranked[2].movement == -1  # b 从第 2 掉到第 3


def test_cross_encoder_reranker_uses_injected_scorer():
    candidates = [_candidate("a", 0.030), _candidate("b", 0.010), _candidate("c", 0.020)]

    def fake_scores(query: str, texts):
        # b 的分数最高，应该被提到第一位
        return [0.1, 0.9, 0.2]

    reranked = CrossEncoderReranker(score_fn=fake_scores).rerank("任意查询", candidates, top_k=2)
    assert [hit.chunk_id for hit in reranked] == ["b", "c"]
    assert reranked[0].rerank_score == 0.9
    assert reranked[0].before_rank == 2
    assert reranked[0].movement == 1


def test_build_reranker_score_kind():
    assert build_reranker("score").name == "fusion-score"


def test_reranker_promotes_lower_ranked_chunk_to_first_place():
    """端到端：把重排前第 3 名的 chunk 提到第 1 名。"""
    retriever = Retriever(embedder=HashingEmbedder())
    query = "车厘子怎么保存"

    before = retriever.search_debug(query, None, 10, rerank=False)
    pool = before["fused"]
    assert len(pool) >= 3
    target = pool[2]  # 重排前第 3 名

    def scorer(_query: str, texts):
        return [1.0 if text == target["text"] else 0.0 for text in texts]

    after = retriever.search_debug(
        query, None, 5, rerank=True, reranker=CrossEncoderReranker(score_fn=scorer)
    )

    assert after["reranked"][0]["chunk_id"] == target["chunk_id"]
    assert after["reranked"][0]["before_rank"] == 3
    assert after["reranked"][0]["movement"] == 2  # 第 3 名 -> 第 1 名


def test_reranker_promotes_keyword_matching_chunk_within_pool():
    """用关键词打分模拟精排：包含"土豆"的 chunk 应排在第一位。"""

    def keyword_scorer(_query: str, texts):
        return [1.0 if "土豆" in text else 0.0 for text in texts]

    retriever = Retriever(embedder=HashingEmbedder())
    after = retriever.search_debug(
        "土豆发芽了还能吃吗",
        None,
        5,
        rerank=True,
        reranker=CrossEncoderReranker(score_fn=keyword_scorer),
    )
    assert "土豆" in after["reranked"][0]["text"]
