"""检索测试：BM25 命中规则条款、向量检索返回有序结果、两路确实互补。"""

from __future__ import annotations

from agent.rag.embedder import HashingEmbedder
from agent.rag.hybrid_search import BM25Retriever, HybridSearcher, tokenize
from agent.rag.store import load_chunks


def test_tokenize_splits_chinese_into_bigrams():
    tokens = tokenize("坏果怎么赔")
    assert "坏果" in tokens
    assert "怎么" in tokens


def test_bm25_hits_payout_rules_for_claim_query():
    chunks = load_chunks("rule")
    hits = BM25Retriever(chunks).search("坏果比例 50% 赔多少", top_k=3)
    assert hits
    assert any(hit.doc_id == "赔付标准表" for hit in hits)


def test_vector_search_returns_sorted_results():
    chunks = load_chunks("rule")
    searcher = HybridSearcher(chunks, HashingEmbedder())
    hits = searcher.search_vector("车厘子怎么保存", top_k=5)
    assert len(hits) == 5
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(hit.retriever == "vector" for hit in hits)


def test_two_retrievers_are_complementary():
    chunks = load_chunks("rule")
    searcher = HybridSearcher(chunks, HashingEmbedder())
    result = searcher.compare("坏果怎么赔", top_k=5)
    assert result["bm25"] and result["vector"]
    # 两路结果不应完全一致，否则说明其中一路没有独立价值
    assert len(result["overlap"]) < len(result["bm25"]) or len(result["overlap"]) < len(result["vector"])


def test_paragraph_strategy_loads_from_index():
    chunks = load_chunks("paragraph")
    assert chunks
    assert all(chunk.strategy == "paragraph" for chunk in chunks)
