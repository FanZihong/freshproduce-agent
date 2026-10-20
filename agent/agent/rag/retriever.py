"""检索统一入口：查询改写 -> 两路召回 -> RRF 融合。

候选池取 top_k 的数倍（默认 4 倍，至少 20 条），融合后再截断到 top_k，
这是生产里的常见做法：融合需要足够多的候选才有意义。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

from agent.rag.chunker import Chunk
from agent.rag.embedder import Embedder, build_embedder
from agent.rag.fusion import DEFAULT_RRF_K, FusedHit, reciprocal_rank_fusion
from agent.rag.hybrid_search import HybridSearcher
from agent.rag.reranker import RerankedHit, Reranker, ScoreReranker, build_reranker
from agent.rag.rewrite import RewriteResult, rewrite_detailed
from agent.rag.store import load_chunks


@dataclass(frozen=True)
class RetrieverHit:
    """最终返回给上层的候选：chunk 元信息 + 每路排名 + 融合分数。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str
    source: str
    fused_score: float
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    retriever: str = "hybrid-rrf"
    rerank_score: float = 0.0
    before_rank: int = 0
    after_rank: int = 0

    @property
    def movement(self) -> int:
        """正数表示相对重排前被提前。"""
        if not self.before_rank or not self.after_rank:
            return 0
        return self.before_rank - self.after_rank

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["fused_score"] = round(self.fused_score, 6)
        payload["rerank_score"] = round(self.rerank_score, 6)
        payload["movement"] = self.movement
        return payload


class Retriever:
    """检索管线：改写 -> BM25 + 向量 -> RRF。"""

    def __init__(
        self,
        chunks: Sequence[Chunk] | None = None,
        embedder: Embedder | None = None,
        *,
        strategy: str = "rule",
        rrf_k: int = DEFAULT_RRF_K,
        weights: dict[str, float] | None = None,
        rewrite_mode: str = "auto",
        reranker: Reranker | None = None,
        rerank_candidates: int = 10,
    ) -> None:
        self.chunks = list(chunks) if chunks is not None else load_chunks(strategy)
        self.strategy = strategy
        self.embedder = embedder or build_embedder()
        self.searcher = HybridSearcher(self.chunks, self.embedder)
        self.rrf_k = rrf_k
        self.weights = weights
        self.rewrite_mode = rewrite_mode
        self.reranker = reranker or ScoreReranker()
        self.rerank_candidates = rerank_candidates

    def _candidate_k(self, top_k: int) -> int:
        return max(top_k * 4, 20)

    @staticmethod
    def _to_hit(
        item,
        before_rank: int = 0,
        after_rank: int = 0,
        rerank_score: float | None = None,
        retriever: str = "hybrid-rrf",
    ) -> RetrieverHit:
        # 兼容两种候选：融合结果 FusedHit 有 fused_score，单路结果 SearchHit 只有 score
        fused_score = float(getattr(item, "fused_score", getattr(item, "score", 0.0)))
        ranks = dict(getattr(item, "ranks", None) or {})
        if not ranks:
            ranks = {str(getattr(item, "retriever", "single")): int(getattr(item, "rank", 0))}
        return RetrieverHit(
            chunk_id=item.chunk_id,
            doc_id=item.doc_id,
            doc_title=item.doc_title,
            section=item.section,
            text=item.text,
            source=item.source,
            fused_score=fused_score,
            ranks=ranks,
            scores=dict(getattr(item, "scores", None) or {}),
            retriever=retriever,
            rerank_score=float(fused_score if rerank_score is None else rerank_score),
            before_rank=before_rank,
            after_rank=after_rank,
        )

    @staticmethod
    def _from_reranked(item: RerankedHit, reranker_name: str) -> RetrieverHit:
        return RetrieverHit(
            chunk_id=item.chunk_id,
            doc_id=item.doc_id,
            doc_title=item.doc_title,
            section=item.section,
            text=item.text,
            source=item.source,
            fused_score=item.fused_score,
            ranks=dict(item.ranks),
            scores={},
            retriever=reranker_name,
            rerank_score=item.rerank_score,
            before_rank=item.before_rank,
            after_rank=item.after_rank,
        )

    def _candidates(self, query: str, kind: str, candidate_k: int):
        """按检索方式取候选，hybrid 时额外返回两路原始结果供调试。"""
        if kind == "bm25":
            return self.searcher.search_bm25(query, candidate_k), None, None
        if kind == "vector":
            return self.searcher.search_vector(query, candidate_k), None, None
        if kind != "hybrid":
            raise ValueError(f"未知检索方式：{kind}")
        bm25_hits = self.searcher.search_bm25(query, candidate_k)
        vector_hits = self.searcher.search_vector(query, candidate_k)
        fused = reciprocal_rank_fusion(
            [("bm25", bm25_hits), ("vector", vector_hits)],
            k=self.rrf_k,
            weights=self.weights,
        )
        return fused, bm25_hits, vector_hits

    def _search(
        self,
        query: str,
        context=None,
        top_k: int = 5,
        *,
        kind: str = "hybrid",
        rerank: bool = True,
        rewrite_mode: str | None = None,
        reranker: Reranker | None = None,
    ) -> tuple[dict, list[RetrieverHit]]:
        """完整检索过程：改写 -> 召回/融合 -> 重排；返回 (调试信息, 结果对象)。"""
        mode = rewrite_mode or self.rewrite_mode
        if mode == "off":
            rewrite_result = RewriteResult(
                original=query, query=query, rewritten=False, reason="未启用改写", strategy="off"
            )
        else:
            rewrite_result: RewriteResult = rewrite_detailed(query, context, mode=mode)
        candidate_k = self._candidate_k(top_k)
        candidates, bm25_hits, vector_hits = self._candidates(
            rewrite_result.query, kind, candidate_k
        )

        pool = list(candidates[: max(self.rerank_candidates, top_k)])
        active_reranker = reranker or self.reranker
        if rerank and pool:
            hits = [
                self._from_reranked(item, active_reranker.name)
                for item in active_reranker.rerank(rewrite_result.query, pool, top_k=top_k)
            ]
        else:
            hits = [
                self._to_hit(item, before_rank=index, after_rank=index)
                for index, item in enumerate(pool[:top_k], start=1)
            ]

        debug = {
            "query": query,
            "rewrite": rewrite_result.to_dict(),
            "rrf_k": self.rrf_k,
            "weights": self.weights or {"bm25": 1.0, "vector": 1.0},
            "candidate_k": candidate_k,
            "embedder": self.embedder.name,
            "strategy": self.strategy,
            "kind": kind,
            "rerank": rerank,
            "reranker": active_reranker.name if rerank else "off",
            "bm25": [hit.to_dict() for hit in (bm25_hits or [])[:top_k]],
            "vector": [hit.to_dict() for hit in (vector_hits or [])[:top_k]],
            "fused": [hit.to_dict() for hit in pool],  # 重排前的候选池
            "reranked": [hit.to_dict() for hit in hits],  # 最终结果
        }
        return debug, hits

    def search_debug(
        self,
        query: str,
        context=None,
        top_k: int = 5,
        *,
        kind: str = "hybrid",
        rerank: bool = True,
        rewrite_mode: str | None = None,
        reranker: Reranker | None = None,
    ) -> dict:
        return self._search(
            query,
            context,
            top_k,
            kind=kind,
            rerank=rerank,
            rewrite_mode=rewrite_mode,
            reranker=reranker,
        )[0]

    def search(
        self,
        query: str,
        context=None,
        top_k: int = 5,
        *,
        kind: str = "hybrid",
        rerank: bool = True,
        rewrite_mode: str | None = None,
        reranker: Reranker | None = None,
    ) -> list[RetrieverHit]:
        return self._search(
            query,
            context,
            top_k,
            kind=kind,
            rerank=rerank,
            rewrite_mode=rewrite_mode,
            reranker=reranker,
        )[1]

    def retrieve(self, query: str, context=None, top_k: int = 5) -> list[RetrieverHit]:
        """对外主接口：融合 + 重排后的候选。"""
        return self.search(query, context, top_k)

    def retrieve_debug(self, query: str, context=None, top_k: int = 5) -> dict:
        """兼容旧调用：等价于 search_debug。"""
        return self.search_debug(query, context, top_k)


_RETRIEVERS: dict[tuple, Retriever] = {}


def get_retriever(
    *,
    strategy: str = "rule",
    rrf_k: int = DEFAULT_RRF_K,
    embedder_kind: str = "auto",
    reranker_kind: str = "score",
) -> Retriever:
    """进程内复用：避免每次检索都重新加载向量模型。"""
    key = (strategy, rrf_k, embedder_kind, reranker_kind)
    if key not in _RETRIEVERS:
        _RETRIEVERS[key] = Retriever(
            strategy=strategy,
            rrf_k=rrf_k,
            embedder=build_embedder(embedder_kind),
            reranker=build_reranker(reranker_kind),
        )
    return _RETRIEVERS[key]


def retrieve(query: str, context=None, top_k: int = 5, **kwargs) -> list[RetrieverHit]:
    """模块级便捷入口。"""
    return get_retriever(**kwargs).retrieve(query, context, top_k)
