"""检索调试接口：返回五级检索过程，供前端调试台展示。"""

from __future__ import annotations

import time

from fastapi import APIRouter, Query

from agent.rag.reranker import Reranker, build_reranker
from agent.rag.retriever import get_retriever
from agent.tools.faq import build_citations

from app.api.schemas import RetrievalSearchOut

router = APIRouter(tags=["retrieval"])

_RERANKERS: dict[str, Reranker] = {}


def _reranker(kind: str) -> Reranker | None:
    """按需构建并缓存重排器（bge 模型只加载一次）。"""
    if kind == "off":
        return None
    if kind not in _RERANKERS:
        _RERANKERS[kind] = build_reranker(kind)
    return _RERANKERS[kind]


def _hit_payload(hit: dict, rank: int) -> dict:
    return {
        "rank": rank,
        "chunkId": hit["chunk_id"],
        "docTitle": hit["doc_title"],
        "section": hit["section"],
        "source": hit["source"],
        "text": hit["text"],
        "score": float(
            hit.get("rerank_score") or hit.get("fused_score") or hit.get("score") or 0.0
        ),
        "fusedScore": float(hit.get("fused_score") or 0.0),
        "rerankScore": float(hit.get("rerank_score") or 0.0),
        "ranks": hit.get("ranks") or {},
        "movement": hit.get("movement"),
    }


@router.get("/retrieval/search", response_model=RetrievalSearchOut)
def retrieval_search(
    query: str = Query(..., min_length=1, max_length=200, description="检索问题"),
    context: str | None = Query(None, description="上一轮用户问题（可选）"),
    product: str | None = Query(None, description="上文识别到的商品（可选）"),
    top_k: int = Query(5, ge=1, le=20, description="最终返回条数"),
    kind: str = Query("hybrid", pattern="^(bm25|vector|hybrid)$", description="检索方式"),
    rerank: str = Query("score", pattern="^(off|score|bge)$", description="重排方式"),
    strategy: str = Query("rule", pattern="^(rule|paragraph)$", description="分块策略"),
    rrf_k: int = Query(60, ge=1, le=500, description="RRF 的 k"),
) -> dict:
    """一次完整检索：改写 -> BM25 / 向量召回 -> RRF 融合 -> 重排 -> 引用。"""
    started = time.perf_counter()
    retriever = get_retriever(strategy=strategy, rrf_k=rrf_k)
    context_payload = {
        "last_question": context or "",
        "product": product or "",
    }
    debug = retriever.search_debug(
        query,
        context_payload if any(context_payload.values()) else None,
        top_k,
        kind=kind,
        rerank=rerank != "off",
        reranker=_reranker(rerank),
    )

    stages = {
        "bm25": [_hit_payload(hit, index) for index, hit in enumerate(debug["bm25"], start=1)],
        "vector": [
            _hit_payload(hit, index) for index, hit in enumerate(debug["vector"], start=1)
        ],
        "fused": [_hit_payload(hit, index) for index, hit in enumerate(debug["fused"], start=1)],
        "reranked": [
            _hit_payload(hit, index) for index, hit in enumerate(debug["reranked"], start=1)
        ],
    }

    return {
        "query": query,
        "rewritten": debug["rewrite"],
        "config": {
            "kind": kind,
            "rerank": rerank,
            "strategy": strategy,
            "rrfK": rrf_k,
            "topK": top_k,
            "candidateK": debug["candidate_k"],
            "embedder": debug["embedder"],
            "reranker": debug["reranker"],
        },
        "stages": stages,
        "citations": build_citations(debug["rewrite"]["query"], debug["reranked"]),
        "elapsedMs": int((time.perf_counter() - started) * 1000),
    }
