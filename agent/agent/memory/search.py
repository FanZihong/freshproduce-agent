"""事实记忆的混合召回：向量 top-k + BM25 top-k → RRF 融合 → 实体标签加成。

为什么两路都要：
- **向量**管"换说法"（"果子烂了" ≈ "坏果"），但会漏掉订单号这类字面精确信息；
- **BM25**（bigram 分词）管字面精确，订单号、商品名一搜一个准，但对改写不敏感；
- RRF 只看排名不看分数，两路量纲不同也能融合，比简单加权稳。

实体加成是"结构化兜底"：查询里出现的订单号/商品名，如果正是某条事实的标签，
就在融合分上再加一点——这是"同实体不同说法"能被提前的关键。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from agent.config import get_float
from agent.intent.slot_extractor import extract_order_id
from agent.memory import facts as facts_store
from agent.memory.index import MemoryIndex
from agent.memory.writer import _default_embedder, _encode, default_index_path
from agent.rag.hybrid_search import tokenize
from agent.rag.rewrite import KNOWN_PRODUCTS

DEFAULT_TOP_K = 5
RRF_K = 60
ENTITY_BOOST = 0.05


def query_entities(text: str) -> set[str]:
    """从查询里抽实体标签：订单号 + 商品名（和抽取阶段的事实标签对齐）。"""
    entities: set[str] = set()
    order_id = extract_order_id(text)
    if order_id:
        entities.add(order_id)
    for product in KNOWN_PRODUCTS:
        if product and product in text:
            entities.add(product)
    return entities


def _bm25_scores(texts: list[str], query: str) -> list[float]:
    """bigram 分词 + BM25：和知识库检索用同一套分词，口径一致。"""
    if not texts:
        return []
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:  # 没装就退化成"字面命中次数"，保证链路可用
        query_tokens = set(tokenize(query))
        return [float(len(query_tokens & set(tokenize(text)))) for text in texts]
    corpus = [tokenize(text) for text in texts]
    scores = BM25Okapi(corpus).get_scores(tokenize(query))
    return [float(score) for score in scores]


def _ranks(scores: list[float], *, positive_only: bool = True) -> dict[int, int]:
    """把分数变成名次（1 起）；`positive_only` 时只给有分的项排名，避免全零噪声。"""
    indexed = [
        (index, score) for index, score in enumerate(scores) if not positive_only or score > 0
    ]
    indexed.sort(key=lambda pair: (-pair[1], pair[0]))
    return {index: rank for rank, (index, _) in enumerate(indexed, start=1)}


def _rrf(rank_maps: list[dict[int, int]], k: int = RRF_K) -> dict[int, float]:
    fused: Counter = Counter()
    for ranks in rank_maps:
        for index, rank in ranks.items():
            fused[index] += 1.0 / (k + rank)
    return dict(fused)


def search_facts(
    query: str,
    session_id: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    db_path: Path | str | None = None,
    index_path: Path | str | None = None,
    embedder=None,
    entity_boost: float | None = None,
    now=None,
    decay: bool | None = None,
    touch: bool = True,
) -> list[dict]:
    """召回该会话最相关的事实；返回带分数、老化系数与各路分数的列表。

    排序分 = `(RRF 融合分 + 实体加成) × 老化系数`：
    - 老化只影响**排序**，不改 `vector_score`——注入门槛看的是相关性，不是新旧；
    - 命中的事实会刷新 `last_accessed_at`（`touch=False` 可关，做只读预览时用）；
    - 已过期的事实不参与召回（读时过滤，和 store 的 TTL 语义一致）。
    """
    text = str(query or "").strip()
    if not text or not session_id:
        return []

    facts = facts_store.list_facts(
        session_id, limit=500, include_expired=False, now=now, db_path=db_path
    )
    if not facts:
        return []

    index = MemoryIndex(index_path or default_index_path()).load()
    embedder = embedder or _default_embedder()
    # 索引里缺向量的事实补算（比如事实是别的进程写的）
    for item in facts:
        record = index.records.get(item["id"])
        if record and record.get("vector"):
            continue
        index.upsert(
            item["id"],
            fact=item["fact"],
            session_id=session_id,
            entities=item.get("entities") or [],
            vector=_encode(embedder, item["fact"]),
        )
    index.save()

    order = {item["id"]: position for position, item in enumerate(facts)}
    texts = [item["fact"] for item in facts]

    # 路 1：向量（按事实 id 对齐回 facts 的顺序）
    vector_scores = [0.0] * len(facts)
    query_vector = _encode(embedder, text)
    ids, matrix = index.vectors_for(session_id)
    if ids and matrix.size:
        similarities = matrix @ query_vector
        for fact_id, similarity in zip(ids, similarities):
            position = order.get(fact_id)
            if position is not None:
                vector_scores[position] = float(similarity)

    # 路 2：BM25
    lexical_scores = _bm25_scores(texts, text)

    fused = _rrf([_ranks(vector_scores), _ranks(lexical_scores)])

    # 实体加成：查询实体与事实标签/文本的交集
    entities = query_entities(text)
    boost = ENTITY_BOOST if entity_boost is None else entity_boost
    results: list[dict] = []
    for position, item in enumerate(facts):
        overlap = 0
        if entities:
            tagged = {str(value) for value in (item.get("entities") or [])}
            for entity in entities:
                if entity in tagged or entity in item["fact"]:
                    overlap += 1
        base = fused.get(position, 0.0) + boost * overlap
        if base <= 0:
            continue
        factor = facts_store.decay_factor(item, now=now, enabled=decay)
        results.append(
            {
                "id": item["id"],
                "fact": item["fact"],
                "category": item.get("category"),
                "entities": item.get("entities") or [],
                "score": round(base * factor, 6),
                "base_score": round(base, 6),
                "decay": factor,
                "age_days": facts_store.fact_age_days(item, now=now),
                "vector_score": round(vector_scores[position], 4),
                "bm25_score": round(lexical_scores[position], 4),
                "entity_hits": overlap,
            }
        )

    # 分数相同看未衰减分，再按 id 稳定排序——避免"同分随机换位"让消融不可复现
    results.sort(key=lambda row: (-row["score"], -row["base_score"], row["id"]))
    top = results[: max(1, int(top_k))]
    if touch:
        # 被召回 = 被用到：刷新访问时间，让它不至于因为"很久没提"而过期
        facts_store.touch_facts([item["id"] for item in top], now=now, db_path=db_path)
    return top


def search(query: str, session_id: str, **kwargs) -> list[dict]:
    """`search(query, session_id)` 的别名，语义更直白。"""
    return search_facts(query, session_id, **kwargs)
