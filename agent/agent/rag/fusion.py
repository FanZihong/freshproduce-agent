"""RRF（Reciprocal Rank Fusion）融合。

公式：score(chunk) = Σ weight(retriever) / (k + rank(retriever, chunk))
只用排名不用原始分数，所以不同量纲的检索路（BM25 分数 vs 余弦相似度）可以直接相加。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

from agent.rag.hybrid_search import SearchHit

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedHit:
    """融合后的候选：保留每路的排名与原始分数，便于调试与消融。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str
    source: str
    fused_score: float
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        # 内部保留全精度，输出给界面/接口时再取整
        payload["fused_score"] = round(self.fused_score, 6)
        return payload


def rrf_score(rank: int, k: int = DEFAULT_RRF_K, weight: float = 1.0) -> float:
    """单路单项的 RRF 贡献分。"""
    return weight / (k + rank)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[tuple[str, Iterable[SearchHit]]],
    k: int = DEFAULT_RRF_K,
    weights: dict[str, float] | None = None,
) -> list[FusedHit]:
    """把多路结果融合成一个排序。

    排序键：融合分降序 -> 最好排名（越小越前）-> chunk_id（保证结果可复现）。
    """
    if k < 0:
        raise ValueError("RRF 的 k 必须为非负数")
    weight_table = weights or {}

    table: dict[str, dict] = {}
    for retriever_name, hits in ranked_lists:
        weight = float(weight_table.get(retriever_name, 1.0))
        for hit in hits:
            entry = table.setdefault(
                hit.chunk_id,
                {
                    "hit": hit,
                    "score": 0.0,
                    "ranks": {},
                    "scores": {},
                },
            )
            entry["score"] += rrf_score(hit.rank, k=k, weight=weight)
            entry["ranks"][retriever_name] = hit.rank
            entry["scores"][retriever_name] = hit.score

    fused = [
        FusedHit(
            chunk_id=entry["hit"].chunk_id,
            doc_id=entry["hit"].doc_id,
            doc_title=entry["hit"].doc_title,
            section=entry["hit"].section,
            text=entry["hit"].text,
            source=entry["hit"].source,
            fused_score=float(entry["score"]),
            ranks=dict(entry["ranks"]),
            scores=dict(entry["scores"]),
        )
        for entry in table.values()
    ]
    fused.sort(key=lambda item: (-item.fused_score, min(item.ranks.values()), item.chunk_id))
    return fused
