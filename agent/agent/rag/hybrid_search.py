"""混合检索：BM25 与向量两路，各自返回带排名的候选。

这一版只做"两路各自检索"，不合并——融合留给下一段的 RRF，
这样才能在消融实验里说清"融合到底带来多少增益"。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from agent.rag.chunker import Chunk
from agent.rag.embedder import Embedder, build_embedder

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+")

# 分词方式：默认字符 bigram（确定性、零依赖，实测在本项目语料上优于 jieba）
_TOKENIZER_MODE = "bigram"


def set_tokenizer(mode: str) -> None:
    """切换分词方式，供消融实验使用：bigram（默认）或 jieba。"""
    global _TOKENIZER_MODE
    if mode not in {"bigram", "jieba"}:
        raise ValueError("分词方式只能是 bigram 或 jieba")
    _TOKENIZER_MODE = mode


def get_tokenizer() -> str:
    return _TOKENIZER_MODE


def tokenize(text: str) -> list[str]:
    """按当前分词方式切词；默认中文按字符 bigram，英文数字按词。"""
    if _TOKENIZER_MODE == "jieba":
        import jieba  # type: ignore

        return [token for token in jieba.cut(text) if token.strip()]
    return tokenize_bigram(text)


def tokenize_bigram(text: str) -> list[str]:
    tokens: list[str] = []
    for piece in TOKEN_PATTERN.findall(text):
        if piece.isascii():
            tokens.append(piece.lower())
        elif len(piece) == 1:
            tokens.append(piece)
        else:
            tokens.extend(piece[index : index + 2] for index in range(len(piece) - 1))
    return tokens


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str
    source: str
    retriever: str
    rank: int
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def _hit(chunk: Chunk, retriever: str, rank: int, score: float) -> SearchHit:
    return SearchHit(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        doc_title=chunk.doc_title,
        section=chunk.section,
        text=chunk.text,
        source=chunk.source,
        retriever=retriever,
        rank=rank,
        score=round(float(score), 4),
    )


class BM25Retriever:
    """关键词路：适合规则条款号、专有名词这类精确匹配。"""

    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        corpus = [tokenize(chunk.text) or [chunk.text] for chunk in self.chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not self.chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        return [
            _hit(self.chunks[index], self.name, rank, scores[index])
            for rank, index in enumerate(order, start=1)
        ]


class VectorRetriever:
    """语义路：适合"怎么赔""能不能退"这类说法多样的问题。"""

    name = "vector"

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder | None = None) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder or build_embedder()
        texts = [chunk.text for chunk in self.chunks]
        self._matrix = self.embedder.encode(texts) if texts else np.zeros((0, self.embedder.dim), dtype=np.float32)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not self.chunks:
            return []
        query_vector = self.embedder.encode([query], is_query=True)[0]
        scores = self._matrix @ query_vector  # 已归一化，点积即余弦相似度
        order = np.argsort(scores)[::-1][:top_k]
        return [
            _hit(self.chunks[index], self.name, rank, scores[index])
            for rank, index in enumerate(order, start=1)
        ]


class HybridSearcher:
    """两路检索的统一入口：既能单独调用，也能对比。"""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder | None = None) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder or build_embedder()
        self.bm25 = BM25Retriever(self.chunks)
        self.vector = VectorRetriever(self.chunks, self.embedder)

    def search_bm25(self, query: str, top_k: int = 5) -> list[SearchHit]:
        return self.bm25.search(query, top_k)

    def search_vector(self, query: str, top_k: int = 5) -> list[SearchHit]:
        return self.vector.search(query, top_k)

    def compare(self, query: str, top_k: int = 5) -> dict:
        """返回两路结果与重合度，用于观察"为什么要混合"。"""
        bm25_hits = self.search_bm25(query, top_k)
        vector_hits = self.search_vector(query, top_k)
        bm25_ids = {hit.chunk_id for hit in bm25_hits}
        vector_ids = {hit.chunk_id for hit in vector_hits}
        return {
            "query": query,
            "top_k": top_k,
            "embedder": self.embedder.name,
            "bm25": bm25_hits,
            "vector": vector_hits,
            "overlap": sorted(bm25_ids & vector_ids),
            "only_bm25": sorted(bm25_ids - vector_ids),
            "only_vector": sorted(vector_ids - bm25_ids),
        }
