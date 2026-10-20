"""评测运行器：把"一次配置 = 一次完整检索评测"的逻辑独立出来，供各脚本复用。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from agent.rag.embedder import Embedder, build_embedder
from agent.rag.reranker import Reranker, build_reranker
from agent.rag.retriever import Retriever
from agent.rag.rewrite import load_llm_config, rewrite_with_cache
from agent.rag.store import load_chunks

DEFAULT_DATASET = Path(__file__).resolve().parent / "datasets" / "qa_golden.jsonl"


@dataclass
class ExperimentConfig:
    """一个实验配置：只改一个变量，其余保持一致。"""

    name: str
    kind: str = "hybrid"  # bm25 | vector | hybrid
    rewrite: str = "off"  # off | rule | llm
    rerank: str = "off"  # off | score | bge
    rrf_k: int = 60
    weights: dict[str, float] | None = None
    strategy: str = "rule"  # rule | paragraph
    note: str = ""


def load_dataset(path: Path = DEFAULT_DATASET) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def map_dataset_to_strategy(
    dataset: list[dict],
    strategy: str,
    source_strategy: str = "rule",
) -> tuple[list[dict], dict]:
    """把标准答案从一种分块策略映射到另一种。

    评测集的 relevant_chunk_ids 是按 rule 分块写的；换成 paragraph 分块后粒度变了，
    直接比对会把所有条判成未命中。这里按"规则文本被目标 chunk 包含"建立映射，
    保证不同分块策略在同一个语义标准下比较。
    """
    if strategy == source_strategy:
        return dataset, {"mapped_queries": len(dataset), "total_queries": len(dataset)}

    source_chunks = {chunk.chunk_id: chunk for chunk in load_chunks(source_strategy)}
    target_chunks = load_chunks(strategy)

    def map_ids(ids: list[str]) -> list[str]:
        mapped: list[str] = []
        for chunk_id in ids:
            source = source_chunks.get(chunk_id)
            if source is None:
                mapped.append(chunk_id)
                continue
            hits = [chunk.chunk_id for chunk in target_chunks if source.text in chunk.text]
            if not hits:  # 窗口切分可能截断，退一步用前缀匹配
                prefix = source.text[:16]
                hits = [
                    chunk.chunk_id
                    for chunk in target_chunks
                    if chunk.doc_id == source.doc_id and prefix in chunk.text
                ]
            mapped.extend(hits or [chunk_id])
        return sorted(set(mapped))

    mapped_rows = []
    covered = 0
    for item in dataset:
        mapped_ids = map_ids(item["relevant_chunk_ids"])
        if mapped_ids:
            covered += 1
        mapped_rows.append({**item, "relevant_chunk_ids": mapped_ids})
    return mapped_rows, {"mapped_queries": covered, "total_queries": len(dataset)}


class ExperimentRunner:
    """复用向量模型与重排模型，避免每个配置重新加载。"""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        embedder_kind: str = "auto",
        ks: Sequence[int] = (1, 3, 5),
        rewrite_cache: dict | None = None,
    ) -> None:
        self.embedder = embedder or build_embedder(embedder_kind)
        self.ks = list(ks)
        self.rewrite_cache = rewrite_cache if rewrite_cache is not None else {}
        self._retrievers: dict[tuple, Retriever] = {}
        self._rerankers: dict[str, Reranker] = {}

    def _retriever(self, config: ExperimentConfig) -> Retriever:
        key = (
            config.strategy,
            config.rrf_k,
            json.dumps(config.weights or {}, sort_keys=True),
        )
        if key not in self._retrievers:
            self._retrievers[key] = Retriever(
                chunks=load_chunks(config.strategy),
                embedder=self.embedder,
                strategy=config.strategy,
                rrf_k=config.rrf_k,
                weights=config.weights,
            )
        return self._retrievers[key]

    def _reranker(self, kind: str) -> Reranker | None:
        if kind == "off":
            return None
        if kind not in self._rerankers:
            self._rerankers[kind] = build_reranker(kind)
        return self._rerankers[kind]

    def run(self, config: ExperimentConfig, dataset: Sequence[dict]) -> dict:
        from eval.metrics import evaluate

        retriever = self._retriever(config)
        reranker = self._reranker(config.rerank)
        top_k = max(self.ks)
        rows: list[dict] = []

        for item in dataset:
            context = item.get("context")
            rewrite_result = None
            if config.rewrite != "off":
                model = (
                    load_llm_config().model if config.rewrite in {"llm", "auto"} else "rule"
                )
                rewrite_result = rewrite_with_cache(
                    item["question"],
                    context,
                    mode=config.rewrite,
                    cache=self.rewrite_cache,
                    model=model,
                )

            debug = retriever.search_debug(
                item["question"],
                context,
                top_k,
                kind=config.kind,
                rerank=config.rerank != "off",
                rewrite_mode=config.rewrite,
                reranker=reranker,
            )
            rows.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "search_query": debug["rewrite"]["query"],
                    "rewrite": rewrite_result.to_dict() if rewrite_result else None,
                    "relevant": item["relevant_chunk_ids"],
                    "retrieved": [hit["chunk_id"] for hit in debug["reranked"]],
                    "pool": [hit["chunk_id"] for hit in debug["fused"]],
                }
            )

        return {"config": config, "rows": rows, "metrics": evaluate(rows, self.ks)}
