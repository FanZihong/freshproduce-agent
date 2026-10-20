"""重排（Rerank）：0 成本基线 + CrossEncoder 精排。

统一接口：rerank(query, candidates, top_k) -> list[RerankedHit]
candidates 可以是 FusedHit 或 RetrieverHit，只要有 chunk_id / text / fused_score 等字段。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, Sequence

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


@dataclass(frozen=True)
class RerankedHit:
    """重排后的候选：带上"重排前排名"以便观察谁被提上来了。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str
    source: str
    rerank_score: float
    fused_score: float
    before_rank: int
    after_rank: int
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def movement(self) -> int:
        """正数表示被提前了，负数表示被压后。"""
        return self.before_rank - self.after_rank

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["rerank_score"] = round(self.rerank_score, 6)
        payload["fused_score"] = round(self.fused_score, 6)
        payload["movement"] = self.movement
        return payload


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: Sequence[Any], top_k: int | None = None) -> list[RerankedHit]:
        ...


def _as_hit(candidate: Any, rerank_score: float, before_rank: int, after_rank: int) -> RerankedHit:
    return RerankedHit(
        chunk_id=candidate.chunk_id,
        doc_id=getattr(candidate, "doc_id", ""),
        doc_title=getattr(candidate, "doc_title", ""),
        section=getattr(candidate, "section", ""),
        text=getattr(candidate, "text", ""),
        source=getattr(candidate, "source", ""),
        rerank_score=float(rerank_score),
        fused_score=float(getattr(candidate, "fused_score", 0.0)),
        before_rank=before_rank,
        after_rank=after_rank,
        ranks=dict(getattr(candidate, "ranks", {}) or {}),
    )


class ScoreReranker:
    """0 成本基线：直接按融合分数排序，不调用任何模型。

    作用有两个：一是作为精排的对照（证明精排到底带来多少收益），
    二是当 CrossEncoder 不可用时兜底，保证链路不中断。
    """

    name = "fusion-score"

    def rerank(self, query: str, candidates: Sequence[Any], top_k: int | None = None) -> list[RerankedHit]:
        indexed = list(enumerate(candidates, start=1))
        ordered = sorted(indexed, key=lambda pair: (-float(getattr(pair[1], "fused_score", 0.0)), pair[0]))
        results = [
            _as_hit(candidate, getattr(candidate, "fused_score", 0.0), before_rank, after_rank)
            for after_rank, (before_rank, candidate) in enumerate(ordered, start=1)
        ]
        return results[:top_k] if top_k else results


class CrossEncoderReranker:
    """BGE 精排：用 CrossEncoder 对 (query, chunk) 逐对打分。

    首次使用会下载约 1.1GB 权重；score_fn 用于注入假打分器，方便单测与离线验证。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        batch_size: int = 8,
        max_length: int = 512,
        score_fn: Callable[[str, Sequence[str]], Sequence[float]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.name = f"cross-encoder:{model_name}"
        self._batch_size = batch_size
        self._score_fn = score_fn
        self._model = None
        if score_fn is None:
            from sentence_transformers import CrossEncoder  # 延迟导入

            self._model = CrossEncoder(model_name, max_length=max_length)

    def _score(self, query: str, texts: Sequence[str]) -> list[float]:
        if self._score_fn is not None:
            return [float(value) for value in self._score_fn(query, texts)]
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        )
        return [float(value) for value in scores]

    def rerank(self, query: str, candidates: Sequence[Any], top_k: int | None = None) -> list[RerankedHit]:
        candidates = list(candidates)
        if not candidates:
            return []
        scores = self._score(query, [str(getattr(item, "text", "")) for item in candidates])
        indexed = list(enumerate(candidates, start=1))
        ordered = sorted(
            zip(indexed, scores),
            key=lambda pair: (-pair[1], pair[0][0]),
        )
        results = [
            _as_hit(candidate, score, before_rank, after_rank)
            for after_rank, ((before_rank, candidate), score) in enumerate(ordered, start=1)
        ]
        return results[:top_k] if top_k else results


def build_reranker(
    kind: str = "auto",
    model_name: str | None = None,
    *,
    score_fn: Callable[[str, Sequence[str]], Sequence[float]] | None = None,
) -> Reranker:
    """kind：score 只用融合分数；bge 强制 CrossEncoder；auto 失败时退回 score。"""
    from agent.config import get_env

    # 允许用环境变量指定本地路径或镜像模型名，例如 RERANK_MODEL=/path/to/bge-reranker-base
    model_name = model_name or get_env("RERANK_MODEL", DEFAULT_RERANK_MODEL)
    if kind == "score":
        return ScoreReranker()
    if kind in {"auto", "bge"}:
        try:
            return CrossEncoderReranker(model_name, score_fn=score_fn)
        except ImportError as exc:
            if kind == "bge":
                raise
            print("[warn] 未安装 sentence-transformers，本次重排退回融合分数基线")
            print(f"       安装命令：pip install sentence-transformers（{exc}）")
            return ScoreReranker()
        except Exception as exc:  # noqa: BLE001 - 网络/模型问题都退化
            hint = (
                "       常见原因：连不上 huggingface.co 或模型未下载。可尝试：\n"
                "         export HF_ENDPOINT=https://hf-mirror.com\n"
                "       或加 --rerank score 用融合分数基线"
            )
            if kind == "bge":
                raise RuntimeError(f"加载重排模型 {model_name} 失败：{exc}\n{hint}") from exc
            print(f"[warn] 加载重排模型 {model_name} 失败，本次退回融合分数基线：{exc}")
            print(hint)
            return ScoreReranker()
    raise ValueError(f"未知重排方式：{kind}")
