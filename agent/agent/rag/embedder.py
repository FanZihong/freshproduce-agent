"""向量化实现。

默认用 sentence-transformers 加载 BGE 中文小模型；
测试与无模型环境用 HashingEmbedder 兜底，保证整条链路仍然可跑。
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np

DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        ...


class SentenceTransformerEmbedder:
    """BGE 中文小模型：CPU 可跑，首次使用会下载约 100MB 权重。"""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入，避免无模型环境启动失败

        self.name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        # sentence-transformers 新版把方法改名了，这里兼容两种写法
        dimension_getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self.dim = int(dimension_getter())
        # bge-*-v1.5 官方说明短查询无需指令前缀；若召回不理想，可用它做一次 A/B
        self.query_instruction = ""

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        inputs = [str(text) for text in texts]
        if is_query and self.query_instruction:
            inputs = [self.query_instruction + text for text in inputs]
        if not inputs:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            inputs,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """字符 n-gram 哈希向量：确定性、零依赖，用于单测与无模型环境。"""

    def __init__(self, dim: int = 512, ngrams: tuple[int, ...] = (1, 2, 3)) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"
        self._ngrams = ngrams

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        rows = []
        for text in texts:
            content = str(text)
            vector = np.zeros(self.dim, dtype=np.float32)
            for size in self._ngrams:
                for index in range(max(0, len(content) - size + 1)):
                    gram = content[index : index + size]
                    digest = hashlib.md5(gram.encode("utf-8")).digest()
                    bucket = int.from_bytes(digest[:4], "little") % self.dim
                    sign = 1.0 if digest[4] % 2 == 0 else -1.0
                    vector[bucket] += sign
            rows.append(vector)
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.vstack(rows))


def build_embedder(kind: str = "auto", model_name: str | None = None) -> Embedder:
    """kind：auto 优先真实模型、失败退化；bge 强制真实模型；hash 强制哈希向量。"""
    from agent.config import get_env

    # 允许用环境变量指定本地路径或镜像模型名，例如 EMBED_MODEL=/path/to/bge-small-zh-v1.5
    model_name = model_name or get_env("EMBED_MODEL", DEFAULT_MODEL_NAME)
    if kind == "hash":
        return HashingEmbedder()
    if kind in {"auto", "bge"}:
        try:
            return SentenceTransformerEmbedder(model_name)
        except ImportError as exc:
            if kind == "bge":
                raise
            print("[warn] 未安装 sentence-transformers，本次退回哈希向量（仅供流程验证）")
            print(f"       安装命令：pip install sentence-transformers（{exc}）")
            return HashingEmbedder()
        except Exception as exc:  # noqa: BLE001 - 网络不可达/模型不存在等都退化为哈希向量
            hint = (
                "       常见原因：连不上 huggingface.co。可尝试：\n"
                "         export HF_ENDPOINT=https://hf-mirror.com   # 使用镜像下载\n"
                "         export HF_HUB_OFFLINE=1                    # 已有缓存时离线加载\n"
                "       或本次直接加 --embedder hash 跳过模型下载"
            )
            if kind == "bge":
                raise RuntimeError(f"加载向量模型 {model_name} 失败：{exc}\n{hint}") from exc
            print(f"[warn] 加载向量模型 {model_name} 失败，本次退回哈希向量（仅供流程验证）：{exc}")
            print(hint)
            return HashingEmbedder()
    raise ValueError(f"未知 embedder 类型：{kind}")
