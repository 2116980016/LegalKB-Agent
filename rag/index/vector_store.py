"""向量索引：numpy 精确检索（默认）与 FAISS 可选后端。

小数据集（<10 万块）用 numpy 的矩阵乘法完全够用，且零依赖、可存成 .npy 直接进 Git LFS。
数据量上去再切 FAISS / Qdrant，接口不变——这就是分层的意义。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NumpyVectorIndex:
    backend = "numpy"

    def __init__(self, vectors: np.ndarray):
        if vectors.ndim != 2:
            raise ValueError("vectors 必须是二维矩阵")
        self.vectors = vectors.astype(np.float32)

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        if len(self) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        scores = self.vectors @ q
        k = min(top_k, len(self))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        (d / "vector_meta.json").write_text(
            json.dumps({"backend": self.backend, "n": len(self), "dim": self.dim}), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str) -> NumpyVectorIndex:
        d = Path(directory)
        return cls(np.load(d / "vectors.npy"))


class FaissVectorIndex(NumpyVectorIndex):
    """需要 faiss-cpu；不可用时自动回退到 numpy。"""

    backend = "faiss"

    def __init__(self, vectors: np.ndarray):
        super().__init__(vectors)
        import faiss  # 延迟导入

        self._faiss = faiss
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.vectors)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        scores, idx = self.index.search(q, min(top_k, len(self)))
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]


def build_vector_index(vectors: np.ndarray, backend: str = "numpy") -> NumpyVectorIndex:
    if (backend or "numpy").lower() == "faiss":
        try:
            return FaissVectorIndex(vectors)
        except Exception:  # faiss 未安装 -> 降级
            return NumpyVectorIndex(vectors)
    return NumpyVectorIndex(vectors)
