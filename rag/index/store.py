"""索引的构建、持久化与加载。

持久化设计（面试可讲）：
- `chunks.jsonl`：可读、可 diff、可直接用 grep 排查；
- `vectors.npy`：稠密矩阵，加载是 mmap 级别；
- `meta.json`：记录 embedding 名称/维度/切块参数，**加载时校验**，
  防止"用 A 模型建的库、用 B 模型查"这种静默错误。
BM25 不落盘，启动时从 chunks 重建（1260 条 < 1s），避免 pickle 安全问题和版本兼容问题。
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rag.index.bm25 import BM25, tokenize
from rag.index.embeddings import Embedder
from rag.index.vector_store import NumpyVectorIndex, build_vector_index
from rag.types import Chunk, Hit


@dataclass
class Meta:
    embedder: str
    dim: int
    n_chunks: int
    n_parents: int
    child_size: int
    child_overlap: int
    backend: str

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


class IndexStore:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        vector_index: NumpyVectorIndex,
        bm25: BM25,
        meta: Meta,
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.vector_index = vector_index
        self.bm25 = bm25
        self.meta = meta
        self._by_id = {c.id: c for c in chunks}

    # ---------- 构建 ----------
    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        backend: str = "numpy",
        child_size: int = 300,
        child_overlap: int = 60,
    ) -> IndexStore:
        chunks = list(chunks)
        texts = [c.text for c in chunks]
        if hasattr(embedder, "fit"):  # 例如 HashEmbedding 需要用语料拟合 IDF
            embedder.fit(texts)
        vectors = embedder.encode(texts)
        vector_index = build_vector_index(vectors, backend=backend)
        bm25 = BM25([tokenize(c.text) for c in chunks])
        meta = Meta(
            embedder=embedder.name,
            dim=embedder.dim,
            n_chunks=len(chunks),
            n_parents=len({c.parent_id for c in chunks}),
            child_size=child_size,
            child_overlap=child_overlap,
            backend=vector_index.backend,
        )
        return cls(chunks, embedder, vector_index, bm25, meta)

    # ---------- 持久化 ----------
    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(
                    json.dumps(
                        {"id": c.id, "text": c.text, "parent_id": c.parent_id,
                         "parent_text": c.parent_text, "meta": c.meta},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        self.vector_index.save(str(d))
        (d / "meta.json").write_text(json.dumps(self.meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str, embedder: Embedder) -> IndexStore:
        d = Path(directory)
        meta = Meta(**json.loads((d / "meta.json").read_text(encoding="utf-8")))
        chunks = [
            Chunk(**json.loads(line))
            for line in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if hasattr(embedder, "fit"):
            embedder.fit([c.text for c in chunks])
        if meta.embedder != embedder.name or meta.dim != embedder.dim:
            raise ValueError(
                f"索引由 {meta.embedder}(dim={meta.dim}) 构建，当前 embedder 为 "
                f"{embedder.name}(dim={embedder.dim})；请重建索引或换回对应模型"
            )
        from rag.index.bm25 import tokenize

        bm25 = BM25([tokenize(c.text) for c in chunks])
        vector_index = NumpyVectorIndex.load(str(d))
        return cls(chunks, embedder, vector_index, bm25, meta)

    # ---------- 检索原语 ----------
    def search_vector(self, query: str, top_k: int = 10) -> list[Hit]:
        qv = self.embedder.encode([query])[0]
        return [Hit(chunk=self.chunks[i], score=score, source="vector")
                for i, score in self.vector_index.search(qv, top_k)]

    def search_bm25(self, query: str, top_k: int = 10) -> list[Hit]:
        return [Hit(chunk=self.chunks[i], score=score, source="bm25")
                for i, score in self.bm25.search(query, top_k)]

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)
