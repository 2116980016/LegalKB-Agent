"""混合检索：加权 RRF 融合。

为什么不用"分数加权平均"：向量相似度（0~1）和 BM25 分数（0~20+）量纲不同，
直接加权没有意义。RRF 只用排名，天然可比：

    score(d) = Σ_i w_i / (k + rank_i(d))      k 通常取 60

k 的作用：压低头部排名的绝对优势，避免某一路的第 1 名直接决定结果。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from rag.index.store import IndexStore
from rag.types import Hit


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Hit]],
    weights: Sequence[float],
    k: int = 60,
    top_n: int = 10,
) -> List[Hit]:
    scores: Dict[str, float] = {}
    proto: Dict[str, Hit] = {}
    for hits, weight in zip(ranked_lists, weights):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + weight / (k + rank)
            proto.setdefault(hit.chunk.id, hit)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [Hit(chunk=proto[cid].chunk, score=score, source="hybrid") for cid, score in ordered]


class HybridRetriever:
    def __init__(
        self,
        store: IndexStore,
        k_vec: int = 10,
        k_bm25: int = 10,
        weights: Tuple[float, float] = (0.5, 0.5),
        rrf_k: int = 60,
        strategy: str = "hybrid",
    ):
        self.store = store
        self.k_vec = k_vec
        self.k_bm25 = k_bm25
        self.weights = weights
        self.rrf_k = rrf_k
        self.strategy = strategy

    def retrieve(self, query: str, top_n: int = 5) -> List[Hit]:
        if self.strategy == "vector":
            return self.store.search_vector(query, top_n)
        if self.strategy == "bm25":
            return self.store.search_bm25(query, top_n)
        vec_hits = self.store.search_vector(query, self.k_vec)
        bm_hits = self.store.search_bm25(query, self.k_bm25)
        return rrf_fuse([vec_hits, bm_hits], list(self.weights), k=self.rrf_k, top_n=top_n)
