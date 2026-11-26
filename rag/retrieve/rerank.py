"""重排。

三种实现，用于做消融实验：
1. IdentityReranker —— 不重排（对照组）；
2. OverlapReranker —— 词面覆盖率 + 原始分，零依赖，可作为"没有 GPU 时的兜底"；
3. CrossEncoderReranker —— bge-reranker，query 与 doc 拼接后一起过编码器，
   精度显著高于双塔，但只能对小候选集做（延迟随候选数线性增长）。
"""
from __future__ import annotations

from collections.abc import Sequence

from rag.index.bm25 import tokenize
from rag.types import Hit


class IdentityReranker:
    name = "identity"

    def rerank(self, query: str, hits: Sequence[Hit], top_n: int = 5) -> list[Hit]:
        return list(hits[:top_n])


class OverlapReranker:
    """query token 覆盖率 + 归一化 RRF 分，按 0.7/0.3 混合。"""

    name = "overlap"

    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha

    def rerank(self, query: str, hits: Sequence[Hit], top_n: int = 5) -> list[Hit]:
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return list(hits[:top_n])
        max_score = max((h.score for h in hits), default=1.0) or 1.0
        scored = []
        for h in hits:
            d_tokens = set(tokenize(h.chunk.text))
            coverage = len(q_tokens & d_tokens) / len(q_tokens)
            norm = h.score / max_score
            scored.append(Hit(chunk=h.chunk, score=self.alpha * coverage + (1 - self.alpha) * norm, source="rerank"))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_n]


class FastEmbedReranker:
    """基于 fastembed 的 cross-encoder 重排（ONNX，不需要 torch）。

    与 CrossEncoderReranker 的区别：只依赖 fastembed + onnxruntime（约 50MB），
    不用装 2GB 的 torch，适合在 CPU / 笔记本上验证"重排到底有没有用"。
    """

    name = "bge-reranker-onnx"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", model_path: str = ""):
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # 延迟导入

        kwargs = {"model_name": model_name}
        if model_path:
            kwargs["specific_model_path"] = model_path
        self.model = TextCrossEncoder(**kwargs)
        self.model_name = model_name

    def rerank(self, query: str, hits: Sequence[Hit], top_n: int = 5) -> list[Hit]:
        if not hits:
            return []
        docs = [h.chunk.parent_text or h.chunk.text for h in hits]
        scores = list(self.model.rerank(query, docs))
        out = [Hit(chunk=h.chunk, score=float(s), source="rerank") for h, s in zip(hits, scores)]
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_n]


class CrossEncoderReranker:
    name = "bge-reranker"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str | None = None):
        from sentence_transformers import CrossEncoder  # 延迟导入

        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name

    def rerank(self, query: str, hits: Sequence[Hit], top_n: int = 5) -> list[Hit]:
        if not hits:
            return []
        pairs = [(query, h.chunk.parent_text or h.chunk.text) for h in hits]
        scores = self.model.predict(pairs)
        out = [Hit(chunk=h.chunk, score=float(s), source="rerank") for h, s in zip(hits, scores)]
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_n]


def build_reranker(kind: str = "overlap", **kwargs):
    kind = (kind or "overlap").lower()
    if kind in {"none", "identity", ""}:
        return IdentityReranker()
    if kind in {"overlap", "lexical"}:
        return OverlapReranker(**kwargs)
    if kind in {"onnx", "fastembed"}:
        return FastEmbedReranker(**kwargs)
    if kind in {"bge", "cross-encoder", "crossencoder"}:
        return CrossEncoderReranker(**kwargs)
    raise ValueError(f"未知的 reranker: {kind}")
