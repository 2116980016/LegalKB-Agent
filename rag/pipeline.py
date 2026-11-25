"""RAG 主流程：检索 → 重排 → 拼 prompt → 生成 → 解析引用。

把「检索策略」做成参数字符串（vector / bm25 / hybrid / hybrid_rerank），
是为了让评测脚本和线上接口共用同一条代码路径——否则你评测的和你上线的不是同一个东西。
"""
from __future__ import annotations

import time
from collections.abc import Iterator

from rag.generate.citation import is_refusal, parse_citations
from rag.generate.llm import EchoLLM
from rag.generate.prompt import build_messages
from rag.index.store import IndexStore
from rag.retrieve.hybrid import HybridRetriever
from rag.retrieve.rerank import IdentityReranker, build_reranker
from rag.types import Answer, Hit, RetrievedContext


class RAGPipeline:
    def __init__(
        self,
        store: IndexStore,
        llm=None,
        k_vec: int = 10,
        k_bm25: int = 10,
        weights: tuple[float, float] = (0.5, 0.5),
        top_n: int = 5,
        reranker: str = "overlap",
    ):
        self.store = store
        self.llm = llm or EchoLLM()
        self.k_vec = k_vec
        self.k_bm25 = k_bm25
        self.weights = weights
        self.top_n = top_n
        self.reranker = build_reranker(reranker) if reranker else IdentityReranker()

    # ---------- 检索 ----------
    def retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        top_n: int | None = None,
        rerank: bool = True,
    ) -> RetrievedContext:
        top_n = top_n or self.top_n
        retriever = HybridRetriever(
            self.store,
            k_vec=self.k_vec,
            k_bm25=self.k_bm25,
            weights=self.weights,
            strategy=strategy,
        )
        started = time.perf_counter()
        if strategy == "hybrid_rerank":
            candidates = retriever.retrieve(query, top_n=max(top_n, self.k_vec))
            hits: list[Hit] = self.reranker.rerank(query, candidates, top_n=top_n) if rerank else candidates[:top_n]
        else:
            hits = retriever.retrieve(query, top_n=top_n)
        elapsed = (time.perf_counter() - started) * 1000
        return RetrievedContext(hits=hits, query=query, strategy=strategy, debug={"retrieve_ms": round(elapsed, 1)})

    # ---------- 生成 ----------
    def answer(self, question: str, strategy: str = "hybrid_rerank", top_n: int | None = None) -> Answer:
        started = time.perf_counter()
        ctx = self.retrieve(question, strategy=strategy, top_n=top_n)
        if not ctx.hits:
            return Answer(text="根据现有资料无法回答", citations=[], hits=[], refused=True,
                          latency_ms=int((time.perf_counter() - started) * 1000))
        messages = build_messages(question, ctx.to_prompt_block())
        resp = self.llm.complete(messages)
        citations = parse_citations(resp.text, ctx)
        return Answer(
            text=resp.text,
            citations=citations,
            hits=ctx.hits,
            refused=is_refusal(resp.text),
            latency_ms=int((time.perf_counter() - started) * 1000),
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
        )

    def stream(self, question: str, strategy: str = "hybrid_rerank", top_n: int | None = None) -> Iterator[str]:
        """逐段产出回答文本；调用方负责在结束后补充引用信息。"""
        ctx = self.retrieve(question, strategy=strategy, top_n=top_n)
        if not ctx.hits:
            yield "根据现有资料无法回答"
            return
        messages = build_messages(question, ctx.to_prompt_block())
        for piece in self.llm.stream(messages):
            yield piece
