"""服务层：把配置、索引、流水线串起来，并记录运行指标。"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from app.core.config import Settings
from app.core.logging import log
from rag.generate.llm import build_llm
from rag.index.embeddings import build_embedder
from rag.index.store import IndexStore
from rag.ingest.loader import load_documents
from rag.ingest.splitter import chunk_stats, split_parent_child
from rag.pipeline import RAGPipeline
from rag.types import Answer, RetrievedContext


@dataclass
class Stats:
    requests: int = 0
    refusals: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: List[int] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, answer: Answer) -> None:
        with self._lock:
            self.requests += 1
            self.refusals += 1 if answer.refused else 0
            self.prompt_tokens += answer.prompt_tokens
            self.completion_tokens += answer.completion_tokens
            self.latency_ms.append(answer.latency_ms)
            if len(self.latency_ms) > 1000:
                self.latency_ms = self.latency_ms[-500:]

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            n = len(self.latency_ms)
            avg = sum(self.latency_ms) / n if n else 0.0
            p95 = sorted(self.latency_ms)[int(n * 0.95) - 1] if n else 0
            return {
                "requests": self.requests,
                "refusals": self.refusals,
                "errors": self.errors,
                "refusal_rate": round(self.refusals / self.requests, 3) if self.requests else 0.0,
                "avg_latency_ms": round(avg, 1),
                "p95_latency_ms": p95,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }


def _embedder(settings: Settings):
    return build_embedder(
        settings.embedding_provider,
        dim=settings.embedding_dim,
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        model_path=settings.embedding_path,
    )


def build_store(settings: Settings) -> IndexStore:
    started = time.perf_counter()
    docs = load_documents(settings.corpus_path)
    chunks = split_parent_child(docs, child_size=settings.child_size, child_overlap=settings.child_overlap)
    embedder = _embedder(settings)
    store = IndexStore.build(
        chunks,
        embedder,
        backend=settings.vector_backend,
        child_size=settings.child_size,
        child_overlap=settings.child_overlap,
    )
    store.save(settings.index_dir)
    log("index.built", docs=len(docs), chunks=chunk_stats(chunks),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1), index_dir=settings.index_dir)
    return store


def load_or_build_store(settings: Settings) -> IndexStore:
    embedder = _embedder(settings)
    meta_path = Path(settings.index_dir) / "meta.json"
    if meta_path.exists():
        try:
            store = IndexStore.load(settings.index_dir, embedder)
            log("index.loaded", chunks=len(store.chunks), index_dir=settings.index_dir)
            return store
        except Exception as exc:  # 配置变更/索引损坏 -> 重建
            log("index.load_failed", level=30, error=str(exc))
    return build_store(settings)


class RAGService:
    def __init__(self, settings: Settings, store: IndexStore):
        self.settings = settings
        self.store = store
        self.stats = Stats()
        llm = build_llm(
            settings.llm_provider,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )
        self.pipeline = RAGPipeline(
            store,
            llm=llm,
            k_vec=settings.k_vec,
            k_bm25=settings.k_bm25,
            weights=settings.weights,
            top_n=settings.top_n,
            reranker=settings.reranker,
        )

    @classmethod
    def create(cls, settings: Optional[Settings] = None) -> "RAGService":
        settings = settings or Settings()
        return cls(settings, load_or_build_store(settings))

    def rebuild(self) -> Dict[str, object]:
        self.store = build_store(self.settings)
        self.pipeline.store = self.store
        return {"chunks": len(self.store.chunks), "parents": self.store.meta.n_parents}

    def answer(self, question: str, top_n: Optional[int] = None, strategy: Optional[str] = None) -> Answer:
        answer = self.pipeline.answer(
            question,
            strategy=strategy or self.settings.strategy,
            top_n=top_n or self.settings.top_n,
        )
        self.stats.record(answer)
        log("chat.answered", question_len=len(question), refused=answer.refused,
            latency_ms=answer.latency_ms, citations=len(answer.citations), hits=len(answer.hits))
        return answer

    def stream(self, question: str, top_n: Optional[int] = None, strategy: Optional[str] = None) -> Iterator[str]:
        return self.pipeline.stream(
            question,
            strategy=strategy or self.settings.strategy,
            top_n=top_n or self.settings.top_n,
        )

    def retrieve(self, question: str, strategy: str = "hybrid_rerank", top_n: int = 5) -> RetrievedContext:
        return self.pipeline.retrieve(question, strategy=strategy, top_n=top_n)
