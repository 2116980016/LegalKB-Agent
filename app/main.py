"""FastAPI 入口。

接口设计原则（面试可讲）：
- `/api/chat` 同步返回，便于评测脚本和脚本化调用；
- `/api/chat/stream` 走 SSE，便于前端逐字渲染；
- 两个接口共用 `RAGService`，保证"评测的和上线的是同一条链路"。
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.logging import log, new_request_id, setup_logging
from app.services.rag_service import RAGService

settings = Settings()
setup_logging(settings.log_level)
_service: Optional[RAGService] = None


def get_service() -> RAGService:
    if _service is None:
        raise HTTPException(status_code=503, detail="索引尚未就绪")
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    log("service.starting", **settings.describe())
    _service = RAGService.create(settings)
    log("service.ready", chunks=len(_service.store.chunks))
    yield
    _service = None


app = FastAPI(title="LegalKB-Agent", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """配置了 API_KEY 才启用鉴权；本地开发默认关闭。"""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="无效的 API Key")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_n: Optional[int] = Field(default=None, ge=1, le=20)
    strategy: Optional[str] = Field(default=None, description="vector | bm25 | hybrid | hybrid_rerank")


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    citations: list
    hits: list
    latency_ms: int
    request_id: str


@app.get("/health")
def health() -> dict:
    service = get_service()
    return {
        "status": "ok",
        "index": service.store.meta.to_dict(),
        "config": settings.describe(),
        "stats": service.stats.snapshot(),
    }


@app.get("/api/stats")
def stats() -> dict:
    return get_service().stats.snapshot()


@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest) -> ChatResponse:
    rid = new_request_id()
    service = get_service()
    try:
        answer = service.answer(req.question, top_n=req.top_n, strategy=req.strategy)
    except Exception as exc:  # 不把堆栈暴露给调用方
        service.stats.errors += 1
        log("chat.error", level=40, error=str(exc))
        raise HTTPException(status_code=500, detail="生成失败，请稍后重试")
    return ChatResponse(
        answer=answer.text,
        refused=answer.refused,
        citations=[
            {"index": c.index, "chunk_id": c.chunk_id, "text": c.text, "valid": c.valid}
            for c in answer.citations
        ],
        hits=[
            {"index": i, "chunk_id": h.chunk.id, "score": round(h.score, 4), "text": h.chunk.text[:200]}
            for i, h in enumerate(answer.hits, start=1)
        ],
        latency_ms=answer.latency_ms,
        request_id=rid,
    )


@app.post("/api/chat/stream", dependencies=[Depends(require_api_key)])
def chat_stream(req: ChatRequest) -> StreamingResponse:
    rid = new_request_id()
    service = get_service()

    def event_stream():
        pieces = []
        try:
            for delta in service.stream(req.question, top_n=req.top_n, strategy=req.strategy):
                pieces.append(delta)
                yield f"event: delta\ndata: {json.dumps({'text': delta}, ensure_ascii=False)}\n\n"
            full = "".join(pieces)
            ctx = service.retrieve(req.question, strategy=req.strategy or settings.strategy,
                                   top_n=req.top_n or settings.top_n)
            from rag.generate.citation import parse_citations

            citations = parse_citations(full, ctx)
            yield f"event: citations\ndata: {json.dumps([{'index': c.index, 'chunk_id': c.chunk_id, 'valid': c.valid} for c in citations], ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            log("chat.stream_error", level=40, error=str(exc))
            yield f"event: error\ndata: {json.dumps({'message': '生成失败'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Request-Id": rid},
    )


@app.post("/api/documents/rebuild", dependencies=[Depends(require_api_key)])
def rebuild() -> dict:
    return get_service().rebuild()
