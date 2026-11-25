"""引用解析与校验。

大模型最常见的两类引用错误：
1. **编号越界**：引用了 [片段7]，但本次只给了 3 个片段；
2. **张冠李戴**：编号合法，但该片段其实不支持这句话。

这里能自动兜住第 1 类（结构性校验），第 2 类需要人工/LLM 评判，
但可以被 `citation_accuracy` 指标量化（见 eval/metrics.py）。
"""
from __future__ import annotations

import re

from rag.types import Answer, Citation, RetrievedContext

CITE_RE = re.compile(r"\[片段\s*(\d+)\]")


def parse_citations(text: str, ctx: RetrievedContext, max_len: int = 120) -> list[Citation]:
    seen = set()
    citations: list[Citation] = []
    for raw in CITE_RE.findall(text):
        idx = int(raw)
        if idx in seen:
            continue
        seen.add(idx)
        hit = ctx.find(idx)
        citations.append(
            Citation(
                index=idx,
                chunk_id=hit.chunk.id if hit else "",
                text=(hit.chunk.parent_text[:max_len] if hit else ""),
                valid=hit is not None,
            )
        )
    return citations


def is_refusal(text: str, refusal_phrase: str = "根据现有资料无法回答") -> bool:
    """以固定句式为主，同时容忍模型多说一句"抱歉"。"""
    stripped = text.strip()
    return stripped.startswith(refusal_phrase) or (refusal_phrase in stripped and len(stripped) < 60)


def attach_citations(answer: Answer, ctx: RetrievedContext) -> Answer:
    answer.citations = parse_citations(answer.text, ctx)
    return answer
