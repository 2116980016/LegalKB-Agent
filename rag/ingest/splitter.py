"""父子切块。

动机：法律条文「检索单元」和「理解单元」不是同一个粒度。
- 子块（~300 字）用于精确检索，避免长文本稀释语义；
- 父块（一条完整法条）用于喂给大模型，避免上下文被截断。

面试可讲：为什么不直接 500/50 一刀切？因为一个法条被切成两半后，
「前半句有主体、后半句有后果」，模型拿到半句会答错。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from rag.types import Chunk


def split_parent_child(
    documents: List[Dict[str, str]],
    child_size: int = 300,
    child_overlap: int = 60,
) -> List[Chunk]:
    """documents: [{"id": "第585条", "text": "...", "code": "CIVIL_CODE"}]"""
    chunks: List[Chunk] = []
    for doc in documents:
        parent_id = f"p_{doc['id']}"
        parent_text = doc["text"].strip()
        if not parent_text:
            continue
        windows = _windows(parent_text, child_size, child_overlap)
        for j, win in enumerate(windows):
            chunks.append(
                Chunk(
                    id=f"{parent_id}_c{j}",
                    text=win,
                    parent_id=parent_id,
                    parent_text=parent_text,
                    meta={"article": doc.get("id", ""), "code": doc.get("code", ""), "part": str(j)},
                )
            )
    return chunks


def _windows(text: str, size: int, overlap: int) -> List[str]:
    if len(text) <= size:
        return [text]
    step = max(size - overlap, 1)
    out: List[str] = []
    start = 0
    while start < len(text):
        out.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += step
    return out


def chunk_stats(chunks: List[Chunk]) -> Dict[str, float]:
    if not chunks:
        return {"n": 0}
    sizes = [len(c.text) for c in chunks]
    parents = {c.parent_id for c in chunks}
    return {
        "n": len(chunks),
        "n_parents": len(parents),
        "avg_size": round(sum(sizes) / len(sizes), 1),
        "max_size": max(sizes),
        "min_size": min(sizes),
        "children_per_parent": round(len(chunks) / len(parents), 2),
    }
