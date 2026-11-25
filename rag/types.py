"""核心数据结构：父子块、检索结果、引用。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """子块：参与检索的最小单位；parent_text 用于喂给大模型。"""

    id: str
    text: str
    parent_id: str
    parent_text: str
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def display_text(self) -> str:
        return self.text.strip()


@dataclass
class Hit:
    chunk: Chunk
    score: float
    source: str = "hybrid"  # vector | bm25 | hybrid | rerank

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass
class Citation:
    index: int          # 回答里的 [片段n] 中的 n
    chunk_id: str
    text: str           # 展示用（截断）
    valid: bool = True  # 编号是否真实存在于本次检索结果中


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    hits: list[Hit]
    refused: bool = False
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def invalid_citations(self) -> list[Citation]:
        return [c for c in self.citations if not c.valid]


@dataclass
class RetrievedContext:
    hits: list[Hit]
    query: str
    strategy: str = "hybrid"
    debug: dict[str, float] = field(default_factory=dict)

    def to_prompt_block(self) -> str:
        lines = []
        for i, hit in enumerate(self.hits, start=1):
            lines.append(f"[片段{i}] {hit.chunk.parent_text.strip()}")
        return "\n\n".join(lines)

    def find(self, index: int) -> Hit | None:
        if 1 <= index <= len(self.hits):
            return self.hits[index - 1]
        return None
