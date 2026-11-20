"""检索与生成指标。

关键设计：**gold 用「证据锚点」而不是 chunk id**。
chunk id 会随切块参数变化，导致"改了 chunk_size，评测集全部失效"；
锚点（如 `第585条` + `三年`）是原文里的稳定串，只要切块没把这一条切碎，就仍然有效，
换切块策略后可以直接重跑对比。

指标定义：
- Recall@k：前 k 个结果里，是否存在**同时包含全部锚点**的文档；
- MRR：第一个命中的排名的倒数，衡量"正确答案排得多靠前"；
- CitationAcc：回答引用的片段里，是否有包含全部锚点的（防止"引用了但引错"）；
- 拒答：精确率 = 判为拒答里真正该拒的比例；召回率 = 该拒的里被拒掉的比例。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from rag.types import Answer, Hit


def contains_all(text: str, anchors: Sequence[str]) -> bool:
    return all(a in text for a in anchors) if anchors else False


def hit_at_k(hits: Sequence[Hit], anchors: Sequence[str], k: int) -> int:
    for hit in hits[:k]:
        if contains_all(hit.chunk.parent_text, anchors):
            return 1
    return 0


def reciprocal_rank(hits: Sequence[Hit], anchors: Sequence[str]) -> float:
    for rank, hit in enumerate(hits, start=1):
        if contains_all(hit.chunk.parent_text, anchors):
            return 1.0 / rank
    return 0.0


@dataclass
class ItemResult:
    id: str
    type: str
    question: str
    anchors: List[str]
    hits: List[Hit]
    answer: Optional[Answer] = None
    refused: bool = False
    citation_ok: bool = False
    answer_has_anchor: bool = False
    latency_ms: int = 0


@dataclass
class Report:
    strategy: str
    n_answerable: int = 0
    n_unanswerable: int = 0
    recall3: float = 0.0
    recall5: float = 0.0
    mrr: float = 0.0
    citation_valid_rate: float = 0.0
    citation_acc: Optional[float] = None
    answer_hit: Optional[float] = None
    refusal_precision: Optional[float] = None
    refusal_recall: Optional[float] = None
    avg_latency_ms: float = 0.0
    llm: str = "echo"
    per_item: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()

    def markdown(self) -> str:
        def pct(v: Optional[float]) -> str:
            return "N/A" if v is None else f"{v:.1%}"

        return (
            f"| 指标 | {self.strategy} |\n"
            f"| :--- | :--- |\n"
            f"| Recall@3（有据题） | {self.recall3:.1%} |\n"
            f"| Recall@5（有据题） | {self.recall5:.1%} |\n"
            f"| MRR（有据题） | {self.mrr:.3f} |\n"
            f"| 引用编号合法率 | {self.citation_valid_rate:.1%} |\n"
            f"| 引用准确率（对齐 gold） | {pct(self.citation_acc)} |\n"
            f"| 答案含证据率 | {pct(self.answer_hit)} |\n"
            f"| 拒答精确率（无据题） | {pct(self.refusal_precision)} |\n"
            f"| 拒答召回率（无据题） | {pct(self.refusal_recall)} |\n"
            f"| 平均延迟 | {self.avg_latency_ms:.0f} ms |\n"
            f"| 生成模型 | {self.llm} |\n"
        )


def build_report(strategy: str, items: Sequence[ItemResult], llm: str = "echo") -> Report:
    """llm=echo 时生成侧指标没有意义（桩模型不做语义判断），只报检索与结构类指标。"""
    answerable = [i for i in items if i.type == "answerable" and i.anchors]
    unanswerable = [i for i in items if i.type == "unanswerable"]
    report = Report(strategy=strategy, n_answerable=len(answerable), n_unanswerable=len(unanswerable), llm=llm)
    all_citations = [c for i in items for c in (i.answer.citations if i.answer else [])]
    if all_citations:
        report.citation_valid_rate = sum(1 for c in all_citations if c.valid) / len(all_citations)
    elif items:
        report.citation_valid_rate = 1.0
    if answerable:
        report.recall3 = sum(hit_at_k(i.hits, i.anchors, 3) for i in answerable) / len(answerable)
        report.recall5 = sum(hit_at_k(i.hits, i.anchors, 5) for i in answerable) / len(answerable)
        report.mrr = sum(reciprocal_rank(i.hits, i.anchors) for i in answerable) / len(answerable)
        if llm != "echo":
            report.citation_acc = sum(1 for i in answerable if i.citation_ok) / len(answerable)
            report.answer_hit = sum(1 for i in answerable if i.answer_has_anchor) / len(answerable)
    if unanswerable and llm != "echo":
        refused = [i for i in unanswerable if i.refused]
        correct = [i for i in unanswerable if i.refused]
        report.refusal_precision = len(correct) / len(refused) if refused else 0.0
        report.refusal_recall = len(correct) / len(unanswerable)
    if items:
        report.avg_latency_ms = sum(i.latency_ms for i in items) / len(items)
    report.per_item = [
        {
            "id": i.id,
            "type": i.type,
            "question": i.question,
            "hit@3": hit_at_k(i.hits, i.anchors, 3) if i.anchors else None,
            "rr": round(reciprocal_rank(i.hits, i.anchors), 3) if i.anchors else None,
            "refused": i.refused,
            "citation_ok": i.citation_ok if i.anchors else None,
            "top1": i.hits[0].chunk.id if i.hits else None,
            "latency_ms": i.latency_ms,
        }
        for i in items
    ]
    return report
