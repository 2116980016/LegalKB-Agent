"""评测入口：一条命令出报告。

用法：
    python -m eval.run --strategy hybrid_rerank
    python -m eval.run --all        # 四种策略横向对比

输出：
    results/<strategy>.json   机器可读
    results/<strategy>.md     可直接贴进 README 的表格
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.services.rag_service import RAGService  # noqa: E402
from eval.metrics import ItemResult, build_report, contains_all  # noqa: E402
from rag.types import Hit  # noqa: E402

STRATEGIES = ["vector", "bm25", "hybrid", "hybrid_rerank"]


def load_dataset(path: str) -> List[dict]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def run_strategy(service: RAGService, dataset: List[dict], strategy: str, top_k: int) -> List[ItemResult]:
    results: List[ItemResult] = []
    for item in dataset:
        started = time.perf_counter()
        ctx = service.retrieve(item["question"], strategy=strategy, top_n=max(top_k, 5))
        answer = service.answer(item["question"], strategy=strategy, top_n=max(top_k, 5))
        latency = int((time.perf_counter() - started) * 1000)
        anchors = item.get("anchors", [])
        citation_ok = False
        if anchors:
            for cite in answer.citations:
                hit: Hit | None = ctx.find(cite.index) if cite.valid else None
                if hit and contains_all(hit.chunk.parent_text, anchors):
                    citation_ok = True
                    break
        results.append(
            ItemResult(
                id=item["id"],
                type=item["type"],
                question=item["question"],
                anchors=anchors,
                hits=ctx.hits,
                answer=answer,
                refused=answer.refused,
                citation_ok=citation_ok,
                answer_has_anchor=contains_all(answer.text, anchors) if anchors else False,
                latency_ms=latency,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="hybrid_rerank", choices=STRATEGIES)
    parser.add_argument("--all", action="store_true", help="跑全部策略并输出对比表")
    parser.add_argument("--dataset", default="eval/dataset/gold.jsonl")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--llm", default="", help="echo（离线）或 dashscope（需 Key）")
    parser.add_argument("--embedding", default="", help="hash / bge")
    args = parser.parse_args()

    settings = Settings()
    if args.llm:
        settings.llm_provider = args.llm
    if args.embedding:
        settings.embedding_provider = args.embedding
    service = RAGService.create(settings)

    dataset = load_dataset(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    strategies = STRATEGIES if args.all else [args.strategy]
    reports = []

    for strategy in strategies:
        items = run_strategy(service, dataset, strategy, args.top_k)
        report = build_report(strategy, items, llm=settings.llm_provider)
        reports.append(report)
        payload = report.to_dict()
        payload["config"] = {
            "embedding": settings.embedding_provider,
            "llm": settings.llm_provider,
            "k_vec": settings.k_vec,
            "k_bm25": settings.k_bm25,
            "reranker": settings.reranker,
            "dataset": args.dataset,
        }
        (out_dir / f"{strategy}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / f"{strategy}.md").write_text(report.markdown(), encoding="utf-8")
        print(f"\n=== {strategy} ===")
        print(report.markdown())

    if len(reports) > 1:
        header = "| 策略 | Recall@3 | Recall@5 | MRR | 引用合法率 | 引用准确率 | 拒答召回 | 延迟 |"
        sep = "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"

        def pct(v):
            return "N/A" if v is None else f"{v:.1%}"

        rows = [
            f"| {r.strategy} | {r.recall3:.1%} | {r.recall5:.1%} | {r.mrr:.3f} | "
            f"{r.citation_valid_rate:.1%} | {pct(r.citation_acc)} | {pct(r.refusal_recall)} | {r.avg_latency_ms:.0f} ms |"
            for r in reports
        ]
        table = "\n".join([header, sep] + rows)
        (out_dir / "summary.md").write_text(table + "\n", encoding="utf-8")
        print("\n=== 对比 ===")
        print(table)


if __name__ == "__main__":
    main()
