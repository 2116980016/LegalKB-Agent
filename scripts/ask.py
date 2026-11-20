"""命令行问答：python -m scripts.ask "违约金太高可以调整吗"
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.services.rag_service import RAGService  # noqa: E402


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or "违约金约定过高，法院可以调整吗？"
    service = RAGService.create(Settings())
    answer = service.answer(question)
    print(f"\n问：{question}\n")
    print(f"答：{answer.text}\n")
    print(f"引用：{[ (c.index, c.valid) for c in answer.citations ]}")
    print(f"耗时：{answer.latency_ms} ms")
    for i, hit in enumerate(answer.hits, start=1):
        print(f"  [片段{i}] {hit.chunk.id} score={hit.score:.4f} {hit.chunk.text[:40]}...")


if __name__ == "__main__":
    main()
