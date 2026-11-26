"""命令行问答：python -m scripts.ask "违约金太高可以调整吗"

没有可用的 LLM（无 Key / 欠费 / 断网）时自动回退到离线桩 echo，
并在输出里明确标注——避免"以为是真实模型效果"。
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
    settings = Settings()
    service = RAGService.create(settings)

    fell_back = False
    try:
        answer = service.answer(question)
    except Exception as exc:  # 模型不可用：仍然演示检索 + 引用链路
        print(f"\n⚠️ 真实模型不可用：{str(exc)[:150]}")
        print("   → 已回退到离线桩 echo，回答是检索片段摘录，不代表真实模型效果")
        print("   → 要真实回答：在 .env 配好 DASHSCOPE_API_KEY，或设 LLM_PROVIDER=echo 消除告警\n")
        from rag.generate.llm import EchoLLM  # noqa: E402

        service.pipeline.llm = EchoLLM()
        answer = service.answer(question)
        fell_back = True

    print(f"\n问：{question}\n")
    print(f"答：{answer.text}\n")
    print(f"引用：{[(c.index, c.valid) for c in answer.citations]}")
    print(f"耗时：{answer.latency_ms} ms" + ("（离线桩）" if fell_back else ""))
    for i, hit in enumerate(answer.hits, start=1):
        print(f"  [片段{i}] {hit.chunk.id} score={hit.score:.4f} {hit.chunk.text[:40]}...")


if __name__ == "__main__":
    main()
