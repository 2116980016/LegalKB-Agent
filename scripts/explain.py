"""把一次问答的每一步打印出来：切块 → 向量召回 → BM25 召回 → 融合 → 重排 → prompt → 回答 → 引用。

用途：
1. 自己搞懂链路（不看代码也能看懂每一步在干什么）；
2. 面试时现场演示"我能解释每一步"；
3. 排查"为什么这题答错了"——是没召回、还是召回错、还是模型没用对。

用法：
    python -m scripts.explain "违约金约定过高，法院可以调整吗"
    python -m scripts.explain "..." --top 3 --no-prompt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.services.rag_service import load_or_build_store  # noqa: E402
from rag.generate.prompt import build_messages  # noqa: E402
from rag.retrieve.hybrid import rrf_fuse  # noqa: E402

LINE = "─" * 78


def title(text: str) -> None:
    print(f"\n{LINE}\n{text}\n{LINE}")


def show(hits, label: str, limit: int = 5) -> None:
    if not hits:
        print(f"  （{label} 无结果）")
        return
    for i, h in enumerate(hits[:limit], start=1):
        art = h.chunk.meta.get("article", "")
        text = h.chunk.text.replace("\n", " ")
        print(f"  {i}. 分数={h.score:7.4f}  {art:8s} {text[:46]}…")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+", help="要问的问题")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--no-prompt", action="store_true", help="不打印完整 prompt")
    parser.add_argument("--strategy", default="hybrid_rerank")
    args = parser.parse_args()
    question = " ".join(args.question)

    settings = Settings()
    t0 = time.perf_counter()

    title("① 载入索引（离线产物，不用重新训练）")
    store = load_or_build_store(settings)
    print(f"  语料：{settings.corpus_path}")
    print(f"  子块 {len(store.chunks)} 个 / 父块 {store.meta.n_parents} 个")
    print(f"  向量模型：{store.meta.embedder}（维度 {store.meta.dim}）")
    print(f"  加载耗时：{(time.perf_counter() - t0) * 1000:.0f} ms")

    title(f"② 用户问题\n  「{question}」")

    title("③ 向量召回（语义相似：换一种说法也能找到）")
    t = time.perf_counter()
    vec_hits = store.search_vector(question, top_k=settings.k_vec)
    vec_ms = (time.perf_counter() - t) * 1000
    show(vec_hits, "向量")
    print(f"  向量检索耗时：{vec_ms:.1f} ms")

    title("③' BM25 召回（字面匹配：法条号、专有名词靠它）")
    t = time.perf_counter()
    bm_hits = store.search_bm25(question, top_k=settings.k_bm25)
    bm_ms = (time.perf_counter() - t) * 1000
    show(bm_hits, "BM25")
    print(f"  BM25 检索耗时：{bm_ms:.1f} ms")

    title("④ 加权 RRF 融合（只比排名，不比分数——两路分数量纲不同）")
    fused = rrf_fuse(
        [vec_hits, bm_hits],
        [settings.weight_vector, settings.weight_bm25],
        k=settings.rrf_k,
        top_n=args.top,
    )
    for i, h in enumerate(fused, start=1):
        print(f"  {i}. 融合分={h.score:.5f}  {h.chunk.meta.get('article', ''):8s} {h.chunk.text[:44]}…")
    print(f"  公式：score = Σ 权重 /（60 + 该路排名）；权重 = {settings.weights}")

    title("⑤ 重排（可选：对候选再打一次分，取前 N 交给模型）")
    print(f"  当前配置：RERANKER={settings.reranker}（none 表示不重排，直接用融合结果）")

    title("⑥ 拼 prompt（把父块全文塞进去，并编号；模型只能看这些）")
    from app.services.rag_service import RAGService  # noqa: E402

    service = RAGService(settings, store)
    ctx = service.retrieve(question, strategy=args.strategy, top_n=args.top)
    messages = build_messages(question, ctx.to_prompt_block())
    if args.no_prompt:
        print(f"  prompt 长度：{len(messages[-1]['content'])} 字符（已省略）")
    else:
        preview = messages[-1]["content"]
        head = preview[:700]
        print(head + ("\n  …（以下省略）" if len(preview) > 700 else ""))

    title("⑦ 生成回答（当前配置的 LLM）")
    t = time.perf_counter()
    try:
        answer = service.answer(question, top_n=args.top, strategy=args.strategy)
    except Exception as exc:  # 模型不可用（欠费/无 key/网络）时仍然演示链路
        print(f"  ⚠️ 真实模型调用失败：{str(exc)[:160]}")
        print("  → 已回退到离线桩（echo）演示链路；回答是检索片段的摘录，不代表真实模型效果")
        from rag.generate.llm import EchoLLM  # noqa: E402

        service.pipeline.llm = EchoLLM()
        answer = service.answer(question, top_n=args.top, strategy=args.strategy)
    print(answer.text)
    print(f"\n  生成耗时（机器计时）：{(time.perf_counter() - t) * 1000:.0f} ms")
    print(f"  LLM 自报时延：{answer.latency_ms} ms")

    title("⑧ 引用校验（把 [片段n] 对回真实片段，越界会被标红）")
    if not answer.citations:
        print("  没有引用（可能拒答了）")
    for c in answer.citations:
        flag = "✓" if c.valid else "✗ 编号不存在"
        print(f"  [片段{c.index}] {flag}  {c.text[:50]}…")

    title("⑨ 全链路小结")
    total_ms = (time.perf_counter() - t0) * 1000
    print(f"  召回：向量 {vec_ms:.0f} ms + BM25 {bm_ms:.0f} ms")
    print(f"  生成：{answer.latency_ms} ms")
    print(f"  端到端：{total_ms:.0f} ms（含进程启动与索引加载）")
    print("\n  下一步：想让答案更准 → 看 ③ 有没有召回正确法条；")
    print("          召回了但答错 → 看 ⑥ 的 prompt 和 ⑦ 的模型输出；")
    print("          答案对但引用错 → 看 ⑧ 的编号映射。")


if __name__ == "__main__":
    main()
