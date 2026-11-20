def test_pipeline_answers_with_valid_citations(pipeline):
    answer = pipeline.answer("违约金约定过高，法院可以调整吗", strategy="hybrid_rerank")
    assert not answer.refused
    assert answer.citations, "应当至少有一个引用"
    assert all(c.valid for c in answer.citations)
    assert "第585条" in answer.hits[0].chunk.parent_text
    assert answer.latency_ms >= 0


def test_pipeline_refuses_when_no_hits(pipeline):
    # 纯 ASCII 查询在中文语料上不会命中任何 BM25 文档 -> 走"无召回直接拒答"分支
    answer = pipeline.answer("zzz qqq xxx", strategy="bm25")
    assert answer.refused
    assert answer.text.startswith("根据现有资料无法回答")
    assert answer.citations == []


def test_pipeline_stream_yields_text(pipeline):
    pieces = list(pipeline.stream("结婚年龄是多少", strategy="hybrid"))
    assert pieces
    assert "二十" in "".join(pieces)


def test_retrieve_returns_debug_info(pipeline):
    ctx = pipeline.retrieve("诉讼时效是几年", strategy="hybrid")
    assert ctx.hits
    assert "retrieve_ms" in ctx.debug
    assert ctx.to_prompt_block().startswith("[片段1]")
