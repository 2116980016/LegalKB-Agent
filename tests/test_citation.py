from rag.generate.citation import is_refusal, parse_citations
from rag.types import Hit, RetrievedContext


def _ctx(chunks):
    return RetrievedContext(hits=[Hit(chunk=c, score=1.0) for c in chunks[:3]], query="q")


def test_parse_valid_and_invalid(chunks):
    ctx = _ctx(chunks)
    text = "结论：[片段1]；依据：[片段2]；乱引：[片段9]"
    cites = parse_citations(text, ctx)
    assert [c.index for c in cites] == [1, 2, 9]
    assert [c.valid for c in cites] == [True, True, False]
    assert cites[2].chunk_id == ""


def test_parse_dedupes(chunks):
    ctx = _ctx(chunks)
    cites = parse_citations("A[片段1] B[片段1] C[片段 2]", ctx)
    assert len(cites) == 2


def test_parse_empty_text(chunks):
    assert parse_citations("没有引用", _ctx(chunks)) == []


def test_is_refusal():
    assert is_refusal("根据现有资料无法回答")
    assert is_refusal("根据现有资料无法回答。")
    assert not is_refusal("根据《民法典》第585条，违约金过高可以请求适当减少。[片段1]")
