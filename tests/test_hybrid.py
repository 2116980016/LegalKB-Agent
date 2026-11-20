from rag.index.store import IndexStore
from rag.retrieve.hybrid import HybridRetriever, rrf_fuse
from rag.types import Hit


def _hit(chunk, score, source):
    return Hit(chunk=chunk, score=score, source=source)


def test_rrf_prefers_doc_hit_by_both_lists(chunks):
    a, b, c = chunks[0], chunks[1], chunks[2]
    list1 = [_hit(a, 0.9, "vector"), _hit(b, 0.8, "vector")]
    list2 = [_hit(a, 5.0, "bm25"), _hit(c, 4.0, "bm25")]
    fused = rrf_fuse([list1, list2], [0.5, 0.5], k=60, top_n=3)
    assert fused[0].chunk.id == a.id
    # a 在两路都排第一，分数应为两路之和
    assert abs(fused[0].score - (0.5 / 61 + 0.5 / 61)) < 1e-9


def test_rrf_respects_weights(chunks):
    a, b = chunks[0], chunks[1]
    fused = rrf_fuse([[_hit(a, 1, "v")], [_hit(b, 1, "b")]], [0.9, 0.1], k=60, top_n=2)
    assert fused[0].chunk.id == a.id


def test_rrf_top_n_truncates(chunks):
    hits = [_hit(c, 1, "v") for c in chunks]
    assert len(rrf_fuse([hits], [1.0], top_n=2)) == 2


def test_hybrid_returns_hits_with_source(store):
    retriever = HybridRetriever(store, k_vec=3, k_bm25=3, strategy="hybrid")
    hits = retriever.retrieve("违约金过高可以请求减少吗", top_n=2)
    assert hits
    assert hits[0].source == "hybrid"
    assert "违约金" in hits[0].chunk.parent_text


def test_vector_and_bm25_strategies_work(store):
    assert HybridRetriever(store, strategy="vector").retrieve("结婚年龄", top_n=1)
    bm_hits = HybridRetriever(store, strategy="bm25").retrieve("结婚年龄", top_n=1)
    assert bm_hits and "结婚年龄" in bm_hits[0].chunk.parent_text
