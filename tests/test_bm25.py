from rag.index.bm25 import BM25, tokenize


def test_tokenize_adds_bigrams():
    tokens = tokenize("违约金")
    assert "违" in tokens
    assert "违约" in tokens
    assert "约金" in tokens


def test_tokenize_breaks_at_punctuation():
    tokens = tokenize("你好，世界！")
    assert "你好" in tokens and "世界" in tokens
    # 标点处必须断开，不能跨标点组成 bigram
    assert "好世" not in tokens


def test_bm25_ranks_relevant_doc_first():
    docs = [
        "当事人可以约定违约金，违约金过高可以请求减少。",
        "结婚年龄男不得早于二十二周岁。",
        "诉讼时效期间为三年。",
    ]
    bm = BM25.from_texts(docs)
    ranked = bm.search("违约金过高怎么办", top_k=3)
    assert ranked, "应当至少命中一个文档"
    assert ranked[0][0] == 0


def test_bm25_no_match_returns_empty():
    bm = BM25.from_texts(["apple banana", "cat dog"])
    assert bm.search("量子力学", top_k=3) == []


def test_idf_is_positive():
    bm = BM25.from_texts(["a b", "a c", "d"])
    assert all(v > 0 for v in bm.idf.values())
