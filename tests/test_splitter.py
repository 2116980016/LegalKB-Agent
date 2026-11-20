from rag.ingest.splitter import _windows, chunk_stats, split_parent_child


def test_children_are_substrings_of_parent(chunks):
    for c in chunks:
        assert c.text in c.parent_text
        assert c.parent_id.startswith("p_")


def test_ids_are_unique(chunks):
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_long_text_is_split_into_multiple_children():
    docs = [{"id": "长文", "code": "TEXT", "text": "甲" * 500}]
    chunks = split_parent_child(docs, child_size=120, child_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)


def test_windows_covers_tail():
    text = "".join(str(i % 10) for i in range(300))
    wins = _windows(text, 100, 20)
    assert wins[0] == text[:100]
    assert wins[-1].endswith(text[-1])


def test_stats(sample_docs):
    stats = chunk_stats(split_parent_child(sample_docs, child_size=120, child_overlap=20))
    assert stats["n"] >= 3
    assert stats["n_parents"] == 3


def test_empty_documents_produce_no_chunks():
    assert split_parent_child([], child_size=100, child_overlap=10) == []
    assert split_parent_child([{"id": "空", "code": "T", "text": "   "}]) == []
