"""BM25（Okapi）实现。

不依赖 jieba / rank_bm25：
- 中文按「单字 + 相邻二字」切分，避免引入分词器依赖，同时保留专有名词的精确匹配能力；
- 英文/数字按词切分并转小写。

面试可讲：为什么中文用 bigram 而不是单字（单字区分度低），
为什么不用 jieba（部署依赖 + 法律术语不在词典里，反而切坏）。
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence


def tokenize(text: str) -> list[str]:
    """返回 token 列表（中文含单字与二字组合，英文/数字按词）。"""
    tokens: list[str] = []
    cjk_buf: list[str] = []
    word_buf: list[str] = []

    def flush_cjk() -> None:
        if cjk_buf:
            tokens.extend(cjk_buf)
            tokens.extend(cjk_buf[i] + cjk_buf[i + 1] for i in range(len(cjk_buf) - 1))
            cjk_buf.clear()

    def flush_word() -> None:
        if word_buf:
            tokens.append("".join(word_buf))
            word_buf.clear()

    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            flush_word()
            cjk_buf.append(ch)
        elif ch.isalnum() or ch == "_":
            flush_cjk()
            word_buf.append(ch.lower())
        else:
            flush_cjk()
            flush_word()
    flush_cjk()
    flush_word()
    return tokens


class BM25:
    """标准 Okapi BM25。k1 控制词频饱和，b 控制文档长度归一化。"""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        if not corpus_tokens:
            raise ValueError("语料为空")
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[Counter] = [Counter(t) for t in corpus_tokens]
        self.doc_len: list[int] = [len(t) for t in corpus_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.n_docs = len(self.doc_tokens)

        df: Counter = Counter()
        for counter in self.doc_tokens:
            df.update(counter.keys())
        self.idf: dict[str, float] = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    @classmethod
    def from_texts(cls, texts: Iterable[str], **kwargs) -> BM25:
        return cls([tokenize(t) for t in texts], **kwargs)

    def score(self, query: str, doc_index: int) -> float:
        counter = self.doc_tokens[doc_index]
        dl = self.doc_len[doc_index]
        total = 0.0
        for term in tokenize(query):
            tf = counter.get(term, 0)
            if not tf:
                continue
            idf = self.idf.get(term, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += idf * tf * (self.k1 + 1) / denom
        return total

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
