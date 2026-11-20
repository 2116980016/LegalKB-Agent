"""文档加载与清洗。

本仓库的语料是「一行一条法条 + 尾部元数据（校验人 / 链接 / 日期）」的格式：

    CIVIL_CODE_585_v1 CIVIL_CODE 第585条 当事人可以约定... 2021-01-01 verified ... https://...

直接把整行切块会把 URL 和校验人姓名一起塞进向量库，既污染检索又浪费 token。
所以这里先做结构化清洗，只保留「条号 + 正文」。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

_LINE = re.compile(r"^(?P<uid>\S+)\s+(?P<code>[A-Z_]+)\s+(?P<article>第\d+条)\s+(?P<rest>.+)$")
_DATE = re.compile(r"\s+\d{4}-\d{2}-\d{2}\s+")
_CODE_NAME = {"CIVIL_CODE": "民法典"}


def load_legal_corpus(path: str) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE.match(line)
        if not m:
            continue
        body = m.group("rest")
        cut = _DATE.search(body)
        if cut:
            body = body[: cut.start()]
        body = body.strip()
        if not body:
            continue
        code = m.group("code")
        name = _CODE_NAME.get(code, code)
        docs.append(
            {
                "id": m.group("article"),
                "code": code,
                "text": f"《{name}》{m.group('article')} {body}",
            }
        )
    return docs


def load_text_corpus(path: str) -> List[Dict[str, str]]:
    """普通 txt：按空行分段，每段一个父块。"""
    text = Path(path).read_text(encoding="utf-8")
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [{"id": f"段{i + 1}", "code": "TEXT", "text": p} for i, p in enumerate(parts)]


def load_documents(path: str) -> List[Dict[str, str]]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if p.suffix in {".txt", ".md"}:
        first = p.read_text(encoding="utf-8")[:500]
        if _LINE.search(first) or first.startswith("CIVIL_CODE"):
            legal = load_legal_corpus(path)
            if legal:
                return legal
        return load_text_corpus(path)
    raise ValueError(f"暂不支持的格式：{p.suffix}（支持 .txt / .md / .jsonl）")
