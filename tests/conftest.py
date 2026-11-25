import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.generate.llm import EchoLLM
from rag.index.embeddings import HashEmbedding
from rag.index.store import IndexStore
from rag.ingest.splitter import split_parent_child
from rag.pipeline import RAGPipeline

SAMPLE_DOCS = [
    {
        "id": "第585条",
        "code": "CIVIL_CODE",
        "text": "《民法典》第585条 当事人可以约定一方违约时应当根据违约情况向对方支付一定数额的违约金，"
        "也可以约定因违约产生的损失赔偿额的计算方法。约定的违约金低于造成的损失的，人民法院或者仲裁机构"
        "可以根据当事人的请求予以增加；约定的违约金过分高于造成的损失的，可以根据当事人的请求予以适当减少。",
    },
    {
        "id": "第1047条",
        "code": "CIVIL_CODE",
        "text": "《民法典》第1047条 结婚年龄，男不得早于二十二周岁，女不得早于二十周岁。",
    },
    {
        "id": "第188条",
        "code": "CIVIL_CODE",
        "text": "《民法典》第188条 向人民法院请求保护民事权利的诉讼时效期间为三年。法律另有规定的，依照其规定。",
    },
]


@pytest.fixture(scope="session")
def sample_docs():
    return SAMPLE_DOCS


@pytest.fixture(scope="session")
def chunks():
    return split_parent_child(SAMPLE_DOCS, child_size=120, child_overlap=20)


@pytest.fixture(scope="session")
def store(chunks):
    return IndexStore.build(chunks, HashEmbedding(dim=256), backend="numpy")


@pytest.fixture()
def pipeline(store):
    return RAGPipeline(store, llm=EchoLLM(), top_n=3, reranker="overlap")
