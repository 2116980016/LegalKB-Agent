"""向量化后端。

设计目标：**离线可跑**。
CI 和不带 Key 的机器上用 HashEmbedding（确定性哈希），有 GPU/网络时切到
sentence-transformers 的 bge 模型，接口完全一致。

这不是"为离线而离线"：面试里被问"你怎么做回归测试"，
答案就是"用确定性 embedding 跑端到端 smoke，保证链路正确性；
效果指标用真实模型另跑一份"。
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from rag.index.bm25 import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # (n, dim) float32
        ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (mat / norm).astype(np.float32)


class HashEmbedding:
    """把 token 哈希到固定维度并做 L2 归一化。

    可选 IDF 加权：不加权时等价于「带哈希技巧的 bag-of-bigrams」，区分度低；
    用语料拟合 IDF 后接近 tf-idf，对法律术语这种长尾词更敏感。

    注意：它没有语义能力（"违约金"和"罚款"在它眼里无关），
    只用于离线链路验证与 CI；真实效果请切到 bge。
    """

    def __init__(self, dim: int = 512, salt: str = "legal-kb"):
        self.dim = dim
        self.salt = salt
        self.name = f"hash-{dim}"
        self.idf: Dict[str, float] = {}

    def fit(self, texts: Sequence[str]) -> "HashEmbedding":
        df: Dict[str, int] = {}
        for text in texts:
            for tok in set(tokenize(text)):
                df[tok] = df.get(tok, 0) + 1
        n = max(len(texts), 1)
        self.idf = {tok: math.log((n + 1) / (cnt + 1)) + 1.0 for tok, cnt in df.items()}
        self.name = f"hash-{self.dim}-idf"
        return self

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = hashlib.md5((self.salt + tok).encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign * self.idf.get(tok, 1.0)
        return vec

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(np.vstack([self._vector(t) for t in texts]))


class SentenceTransformerEmbedding:
    """bge / m3e 等句向量模型（可选依赖）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer  # 延迟导入

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.name = model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(
            list(texts), batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


class OpenAICompatEmbedding:
    """走 OpenAI 兼容的 /embeddings 接口（DashScope、智谱、vLLM、OpenAI 均可）。

    好处：不需要装 torch（本地 bge 要 2GB+），部署时也不用把模型塞进镜像。
    代价：每次建索引都要调 API（1260 条 ≈ 130 次请求，几毛钱），且断网不可用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "text-embedding-v3",
        dim: int = 1024,
        batch_size: int = 10,
        timeout: int = 60,
    ):
        if not api_key:
            raise ValueError("缺少 embedding API Key：请在 .env 中配置 DASHSCOPE_API_KEY")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.timeout = timeout
        self.name = f"api:{model}:{dim}"

    def _post(self, texts: Sequence[str]) -> List[List[float]]:
        body = json.dumps(
            {"model": self.model, "input": list(texts), "encoding_format": "float",
             "dimensions": self.dim},
            ensure_ascii=False,
        ).encode("utf-8")
        last: Optional[Exception] = None
        for attempt in range(4):
            req = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                data = sorted(payload["data"], key=lambda d: d.get("index", 0))
                return [d["embedding"] for d in data]
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                raise
            except urllib.error.URLError as e:
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"embedding 请求失败：{last}")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._post(texts[i : i + self.batch_size]))
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))


def build_embedder(
    provider: str = "hash",
    dim: int = 512,
    model: str = "",
    api_key: str = "",
    base_url: str = "",
) -> Embedder:
    provider = (provider or "hash").lower()
    if provider == "hash":
        return HashEmbedding(dim=dim)
    if provider in {"bge", "sentence-transformers", "st"}:
        return SentenceTransformerEmbedding(model or "BAAI/bge-small-zh-v1.5")
    if provider in {"api", "openai", "dashscope", "zhipu"}:
        return OpenAICompatEmbedding(
            api_key=api_key,
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=model or "text-embedding-v3",
            dim=dim or 1024,
        )
    raise ValueError(f"未知的 embedding provider: {provider}")
