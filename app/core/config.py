"""配置：全部走环境变量，代码里不留任何密钥。

.env 只在本地存在（.gitignore），仓库里只有 .env.example。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """优先用 python-dotenv；没装就手写一个极简解析，保证零依赖也能跑。"""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
        return
    except Exception:
        pass
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def _get(name: str, default: str) -> str:
    value = os.environ.get(name, "")
    return value.strip() if value.strip() else default


@dataclass
class Settings:
    # 数据与索引
    corpus_path: str = field(default_factory=lambda: _get("CORPUS_PATH", "data/law.txt"))
    index_dir: str = field(default_factory=lambda: _get("INDEX_DIR", "data/index"))
    # 切块
    child_size: int = field(default_factory=lambda: int(_get("CHILD_SIZE", "300")))
    child_overlap: int = field(default_factory=lambda: int(_get("CHILD_OVERLAP", "60")))
    # 向量
    embedding_provider: str = field(default_factory=lambda: _get("EMBEDDING_PROVIDER", "hash"))
    embedding_dim: int = field(default_factory=lambda: int(_get("EMBEDDING_DIM", "512")))
    embedding_model: str = field(default_factory=lambda: _get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"))
    embedding_base_url: str = field(
        default_factory=lambda: _get("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )
    embedding_path: str = field(default_factory=lambda: _get("EMBEDDING_PATH", ""))
    embedding_api_key: str = field(
        default_factory=lambda: os.environ.get("DASHSCOPE_API_KEY", "").strip()
        or os.environ.get("EMBEDDING_API_KEY", "").strip()
    )
    vector_backend: str = field(default_factory=lambda: _get("VECTOR_BACKEND", "numpy"))
    # 检索
    k_vec: int = field(default_factory=lambda: int(_get("K_VEC", "10")))
    k_bm25: int = field(default_factory=lambda: int(_get("K_BM25", "10")))
    top_n: int = field(default_factory=lambda: int(_get("TOP_N", "5")))
    reranker: str = field(default_factory=lambda: _get("RERANKER", "overlap"))
    reranker_model: str = field(default_factory=lambda: _get("RERANKER_MODEL", "BAAI/bge-reranker-base"))
    reranker_path: str = field(default_factory=lambda: _get("RERANKER_PATH", ""))
    rrf_k: int = field(default_factory=lambda: int(_get("RRF_K", "60")))
    strategy: str = field(default_factory=lambda: _get("STRATEGY", "hybrid_rerank"))
    weight_vector: float = field(default_factory=lambda: float(_get("WEIGHT_VECTOR", "0.5")))
    weight_bm25: float = field(default_factory=lambda: float(_get("WEIGHT_BM25", "0.5")))
    # LLM
    llm_provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "echo"))
    llm_model: str = field(default_factory=lambda: _get("LLM_MODEL", "qwen-plus"))
    llm_base_url: str = field(default_factory=lambda: _get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    llm_api_key: str = field(default_factory=lambda: os.environ.get("DASHSCOPE_API_KEY", "").strip()
                             or os.environ.get("LLM_API_KEY", "").strip())
    llm_max_tokens: int = field(default_factory=lambda: int(_get("LLM_MAX_TOKENS", "800")))
    # 服务
    api_key: str = field(default_factory=lambda: _get("API_KEY", ""))
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO"))

    @property
    def weights(self) -> tuple[float, float]:
        return (self.weight_vector, self.weight_bm25)

    def describe(self) -> dict:
        return {
            "corpus": self.corpus_path,
            "embedding": f"{self.embedding_provider}:{self.embedding_model if self.embedding_provider != 'hash' else self.embedding_dim}",
            "vector_backend": self.vector_backend,
            "reranker": self.reranker,
            "strategy": self.strategy,
            "llm": self.llm_provider,
            "has_api_key": bool(self.llm_api_key),
        }
