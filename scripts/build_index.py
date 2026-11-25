"""构建/重建索引：python -m scripts.build_index"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.core.logging import setup_logging
from app.services.rag_service import build_store


def main() -> None:
    settings = Settings()
    setup_logging(settings.log_level)
    store = build_store(settings)
    print(f"索引已构建：{len(store.chunks)} 个子块 / {store.meta.n_parents} 个父块")
    print(f"embedding={store.meta.embedder} backend={store.meta.backend} dir={settings.index_dir}")


if __name__ == "__main__":
    main()
