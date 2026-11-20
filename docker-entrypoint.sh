#!/bin/sh
set -e

# 没有索引就先建（首次启动约 1 秒；换 embedding 模型后需删掉 data/index 重建）
if [ ! -f "${INDEX_DIR:-data/index}/meta.json" ]; then
  echo "[entrypoint] 未发现索引，开始构建..."
  python -m scripts.build_index
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
