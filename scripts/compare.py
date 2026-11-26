"""对比多个 results 目录，打印一张合并表。

用法：
    python -m scripts.compare results/size80 results/size150 results/size300
    python -m scripts.compare results/*          # 一次比全部

读的是每个目录下的 `<策略>.json`（评测脚本产出），不需要重新跑。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STRATEGIES = ["vector", "bm25", "hybrid", "hybrid_rerank"]


def _natural_key(directory: Path) -> list:
    """按数字自然排序：size80 < size150 < size300（而不是字典序）。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", directory.name)]


def load_dirs(args: list) -> list:
    dirs = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            dirs.append(p)
    if not dirs:
        dirs = [d for d in (ROOT / "results").iterdir() if d.is_dir()]
    return sorted(dirs, key=_natural_key)


def read_metrics(directory: Path) -> list:
    rows = []
    for strategy in STRATEGIES:
        f = directory / f"{strategy}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        cfg = data.get("config", {})
        rows.append(
            {
                "dir": directory.name,
                "strategy": strategy,
                "r3": data.get("recall3", 0.0),
                "r5": data.get("recall5", 0.0),
                "mrr": data.get("mrr", 0.0),
                "latency": data.get("avg_latency_ms", 0.0),
                "embedding": cfg.get("embedding", "-"),
                "llm": cfg.get("llm", "-"),
            }
        )
    return rows


def main() -> None:
    dirs = load_dirs(sys.argv[1:])
    rows = []
    for d in dirs:
        rows.extend(read_metrics(d))

    if not rows:
        print("没有找到结果，先跑：python -m eval.run --all --llm echo --out-dir results/xxx")
        return

    header = f"{'实验目录':<14}{'策略':<16}{'R@3':>7}{'R@5':>7}{'MRR':>8}{'延迟':>8}  {'embedding':<12}"
    print(header)
    print("-" * len(header))
    last = None
    for r in rows:
        mark = "" if last in (None, r["dir"]) else ""
        print(
            f"{r['dir']:<14}{r['strategy']:<16}{r['r3']:>6.1%}{r['r5']:>7.1%}"
            f"{r['mrr']:>8.3f}{r['latency']:>7.0f}ms  {r['embedding'][:12]:<12}{mark}"
        )
        last = r["dir"]

    # 按策略汇总最优 MRR，方便一眼看谁最好
    print("\n各策略的最佳 MRR：")
    for strategy in STRATEGIES:
        subset = [r for r in rows if r["strategy"] == strategy]
        if subset:
            best = max(subset, key=lambda x: x["mrr"])
            print(f"  {strategy:<16} {best['mrr']:.3f}  ← {best['dir']}")


if __name__ == "__main__":
    main()
