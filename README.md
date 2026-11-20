# LegalKB-Agent

面向法律条文的 RAG 问答服务：**混合检索 + 重排 + 引用溯源 + 可复现评测**。
不依赖任何 LLM API 也能跑起来（内置离线桩），配置后可直接接 OpenAI 兼容模型。

> 语料：《民法典》1260 条（`data/law.txt`）。所有代码可离线运行，评测一条命令出报告。

---

## 它解决什么问题

法律问答的三个真实难点，不是"接个模型"能解决的：

1. **术语精确匹配**："违约金" vs "赔偿金" 语义近但适用不同条款 → 需要 BM25 兜底；
2. **上下文不能碎**：一个法条被切成两半，模型只看到"条件"或只看到"后果" → 父子切块；
3. **必须可溯源**：律师/法务不会相信没有出处的回答 → 强制 `[片段n]` 引用 + 编号合法性校验。

## 架构

```
law.txt ─▶ 清洗 ─▶ 父子切块 ─┬─▶ 向量索引（numpy / FAISS 可选）
                             └─▶ BM25 倒排
question ─▶ vector top-10 ┐
           bm25   top-10 ├─▶ 加权 RRF ─▶ 重排 top-5 ─▶ prompt ─▶ LLM
                         ┘                                      │
                       引用解析 [片段n] ◀── 回答 ────────────────┘
```

细节见 [`docs/architecture.md`](docs/architecture.md)，取舍见 [`docs/design-decisions.md`](docs/design-decisions.md)。

## 快速开始

```bash
pip install -r requirements.txt
python -m scripts.build_index          # 建索引（约 0.4s，1265 个子块）
python -m scripts.ask "违约金约定过高，法院可以调整吗"
```

起服务：

```bash
uvicorn app.main:app --reload --port 8000
# 或
docker compose up --build
```

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"结婚年龄是多少？"}'
```

流式（SSE）：`POST /api/chat/stream`，事件为 `delta` → `citations` → `done`。

## 评测

```bash
python -m eval.run --all        # 四种检索策略横向对比，输出 results/
```

**离线基线（hash-512-idf 向量 + echo 桩模型，16 题：12 有据 / 4 无据）**

| 策略 | Recall@3 | Recall@5 | MRR | 引用编号合法率 |
| :--- | ---: | ---: | ---: | ---: |
| vector（纯向量） | 66.7% | 75.0% | 0.595 | 100% |
| bm25（纯关键词） | 83.3% | 91.7% | 0.674 | 100% |
| hybrid（加权 RRF） | 75.0% | 91.7% | 0.653 | 100% |
| **hybrid_rerank** | 75.0% | 91.7% | **0.681** | 100% |

**怎么读这张表**（结论比数字重要）：

- 字面向量（哈希）与 BM25 同源，所以 **hybrid 的 Recall@3 反而低于纯 BM25**：两路没有互补信息，融合只会稀释 BM25 的强项。
- 但 `hybrid_rerank` 的 **MRR 最高（0.681）**：融合把候选面铺开、重排再把正确结果往前推——召回与排序是两个问题，指标要分开看。
- 换成语义向量（bge）后，向量支路才会提供互补信号；这一步没跑，所以**这里不写"混合一定更好"**。
- 生成侧（引用准确率、拒答率）在 echo 桩下无意义，报告里标 `N/A`，接真实 LLM 后再补。

指标定义与"为什么用锚点不用 chunk id"见 [`eval/metrics.py`](eval/metrics.py) 头部注释。

## 目录结构

```
app/            FastAPI 接入层 + 服务编排
├─ main.py         路由 / SSE / 鉴权
├─ core/           配置、结构化日志
└─ services/       索引生命周期、运行指标
rag/            核心 RAG（不依赖 app，可单独测）
├─ ingest/         清洗、父子切块
├─ index/          BM25、向量、持久化、embedding 适配
├─ retrieve/       RRF 融合、三种重排
├─ generate/       prompt、LLM 适配、引用解析
└─ pipeline.py     检索→重排→生成→引用
eval/           金标集 + 指标 + 一条命令出报告
tests/          25 个用例（切块/BM25/RRF/引用/端到端/API；API 用例需 httpx）
docs/           架构、ADR
```

## 已知局限

1. **只有《民法典》**：刑法、劳动法、公司法不在语料里，跨法域问题会误答（评测集里 4 道无据题就是为了量化这个问题）。
2. **拒答只能靠 prompt**：没有独立的"相关性判别"模型，域外问题（如"美国加州离婚"）因为词汇都在语料里，检索会命中，模型也就不会拒答。
3. **父子切块在本语料退化为 1:1**（单条平均 86.6 字），对长文档才有收益。
4. **单机索引**：numpy 精确检索，适合 <10 万块；再往上要换 FAISS/Qdrant。
5. **无会话记忆**：每次请求独立，多轮指代（"那它呢？"）不支持。

## Roadmap

- [ ] 接 bge-small-zh + bge-reranker，补真实模型下的评测表
- [ ] 语义拒答：用 LLM 判断"检索结果能否回答问题"，量化误拒率
- [ ] LangGraph 编排：路由（问答 / 计算 / 拒答）→ 检索 → 校验
- [ ] 会话持久化 + 前端引用高亮（Vue3）
- [ ] 金标集扩到 100 题，加 LLM-as-judge 与人工抽检一致性

## 开发

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```
