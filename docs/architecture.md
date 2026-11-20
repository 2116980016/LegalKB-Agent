# 架构

## 1. 数据流

```
                       ┌──────────────── 离线（一次性） ────────────────┐
 data/law.txt ──▶ loader（清洗元数据）──▶ splitter（父块=法条，子块≈300字）
                                              │
                                              ├─▶ HashEmbedding / bge ──▶ vectors.npy
                                              └─▶ tokenize ──▶ BM25 统计
                       └───────────────────────────────────────────────┘

                       ┌──────────────── 在线（每请求） ───────────────┐
 question ──▶ vector top-k ┐
              bm25  top-k   ├─▶ 加权 RRF 融合 ──▶ Rerank(top-n) ──▶ prompt
                            ┘                                        │
                                                                     ▼
                                                    LLM（流式 / 非流式）
                                                                     │
                       引用解析 [片段n] ◀──── 回答文本 ◀───────────────┘
                              │
                              └─▶ 校验编号是否越界 ──▶ citations[] ──▶ 前端高亮
                       └───────────────────────────────────────────────┘
```

## 2. 模块职责

| 层 | 文件 | 职责 | 不做什么 |
|---|---|---|---|
| 接入 | `app/main.py` | 路由、SSE、鉴权、错误码 | 不写业务逻辑 |
| 编排 | `rag/pipeline.py` | 检索→重排→生成→引用解析 | 不知道 HTTP |
| 服务 | `app/services/rag_service.py` | 生命周期、指标统计、索引重建 | 不关心参数来源 |
| 检索 | `rag/retrieve/*` | RRF 融合、重排策略 | 不碰 LLM |
| 索引 | `rag/index/*` | 向量、BM25、持久化、校验 | 不做切块 |
| 数据 | `rag/ingest/*` | 清洗、父子切块 | 不做检索 |
| 生成 | `rag/generate/*` | prompt、LLM 适配、引用解析 | 不做检索 |
| 评测 | `eval/*` | 金标集、指标、报告 | 不依赖线上服务 |

分层的判据只有一条：**能不能单独测**。`rag/` 不 import `app/`，所以评测可以直接调 `RAGPipeline`，
不需要起 HTTP 服务；这也是 `eval/run.py` 能在 CI 里 2 秒跑完的原因。

## 3. 一次请求发生了什么

1. `POST /api/chat` → `RAGService.answer`
2. `RAGPipeline.retrieve`：向量 top-10 与 BM25 top-10 各自排序 → RRF 融合 → 重排取 top-5
3. `to_prompt_block`：把 **父块全文**（不是子块）按 `[片段1]…` 编号拼进 prompt
4. `LLM.complete` / `LLM.stream`
5. `parse_citations`：正则抽 `[片段n]`，越界的标记 `valid=false`
6. 记录 latency / tokens / refused，返回

## 4. 部署

单容器即可跑（numpy 索引 + 内置语料）；生产化时把 `data/` 挂 volume，
`INDEX_DIR` 指向持久盘，多个副本共享同一份只读索引。

```
浏览器 ──▶ Nginx ──▶ uvicorn(1 worker) ──▶ 进程内索引(numpy) ──▶ 外部 LLM API
```
