# Agent 层（待实现）

LangGraph 工作流与 RAG、工具、记忆、评测、Trace 的实现，作为独立 Python 包供后端调用，不依赖 FastAPI。

## 计划结构

```
agent/
├─ agent/
│  ├─ graph/     state、nodes、router、workflow（含 checkpoint）
│  ├─ intent/    few-shot 分类器、本地微调模型推理、提示词
│  ├─ rag/       chunker、rewrite、hybrid_search、fusion(RRF)、reranker、retriever
│  ├─ tools/     order、logistics、policy、ticket、faq、mcp_server(FastMCP)
│  ├─ memory/    短期槽位、长期摘要
│  └─ prompts/   各节点提示词
├─ eval/         datasets、metrics、judge、run_eval
├─ tracing/      Langfuse 接入
├─ data/         sqlite（模拟订单/工单）、knowledge（政策/赔付标准/FAQ）
└─ scripts/      build_knowledge、label_intent、train_intent、evolve
```

## 关键约定

- 检索链路：多轮查询改写 -> BM25 + 向量 -> RRF 融合 -> Reranker -> 引用溯源
- 工作流：意图路由到查单/理赔/规则/转人工分支，条件边 + checkpoint，循环护栏
- 自进化：线上 trace -> badcase 归类 -> 扩充意图训练集 / 补齐知识库 / 更新 reranker -> 评测门禁 -> 灰度
- 评测：意图 F1、recall@k/MRR、工具成功率、槽位填充率、端到端达成率、转人工率
