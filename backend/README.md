# 后端服务

基于 FastAPI 提供会话、工单、评测、监控、知识库与意图模型的接口，并作为 Agent 层的调用入口。

## 启动

```bash
cd /mnt/d/生鲜售后agent/backend
source ../agent/.venv/bin/activate
pip install -r requirements.txt      # 或 pip install fastapi "uvicorn[standard]"
uvicorn app.main:app --reload --port 8000
```

前端把 `VITE_USE_MOCK` 改成 `false` 即可联调（Vite 已将 /api 代理到 8000）。

## 接口实现状态

| 接口 | 状态 |
| --- | --- |
| GET /api/health | 完整 |
| GET/POST /api/sessions、GET /api/sessions/{id}/messages | 完整（SQLite 持久化） |
| POST /api/chat/stream | 完整（SSE，直连 Agent 图） |
| GET/POST/PATCH /api/tickets | 完整（读写 agent 的 SQLite） |
| GET /api/traces、/api/traces/{id} | 完整（内存暂存，重启清空） |
| GET /api/retrieval/search | 完整（五级检索过程：改写 / BM25 / 向量 / RRF / 精排 + 引用） |
| GET /api/knowledge/docs | 完整（真实读取知识库目录并统计分块） |
| GET /api/memory | 完整（事实记忆只读视图：年龄 / 老化系数 / 过期时间） |
| evals / intent | 骨架实现（字段对齐前端，真实流水线后续接入） |

### 检索调试接口

```
GET /api/retrieval/search?query=车厘子怎么保存&top_k=3
GET /api/retrieval/search?query=那它呢&context=草莓怎么保存&product=草莓
GET /api/retrieval/search?query=坏果六成怎么算赔付&kind=hybrid&rerank=bge&strategy=rule&rrf_k=60
```

参数：`query`（必填）、`context`（上一轮问题）、`product`（上文商品）、`top_k`、`kind=bm25|vector|hybrid`、
`rerank=off|score|bge`、`strategy=rule|paragraph`、`rrf_k`。

返回五级结构：`rewritten`（改写信息）、`stages.bm25`、`stages.vector`、`stages.fused`（RRF 候选池）、
`stages.reranked`（最终结果）、`citations`（符合"有据"规则的引用）、`config`、`elapsedMs`。

服务启动时会预热检索层（加载向量模型），避免第一个请求等十几秒；可用环境变量 `BACKEND_WARMUP=false` 关闭。

## SSE 事件格式

每行一条 `data:` 帧，JSON 与前端 useChatStream 的约定一致：

```
data: {"type": "intent", "intent": {"label": "bad_fruit_claim", "name": "坏果理赔", "confidence": 0.78}}
data: {"type": "tool", "toolCall": {"id": "tc-...", "name": "query_order", "label": "查询订单", "status": "success", "args": {...}, "result": "...", "durationMs": 6}}
data: {"type": "token", "content": "请提供订单号"}
data: {"type": "done", "traceId": "tr-..."}
```

## 目录结构

```
backend/
├─ app/
│  ├─ main.py            入口、路由注册、CORS
│  ├─ config.py          配置加载
│  ├─ api/routes/        chat / tickets / evals / traces / knowledge / intent / health
│  ├─ api/schemas.py     请求与响应模型
│  └─ services/          chat_service（编排 + SSE）、session_service（落库）、trace_service、formatting
└─ （数据库复用 agent/agent/db.py，不单独建 db 层）
```

## 与前端约定的接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | /api/sessions | 会话列表 |
| GET | /api/sessions/{id}/messages | 会话消息 |
| POST | /api/chat/stream | 流式对话（SSE：intent / tool / token / done 事件） |
| GET | /api/tickets | 工单列表 |
| GET/POST/PATCH | /api/tickets[/{id}] | 工单详情、创建、状态更新 |
| GET | /api/evals/datasets | 评测数据集 |
| GET | /api/evals/datasets/{id}/samples | 数据集样本 |
| POST | /api/evals/runs | 发起评测 |
| GET | /api/evals/runs/{id}/report | 评测报告 |
| GET | /api/traces、/api/traces/{id} | Trace 列表与链路详情 |
| GET/POST | /api/knowledge/docs | 知识库文档 |
| GET | /api/memory?sessionId=... | 事实记忆（只读，含老化与过期状态） |
| GET | /api/intent/versions | 意图模型版本 |
| POST | /api/intent/train | 发起训练 |
| GET | /api/health | 健康检查 |

接口响应字段与 `frontend/src/types/index.ts` 保持一致。
