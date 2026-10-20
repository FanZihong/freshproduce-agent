# 前端控制台

Vite + React + TypeScript + Tailwind CSS + React Router + zustand 实现的售后 Agent 控制台。

## 启动

```bash
npm install
npm run dev      # 开发环境，默认 Mock 数据
npm run build    # 类型检查 + 构建产物到 dist/
npm run preview  # 预览构建产物
```

## 环境变量（.env.development）

| 变量 | 说明 |
| --- | --- |
| VITE_API_BASE_URL | 后端接口前缀，默认 /api |
| VITE_WS_URL | WebSocket 前缀，默认 /ws |
| VITE_USE_MOCK | true 时使用本地 Mock 数据，无需后端 |

开发服务器已配置代理：`/api` 与 `/ws` 转发到 http://localhost:8000。

## 目录说明

```
src/
├─ api/           接口封装，USE_MOCK 为 true 时走 mock.ts
├─ components/
│  ├─ ui/         基础组件：Button、Input、Tag、Card、Table、Modal、Toast、Empty、Skeleton、Tooltip
│  └─ business/   业务组件：MessageBubble、ToolCallCard、IntentBadge、SlotPanel、TraceTree、MetricCard
├─ constants/     路由常量、状态映射
├─ hooks/         useChatStream（对话流式输出）、usePagination
├─ layouts/       AppLayout（侧边栏 + 顶栏）、ChatLayout（对话三栏）
├─ pages/         7 个功能页面
├─ router/        路由表
├─ stores/        zustand 状态（会话、Toast、详情抽屉）
├─ styles/        设计令牌与全局样式
├─ types/         类型定义
└─ utils/         格式化与日期工具
```

## 对接后端

接口按模块划分，切换真实后端时只需把 `VITE_USE_MOCK` 置为 `false`：

- 对话：GET /api/sessions、GET /api/sessions/{id}/messages、POST /api/chat/stream（SSE）
- 工单：GET /api/tickets、GET /api/tickets/{id}、POST /api/tickets、PATCH /api/tickets/{id}
- 评测：GET /api/evals/datasets、GET /api/evals/datasets/{id}/samples、POST /api/evals/runs、GET /api/evals/runs/{id}/report
- 监控：GET /api/traces、GET /api/traces/{id}
- 知识库：GET /api/knowledge/docs、POST /api/knowledge/docs
- 意图模型：GET /api/intent/versions、POST /api/intent/train
