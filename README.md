# 生鲜售后智能客服 Agent

面向生鲜电商售后场景的智能客服 Agent 项目，覆盖意图识别、混合 RAG 检索、LangGraph 工作流、工具调用、记忆管理、全链路 Trace 与自动评测，并打通"badcase 挖掘 -> 知识/数据反哺 -> 评测门禁"的自进化闭环。

## 项目结构

```
fresh-service-agent/
├─ frontend/   前端控制台（Vite + React + TypeScript）
├─ backend/    后端服务（FastAPI）
├─ agent/      Agent 层（LangGraph 工作流、RAG、工具、评测）
└─ docs/       产品需求文档、前端设计规划、设计决策日志
```

## 当前进度

| 模块 | 状态 |
| --- | --- |
| frontend | 已完成页面骨架与交互（默认 Mock 数据模式，可直接运行） |
| backend | 待实现，接口约定见 backend/README.md |
| agent | 待实现，模块划分见 agent/README.md |

## 快速开始（前端）

```bash
cd frontend
npm install
npm run dev
```

默认 `VITE_USE_MOCK=true`，无需后端即可浏览全部页面；接入真实后端时把 `.env.development` 中的 `VITE_USE_MOCK` 改为 `false`，并把 `/api` 代理指向 FastAPI 服务。

## 页面一览

| 路由 | 页面 | 说明 |
| --- | --- | --- |
| /chat | 对话体验台 | 三栏布局：会话列表、对话区、状态详情抽屉 |
| /overview | 概览 | 核心指标卡、趋势图、失败类型分布 |
| /tickets | 工单中心 | 工单列表与详情，支持回放原始对话 |
| /evals | 评测中心 | 数据集、样本、评测任务与报告 |
| /traces | 监控中心 | Trace 列表、执行链路树、节点详情 |
| /knowledge | 知识库 | 文档上传、索引状态、分块预览 |
| /intent | 意图模型 | 版本指标对比、发起训练 |

## 设计约定

- 视觉：白底 + 灰阶 + 单一品牌色（#16A34A），无渐变、无插画、无 emoji
- 字体：系统字体栈（PingFang SC / Microsoft YaHei / Noto Sans SC），数字使用等宽数字
- 交互：可点击元素统一手型光标，禁用态禁止点击，加载用骨架屏，危险操作二次确认
- 详见 `docs/前端设计规划.md`
