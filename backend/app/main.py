"""FastAPI 入口。

启动（在 backend 目录下）：
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import API_PREFIX, CORS_ORIGINS
from app.api.routes import (
    chat,
    evals,
    health,
    intent,
    knowledge,
    memory,
    retrieval,
    tickets,
    traces,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时确保数据库表结构与模拟订单就绪。"""
    import os

    from agent.db import init_db, seed_orders

    init_db()
    seed_orders()

    # 过期事实清理：服务启动跑一次，避免长期只靠"读时过滤"把死数据留在库里
    if os.environ.get("MEMORY_PURGE_ON_STARTUP", "true").lower() not in {"0", "false", "no"}:
        try:
            from agent.memory import facts as facts_store

            removed = facts_store.purge_expired_facts()
            if removed:
                print(f"[startup] 清理过期事实 {len(removed)} 条")
        except Exception as exc:  # noqa: BLE001 - 清理失败不影响服务启动
            print(f"[startup][warn] 过期事实清理失败（{type(exc).__name__}: {exc}）")

    # 预热检索层：向量模型首次加载要十几秒，放在启动阶段而不是第一个用户请求里
    if os.environ.get("BACKEND_WARMUP", "true").lower() not in {"0", "false", "no"}:
        try:
            from agent.rag.retriever import get_retriever

            retriever = get_retriever()
            print(f"[startup] 检索层就绪：{len(retriever.chunks)} 个 chunk，模型 {retriever.embedder.name}")
        except Exception as exc:  # noqa: BLE001 - 预热失败不影响服务启动
            print(f"[startup][warn] 检索层预热失败（{exc}），将在首次请求时懒加载")
    yield


app = FastAPI(title="生鲜售后 Agent API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for module in (health, chat, tickets, traces, retrieval, evals, knowledge, intent, memory):
    app.include_router(module.router, prefix=API_PREFIX)
