"""评测接口（骨架实现：数据结构对齐前端，真实评测流水线后续接入）。"""

from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.api.schemas import EvalDatasetOut, EvalRunOut, EvalRunRequest, EvalSampleOut

router = APIRouter(tags=["evals"])

DATASETS: list[dict] = [
    {
        "id": "ds-golden",
        "name": "生鲜售后黄金集 v1",
        "description": "覆盖 10 类意图的人工标注样本，作为回归基线",
        "sampleCount": 4,
        "updatedAt": "2026-10-08 18:30:00",
    }
]

SAMPLES: list[dict] = [
    {
        "id": "sp-001",
        "datasetId": "ds-golden",
        "input": "草莓到货坏了一半，怎么赔",
        "expectedIntent": "bad_fruit_claim",
        "expectedAnswer": "收集订单号与坏果比例，按赔付档位建工单",
        "note": "核心场景",
    },
    {
        "id": "sp-002",
        "datasetId": "ds-golden",
        "input": "我的订单到哪了",
        "expectedIntent": "query_order",
        "expectedAnswer": "查询订单与物流状态并回复",
    },
    {
        "id": "sp-003",
        "datasetId": "ds-golden",
        "input": "这个不行，我要全退",
        "expectedIntent": "human_handoff",
        "expectedAnswer": "识别争议并带上下文转人工",
        "note": "争议升级用例",
    },
    {
        "id": "sp-004",
        "datasetId": "ds-golden",
        "input": "今天天气怎么样",
        "expectedIntent": "fallback",
        "expectedAnswer": "域外问题礼貌拒答并说明能力边界",
    },
]

RUNS: dict[str, dict] = {}


async def _advance(run_id: str) -> None:
    while run_id in RUNS and RUNS[run_id]["progress"] < 100:
        await asyncio.sleep(0.5)
        run = RUNS.get(run_id)
        if run is None:
            return
        run["progress"] = min(100, run["progress"] + 20)
        if run["progress"] >= 100:
            run["status"] = "finished"
            run["finishedAt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@router.get("/evals/datasets", response_model=list[EvalDatasetOut])
def list_datasets() -> list[dict]:
    return DATASETS


@router.get("/evals/datasets/{dataset_id}/samples", response_model=list[EvalSampleOut])
def list_samples(dataset_id: str) -> list[dict]:
    return [sample for sample in SAMPLES if sample["datasetId"] == dataset_id]


@router.get("/evals/runs", response_model=list[EvalRunOut])
def list_runs() -> list[dict]:
    return sorted(RUNS.values(), key=lambda item: item["startedAt"], reverse=True)


@router.post("/evals/runs", response_model=EvalRunOut)
async def create_run(payload: EvalRunRequest) -> dict:
    dataset = next((item for item in DATASETS if item["id"] == payload.datasetId), None)
    if dataset is None:
        raise HTTPException(status_code=404, detail="数据集不存在")

    run_id = f"run-{uuid4().hex[:8]}"
    run = {
        "id": run_id,
        "datasetId": dataset["id"],
        "datasetName": dataset["name"],
        "version": "intent-v3 + rag-rrf-rerank",
        "status": "running",
        "progress": 0,
        "startedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "finishedAt": None,
    }
    RUNS[run_id] = run
    asyncio.create_task(_advance(run_id))
    return run


@router.get("/evals/runs/{run_id}", response_model=EvalRunOut)
def get_run(run_id: str) -> dict:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="评测任务不存在")
    return run


@router.get("/evals/runs/{run_id}/report")
def get_report(run_id: str) -> dict:
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="评测任务不存在")
    return {
        "runId": run_id,
        "version": "intent-v3 + rag-rrf-rerank",
        "previousVersion": "intent-v2 + rag-hybrid",
        "metrics": [
            {"key": "intent_f1", "label": "意图识别 macro-F1", "value": "0.912", "target": ">= 0.90", "delta": 0.038},
            {"key": "recall5", "label": "检索 recall@5", "value": "0.867", "target": ">= 0.85", "delta": 0.061},
            {"key": "resolution", "label": "端到端达成率", "value": "82.3%", "target": ">= 80%", "delta": 5.4},
            {"key": "handoff", "label": "转人工率", "value": "21.8%", "target": "<= 25%", "delta": -6.2},
        ],
        "comparisons": [
            {"label": "意图 macro-F1", "current": "0.912", "previous": "0.874", "delta": 0.038},
            {"label": "检索 recall@5", "current": "0.867", "previous": "0.806", "delta": 0.061},
            {"label": "端到端达成率", "current": "82.3%", "previous": "76.9%", "delta": 5.4},
            {"label": "转人工率", "current": "21.8%", "previous": "28.0%", "delta": -6.2},
        ],
    }
