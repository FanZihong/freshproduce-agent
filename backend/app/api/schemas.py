"""接口响应模型，字段与 frontend/src/types/index.ts 一一对应。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class IntentResultOut(BaseModel):
    label: str
    name: str
    confidence: float


class ToolCallOut(BaseModel):
    id: str
    name: str
    label: str
    status: str
    args: dict = Field(default_factory=dict)
    result: str | None = None
    durationMs: int | None = None


class SlotItemOut(BaseModel):
    key: str
    label: str
    value: str | None = None
    required: bool = True


class ChatMessageOut(BaseModel):
    id: str
    sessionId: str
    role: str
    content: str
    createdAt: str
    intent: IntentResultOut | None = None
    slots: list[SlotItemOut] | None = None
    toolCalls: list[ToolCallOut] | None = None
    retrieval: list[dict] | None = None
    traceId: str | None = None


class ChatSessionOut(BaseModel):
    id: str
    title: str
    status: str
    lastMessage: str
    updatedAt: str


class ChatStreamRequest(BaseModel):
    sessionId: str
    message: str


class TicketOut(BaseModel):
    id: str
    sessionId: str
    orderId: str
    reason: str
    summary: str
    status: str
    priority: str
    createdAt: str
    assignee: str


class TicketCreateRequest(BaseModel):
    sessionId: str
    orderId: str
    reason: str
    summary: str = ""
    priority: str = "normal"


class TicketUpdateRequest(BaseModel):
    status: str


class MemoryFactOut(BaseModel):
    """一条事实记忆；老化字段由 agent 侧现算，便于前端/运维直接看。"""

    id: str
    sessionId: str
    fact: str
    category: str
    entities: list[str] = Field(default_factory=list)
    importance: float
    source: str
    createdAt: str
    updatedAt: str
    lastAccessedAt: str | None = None
    ageDays: float
    decay: float
    expiresAt: str
    expired: bool


class MemoryListOut(BaseModel):
    sessionId: str | None = None
    ttlDays: float
    decayHalfLifeDays: float
    decayFloor: float
    total: int
    items: list[MemoryFactOut]


class TraceSummaryOut(BaseModel):
    id: str
    sessionId: str
    intent: str
    status: str
    failureType: str | None = None
    durationMs: int
    tokenCost: float
    createdAt: str


class EvalDatasetOut(BaseModel):
    id: str
    name: str
    description: str
    sampleCount: int
    updatedAt: str


class EvalSampleOut(BaseModel):
    id: str
    datasetId: str
    input: str
    expectedIntent: str
    expectedAnswer: str
    note: str | None = None


class EvalRunOut(BaseModel):
    id: str
    datasetId: str
    datasetName: str
    version: str
    status: str
    progress: int
    startedAt: str
    finishedAt: str | None = None


class EvalRunRequest(BaseModel):
    datasetId: str


class KnowledgeDocOut(BaseModel):
    id: str
    name: str
    category: str
    chunkCount: int
    status: str
    updatedAt: str
    docId: str | None = None
    chunkCountParagraph: int | None = None
    sections: int | None = None
    source: str | None = None


class KnowledgeDocRequest(BaseModel):
    fileName: str
    category: str = "售后政策"


class IntentVersionOut(BaseModel):
    id: str
    version: str
    status: str
    f1: float
    latencyMs: int
    costPerK: float
    note: str
    createdAt: str
    # 以下字段由 agent 侧的真实产物填充，缺失时前端显示占位
    promptDetail: str | None = None
    maxLen: int | None = None
    baseModel: str | None = None
    trainSamples: int | None = None
    adapterDir: str | None = None
    active: bool = False
    accuracy: float | None = None
    errors: int | None = None
    split: str | None = None
    queries: int | None = None
    perClass: dict | None = None
    calibration: dict | None = None
    ood: dict | None = None


class IntentTrainRequest(BaseModel):
    note: str = "手动触发训练"


class RetrievalHitOut(BaseModel):
    rank: int
    chunkId: str
    docTitle: str
    section: str
    source: str
    text: str
    score: float
    fusedScore: float | None = None
    rerankScore: float | None = None
    ranks: dict[str, int] = Field(default_factory=dict)
    movement: int | None = None


class RetrievalSearchOut(BaseModel):
    query: str
    rewritten: dict
    config: dict
    stages: dict[str, list[RetrievalHitOut]]
    citations: list[dict]
    elapsedMs: int
