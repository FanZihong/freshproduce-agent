export type IntentLabel =
  | 'query_order'
  | 'urge_logistics'
  | 'bad_fruit_claim'
  | 'short_weight'
  | 'refund_consult'
  | 'invoice'
  | 'quality_qa'
  | 'complaint'
  | 'chitchat'
  | 'human_handoff'

export interface IntentResult {
  label: IntentLabel
  name: string
  confidence: number
}

export type ToolStatus = 'running' | 'success' | 'failed'

export interface ToolCall {
  id: string
  name: string
  label: string
  status: ToolStatus
  args: Record<string, string | number>
  result?: string
  durationMs?: number
}

export interface SlotItem {
  key: string
  label: string
  value: string | null
  required: boolean
}

export interface RetrievalHit {
  id: string
  title: string
  source: string
  score: number
}

export type MessageRole = 'user' | 'assistant' | 'system'

export interface ChatMessage {
  id: string
  sessionId: string
  role: MessageRole
  content: string
  createdAt: string
  streaming?: boolean
  intent?: IntentResult
  slots?: SlotItem[]
  toolCalls?: ToolCall[]
  retrieval?: RetrievalHit[]
  traceId?: string
}

export type SessionStatus = 'active' | 'handoff' | 'closed'

export interface ChatSession {
  id: string
  title: string
  status: SessionStatus
  lastMessage: string
  updatedAt: string
}

export type TicketStatus = 'pending' | 'processing' | 'closed'
export type TicketPriority = 'high' | 'normal'

export interface Ticket {
  id: string
  sessionId: string
  orderId: string
  reason: string
  summary: string
  status: TicketStatus
  priority: TicketPriority
  createdAt: string
  assignee: string
}

export interface EvalDataset {
  id: string
  name: string
  description: string
  sampleCount: number
  updatedAt: string
}

export interface EvalSample {
  id: string
  datasetId: string
  input: string
  expectedIntent: string
  expectedAnswer: string
  note?: string
}

export interface EvalMetric {
  key: string
  label: string
  value: string
  target?: string
  delta?: number
}

export interface EvalRun {
  id: string
  datasetId: string
  datasetName: string
  version: string
  status: 'running' | 'finished' | 'failed'
  progress: number
  startedAt: string
  finishedAt?: string
}

export interface EvalComparison {
  label: string
  current: string
  previous: string
  delta: number
}

export interface EvalReport {
  runId: string
  version: string
  previousVersion: string
  metrics: EvalMetric[]
  comparisons: EvalComparison[]
}

export interface TraceSummary {
  id: string
  sessionId: string
  intent: string
  status: 'success' | 'failed'
  failureType?: string
  durationMs: number
  tokenCost: number
  createdAt: string
}

export type SpanType = 'workflow' | 'intent' | 'retrieval' | 'tool' | 'llm'

export interface TraceSpan {
  id: string
  name: string
  type: SpanType
  status: 'success' | 'failed'
  durationMs: number
  detail: Record<string, string>
  children?: TraceSpan[]
}

export interface KnowledgeDoc {
  id: string
  name: string
  category: string
  chunkCount: number
  status: 'indexed' | 'indexing' | 'failed'
  updatedAt: string
  docId?: string | null
  chunkCountParagraph?: number | null
  sections?: number | null
  source?: string | null
}

export interface IntentVersion {
  id: string
  version: string
  status: 'ready' | 'training' | 'failed'
  f1: number
  latencyMs: number
  costPerK: number
  note: string
  createdAt: string
  /** 以下字段来自 agent 侧真实产物，缺失时页面显示占位 */
  promptDetail?: string | null
  maxLen?: number | null
  baseModel?: string | null
  trainSamples?: number | null
  adapterDir?: string | null
  active?: boolean
  accuracy?: number | null
  errors?: number | null
  split?: string | null
  queries?: number | null
  calibration?: IntentCalibration | null
  ood?: IntentOodMetrics | null
}

/** 置信度校准与拒答阈值（scripts/fit_intent_calibration.py 产出） */
export interface IntentCalibration {
  temperature: number | null
  eceBefore: number | null
  eceAfter: number | null
  acceptThreshold: number | null
  handoffThreshold: number | null
  refusalRate: number | null
  refusedErrorRate: number | null
  acceptedErrorRate: number | null
  ratio: number | null
  fitSplit: string | null
  evalSplit: string | null
  samples: number | null
}

/** 域外检测指标（scripts/eval_ood.py 产出） */
export interface IntentOodMetrics {
  oodRecall: number | null
  falseKillRate: number | null
  modelOnlyRecall: number | null
  withEmbedder: boolean | null
  backend: string | null
}

export interface RewriteInfo {
  original: string
  query: string
  rewritten: boolean
  reason: string
  strategy: string
  model?: string
  latency_ms?: number
  error?: string
}

export interface RetrievalHitItem {
  rank: number
  chunkId: string
  docTitle: string
  section: string
  source: string
  text: string
  score: number
  fusedScore?: number
  rerankScore?: number
  ranks?: Record<string, number>
  movement?: number | null
}

export interface RetrievalCitation {
  chunk_id: string
  title: string
  section: string
  source: string
  snippet: string
  score: number
}

export interface RetrievalConfig {
  kind: string
  rerank: string
  strategy: string
  rrfK: number
  topK: number
  candidateK: number
  embedder: string
  reranker: string
}

export interface RetrievalSearchResult {
  query: string
  rewritten: RewriteInfo
  config: RetrievalConfig
  stages: {
    bm25: RetrievalHitItem[]
    vector: RetrievalHitItem[]
    fused: RetrievalHitItem[]
    reranked: RetrievalHitItem[]
  }
  citations: RetrievalCitation[]
  elapsedMs: number
}
