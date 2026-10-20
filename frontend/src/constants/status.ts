import type { IntentVersion, KnowledgeDoc, SessionStatus, TicketPriority, TicketStatus } from '@/types'

export type TagVariant = 'neutral' | 'success' | 'danger' | 'warning' | 'info'

interface StatusMeta {
  label: string
  variant: TagVariant
}

export const SESSION_STATUS: Record<SessionStatus, StatusMeta> = {
  active: { label: '进行中', variant: 'success' },
  handoff: { label: '已转人工', variant: 'warning' },
  closed: { label: '已结束', variant: 'neutral' },
}

export const TICKET_STATUS: Record<TicketStatus, StatusMeta> = {
  pending: { label: '待处理', variant: 'warning' },
  processing: { label: '处理中', variant: 'info' },
  closed: { label: '已关闭', variant: 'neutral' },
}

export const TICKET_PRIORITY: Record<TicketPriority, StatusMeta> = {
  high: { label: '高优先级', variant: 'danger' },
  normal: { label: '普通', variant: 'neutral' },
}

export const DOC_STATUS: Record<KnowledgeDoc['status'], StatusMeta> = {
  indexed: { label: '已索引', variant: 'success' },
  indexing: { label: '索引中', variant: 'warning' },
  failed: { label: '索引失败', variant: 'danger' },
}

export const VERSION_STATUS: Record<IntentVersion['status'], StatusMeta> = {
  ready: { label: '可用', variant: 'success' },
  training: { label: '训练中', variant: 'warning' },
  failed: { label: '训练失败', variant: 'danger' },
}

export const TRACE_STATUS: Record<'success' | 'failed', StatusMeta> = {
  success: { label: '成功', variant: 'success' },
  failed: { label: '失败', variant: 'danger' },
}

export const SPAN_TYPE_LABEL: Record<string, string> = {
  workflow: '工作流',
  intent: '意图识别',
  retrieval: '检索',
  tool: '工具调用',
  llm: '模型生成',
}
