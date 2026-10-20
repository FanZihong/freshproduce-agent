import type { TraceSpan, TraceSummary } from '@/types'
import { http, USE_MOCK } from './client'
import { buildMockTraceDetail, delay, mockTraces } from './mock'

export async function fetchTraces(): Promise<TraceSummary[]> {
  if (USE_MOCK) {
    await delay(200)
    return mockTraces
  }
  const { data } = await http.get<TraceSummary[]>('/traces')
  return data
}

export async function fetchTraceDetail(traceId: string): Promise<TraceSpan[]> {
  if (USE_MOCK) {
    await delay(220)
    return buildMockTraceDetail(traceId)
  }
  const { data } = await http.get<TraceSpan[]>(`/traces/${traceId}`)
  return data
}
