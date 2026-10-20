import type { EvalDataset, EvalReport, EvalRun, EvalSample } from '@/types'
import { http, USE_MOCK } from './client'
import { delay, mockDatasets, mockReport, mockRuns, mockSamples } from './mock'

export async function fetchDatasets(): Promise<EvalDataset[]> {
  if (USE_MOCK) {
    await delay(200)
    return mockDatasets
  }
  const { data } = await http.get<EvalDataset[]>('/evals/datasets')
  return data
}

export async function fetchSamples(datasetId: string): Promise<EvalSample[]> {
  if (USE_MOCK) {
    await delay(180)
    return mockSamples.filter((sample) => sample.datasetId === datasetId)
  }
  const { data } = await http.get<EvalSample[]>(`/evals/datasets/${datasetId}/samples`)
  return data
}

export async function fetchRuns(): Promise<EvalRun[]> {
  if (USE_MOCK) {
    await delay(180)
    return mockRuns
  }
  const { data } = await http.get<EvalRun[]>('/evals/runs')
  return data
}

export async function fetchRun(runId: string): Promise<EvalRun | undefined> {
  if (USE_MOCK) {
    await delay(160)
    return mockRuns.find((run) => run.id === runId)
  }
  const { data } = await http.get<EvalRun>(`/evals/runs/${runId}`)
  return data
}

export async function fetchReport(runId: string): Promise<EvalReport> {
  if (USE_MOCK) {
    await delay(220)
    return { ...mockReport, runId }
  }
  const { data } = await http.get<EvalReport>(`/evals/runs/${runId}/report`)
  return data
}

export async function startRun(datasetId: string): Promise<EvalRun> {
  if (USE_MOCK) {
    await delay(300)
    const runId = `run-${Date.now().toString().slice(-8)}`
    return {
      id: runId,
      datasetId,
      datasetName: '生鲜售后黄金集 v1',
      version: 'intent-v3 + rag-rrf-rerank',
      status: 'running',
      progress: 0,
      startedAt: new Date().toISOString(),
    }
  }
  const { data } = await http.post<EvalRun>('/evals/runs', { datasetId })
  return data
}
