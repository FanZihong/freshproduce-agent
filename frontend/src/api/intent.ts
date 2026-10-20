import type { IntentVersion } from '@/types'
import { http, USE_MOCK } from './client'
import { delay, mockVersions } from './mock'

export async function fetchVersions(): Promise<IntentVersion[]> {
  if (USE_MOCK) {
    await delay(200)
    return mockVersions
  }
  const { data } = await http.get<IntentVersion[]>('/intent/versions')
  return data
}

export async function startTraining(note: string): Promise<IntentVersion> {
  if (USE_MOCK) {
    await delay(360)
    return {
      id: `iv-${Date.now().toString().slice(-5)}`,
      version: `intent-v${Math.floor(Math.random() * 4 + 5)}-lora`,
      status: 'training',
      f1: 0,
      latencyMs: 0,
      costPerK: 0,
      note,
      createdAt: new Date().toISOString(),
    }
  }
  const { data } = await http.post<IntentVersion>('/intent/train', { note })
  return data
}
