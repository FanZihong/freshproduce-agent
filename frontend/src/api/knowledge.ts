import type { KnowledgeDoc } from '@/types'
import { http, USE_MOCK } from './client'
import { delay, mockDocs } from './mock'

export async function fetchDocs(): Promise<KnowledgeDoc[]> {
  if (USE_MOCK) {
    await delay(200)
    return mockDocs
  }
  const { data } = await http.get<KnowledgeDoc[]>('/knowledge/docs')
  return data
}

export async function uploadDoc(fileName: string, category: string): Promise<KnowledgeDoc> {
  if (USE_MOCK) {
    await delay(420)
    return {
      id: `doc-${Date.now().toString().slice(-6)}`,
      name: fileName,
      category,
      chunkCount: 0,
      status: 'indexing',
      updatedAt: new Date().toISOString(),
    }
  }
  const { data } = await http.post<KnowledgeDoc>('/knowledge/docs', { fileName, category })
  return data
}
