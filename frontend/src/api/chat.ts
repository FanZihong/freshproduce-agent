import type { ChatMessage, ChatSession } from '@/types'
import { http, USE_MOCK } from './client'
import { delay, mockMessages, mockSessions } from './mock'

export async function fetchSessions(): Promise<ChatSession[]> {
  if (USE_MOCK) {
    await delay(180)
    return mockSessions
  }
  const { data } = await http.get<ChatSession[]>('/sessions')
  return data
}

export async function fetchMessages(sessionId: string): Promise<ChatMessage[]> {
  if (USE_MOCK) {
    await delay(160)
    return mockMessages[sessionId] ?? []
  }
  const { data } = await http.get<ChatMessage[]>(`/sessions/${sessionId}/messages`)
  return data
}

export async function createSession(): Promise<ChatSession> {
  if (USE_MOCK) {
    await delay(120)
    return {
      id: `s-${Date.now().toString().slice(-6)}`,
      title: '新的售后咨询',
      status: 'active',
      lastMessage: '会话已创建，请输入您的问题',
      updatedAt: new Date().toISOString(),
    }
  }
  const { data } = await http.post<ChatSession>('/sessions')
  return data
}
