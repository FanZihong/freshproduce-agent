import { useCallback, useState } from 'react'
import type { ChatMessage, ToolCall } from '@/types'
import { API_BASE_URL, USE_MOCK } from '@/api/client'
import { buildMockReply, delay } from '@/api/mock'
import { useSessionStore } from '@/stores/sessionStore'
import { useUiStore } from '@/stores/uiStore'

function chunkText(text: string, size = 4): string[] {
  const chunks: string[] = []
  for (let index = 0; index < text.length; index += size) {
    chunks.push(text.slice(index, index + size))
  }
  return chunks
}

function createId(prefix: string): string {
  return `${prefix}-${Date.now().toString().slice(-8)}`
}

export function useChatStream() {
  const currentSessionId = useSessionStore((state) => state.currentSessionId)
  const appendMessage = useSessionStore((state) => state.appendMessage)
  const updateMessage = useSessionStore((state) => state.updateMessage)
  const updateSession = useSessionStore((state) => state.updateSession)
  const loadSessions = useSessionStore((state) => state.loadSessions)
  const pushToast = useUiStore((state) => state.pushToast)
  const [sending, setSending] = useState(false)

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || !currentSessionId || sending) return
      const sessionId = currentSessionId
      setSending(true)

      const now = new Date().toISOString()
      appendMessage({
        id: createId('m-user'),
        sessionId,
        role: 'user',
        content: trimmed,
        createdAt: now,
      })
      updateSession(sessionId, {
        lastMessage: trimmed,
        updatedAt: now,
        title: trimmed.slice(0, 16),
      })

      const assistantId = createId('m-assistant')
      appendMessage({
        id: assistantId,
        sessionId,
        role: 'assistant',
        content: '',
        createdAt: new Date().toISOString(),
        streaming: true,
      })

      try {
        if (USE_MOCK) {
          await delay(420)
          const reply = buildMockReply(trimmed)
          updateMessage(sessionId, assistantId, {
            intent: reply.intent,
            toolCalls: reply.toolCalls?.map((tool) => ({ ...tool, status: 'running' as const })),
            slots: reply.slots,
            retrieval: reply.retrieval,
          })
          if (reply.toolCalls?.length) {
            await delay(520)
            updateMessage(sessionId, assistantId, {
              toolCalls: reply.toolCalls.map((tool) => ({ ...tool, status: 'success' as const })),
            })
          }
          let buffer = ''
          for (const chunk of chunkText(reply.content)) {
            buffer += chunk
            updateMessage(sessionId, assistantId, { content: buffer })
            await delay(24)
          }
          updateMessage(sessionId, assistantId, {
            streaming: false,
            traceId: createId('tr'),
          })
          updateSession(sessionId, {
            lastMessage: reply.content,
            updatedAt: new Date().toISOString(),
          })
          return
        }

        const response = await fetch(`${API_BASE_URL}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sessionId, message: trimmed }),
        })
        if (!response.body) throw new Error('响应流不可用')
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let content = ''
        let toolCalls: ToolCall[] = []
        for (;;) {
          const { value, done } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''
          for (const line of lines) {
            if (!line.startsWith('data:')) continue
            const payload = line.slice(5).trim()
            if (!payload) continue
            const event = JSON.parse(payload) as {
              type: 'intent' | 'tool' | 'token' | 'done'
              intent?: ChatMessage['intent']
              toolCall?: ToolCall
              content?: string
              traceId?: string
            }
            if (event.type === 'intent' && event.intent) {
              updateMessage(sessionId, assistantId, { intent: event.intent })
            } else if (event.type === 'tool' && event.toolCall) {
              toolCalls = [...toolCalls, event.toolCall]
              updateMessage(sessionId, assistantId, { toolCalls })
            } else if (event.type === 'token' && event.content) {
              content += event.content
              updateMessage(sessionId, assistantId, { content })
            } else if (event.type === 'done') {
              updateMessage(sessionId, assistantId, {
                streaming: false,
                traceId: event.traceId,
              })
              // 刷新会话列表，让标题与最后一条消息跟后端保持一致
              void loadSessions()
            }
          }
        }
      } catch (error) {
        updateMessage(sessionId, assistantId, {
          streaming: false,
          content: '服务暂时不可用，请稍后重试或转人工客服。',
        })
        pushToast({
          title: '请求失败',
          description: error instanceof Error ? error.message : '未知错误',
          variant: 'danger',
        })
      } finally {
        setSending(false)
      }
    },
    [appendMessage, currentSessionId, loadSessions, pushToast, sending, updateMessage, updateSession],
  )

  return { sending, send }
}
