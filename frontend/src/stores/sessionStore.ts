import { create } from 'zustand'
import type { ChatMessage, ChatSession } from '@/types'
import { createSession as createSessionApi, fetchMessages, fetchSessions } from '@/api/chat'

interface SessionState {
  sessions: ChatSession[]
  messages: Record<string, ChatMessage[]>
  currentSessionId: string | null
  loadingSessions: boolean
  loadingMessages: boolean
  loadSessions: () => Promise<void>
  selectSession: (sessionId: string) => Promise<void>
  createSession: () => Promise<string>
  appendMessage: (message: ChatMessage) => void
  updateMessage: (sessionId: string, messageId: string, patch: Partial<ChatMessage>) => void
  updateSession: (sessionId: string, patch: Partial<ChatSession>) => void
}

export const useSessionStore = create<SessionState>((set, get) => ({
  sessions: [],
  messages: {},
  currentSessionId: null,
  loadingSessions: false,
  loadingMessages: false,

  loadSessions: async () => {
    set({ loadingSessions: true })
    const sessions = await fetchSessions()
    set({ sessions, loadingSessions: false })
    const current = get().currentSessionId
    if (!current && sessions.length > 0) {
      await get().selectSession(sessions[0].id)
    }
  },

  selectSession: async (sessionId) => {
    set({ currentSessionId: sessionId, loadingMessages: true })
    if (!get().messages[sessionId]) {
      const messages = await fetchMessages(sessionId)
      set((state) => ({ messages: { ...state.messages, [sessionId]: messages } }))
    }
    set({ loadingMessages: false })
  },

  createSession: async () => {
    const session = await createSessionApi()
    set((state) => ({
      sessions: [session, ...state.sessions],
      messages: { ...state.messages, [session.id]: [] },
      currentSessionId: session.id,
    }))
    return session.id
  },

  appendMessage: (message) =>
    set((state) => {
      const list = state.messages[message.sessionId] ?? []
      return {
        messages: { ...state.messages, [message.sessionId]: [...list, message] },
      }
    }),

  updateMessage: (sessionId, messageId, patch) =>
    set((state) => {
      const list = state.messages[sessionId] ?? []
      return {
        messages: {
          ...state.messages,
          [sessionId]: list.map((message) =>
            message.id === messageId ? { ...message, ...patch } : message,
          ),
        },
      }
    }),

  updateSession: (sessionId, patch) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.id === sessionId ? { ...session, ...patch } : session,
      ),
    })),
}))
