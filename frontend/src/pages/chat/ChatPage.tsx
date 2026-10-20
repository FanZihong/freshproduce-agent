import { useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { ChatLayout } from '@/layouts/ChatLayout'
import { useSessionStore } from '@/stores/sessionStore'
import { useUiStore } from '@/stores/uiStore'
import { SessionList } from './SessionList'
import { MessageList } from './MessageList'
import { Composer } from './Composer'
import { ChatDetailDrawer } from './ChatDetailDrawer'

export function ChatPage() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const navigate = useNavigate()
  const loadSessions = useSessionStore((state) => state.loadSessions)
  const selectSession = useSessionStore((state) => state.selectSession)
  const currentSessionId = useSessionStore((state) => state.currentSessionId)
  const drawerOpen = useUiStore((state) => state.drawerOpen)

  useEffect(() => {
    void loadSessions()
  }, [loadSessions])

  useEffect(() => {
    if (sessionId) {
      void selectSession(sessionId)
    }
  }, [selectSession, sessionId])

  useEffect(() => {
    if (!sessionId && currentSessionId) {
      void navigate(`/chat/${currentSessionId}`, { replace: true })
    }
  }, [currentSessionId, navigate, sessionId])

  return (
    <ChatLayout
      sidebar={<SessionList />}
      drawer={<ChatDetailDrawer />}
      drawerOpen={drawerOpen}
    >
      <MessageList />
      <Composer />
    </ChatLayout>
  )
}
