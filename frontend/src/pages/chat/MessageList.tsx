import { useEffect, useMemo, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { MessageBubble } from '@/components/business/MessageBubble'
import { Empty } from '@/components/ui/Empty'
import { Skeleton } from '@/components/ui/Skeleton'
import { ROUTES } from '@/constants/routes'
import { useSessionStore } from '@/stores/sessionStore'

export function MessageList() {
  const navigate = useNavigate()
  const currentSessionId = useSessionStore((state) => state.currentSessionId)
  const loadingMessages = useSessionStore((state) => state.loadingMessages)
  const messagesMap = useSessionStore((state) => state.messages)
  const messages = useMemo(
    () => (currentSessionId ? messagesMap[currentSessionId] ?? [] : []),
    [currentSessionId, messagesMap],
  )
  const containerRef = useRef<HTMLDivElement>(null)
  const stickToBottom = useRef(true)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    if (stickToBottom.current) {
      container.scrollTop = container.scrollHeight
    }
  }, [messages])

  const handleScroll = () => {
    const container = containerRef.current
    if (!container) return
    stickToBottom.current =
      container.scrollHeight - container.scrollTop - container.clientHeight < 80
  }

  if (loadingMessages) {
    return (
      <div className="flex-1 p-6">
        <Skeleton lines={4} />
      </div>
    )
  }

  if (!currentSessionId) {
    return (
      <div className="flex flex-1 items-center justify-center p-6">
        <Empty title="请选择或新建一个会话" description="左侧会话列表可以选择历史会话" />
      </div>
    )
  }

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center p-6">
        <Empty
          title="开始新的售后咨询"
          description="可以试试：草莓坏了一半，怎么赔"
        />
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      className="flex flex-1 flex-col gap-4 overflow-auto px-5 py-5"
    >
      {messages.map((message) => (
        <MessageBubble
          key={message.id}
          message={message}
          onTraceClick={(traceId) => navigate(ROUTES.traceDetail(traceId))}
        />
      ))}
    </div>
  )
}
