import ReactMarkdown from 'react-markdown'
import type { ChatMessage } from '@/types'
import { cn } from '@/utils/format'
import { formatTime } from '@/utils/date'
import { IntentBadge } from './IntentBadge'
import { ToolCallCard } from './ToolCallCard'
import { useUiStore } from '@/stores/uiStore'

interface MessageBubbleProps {
  message: ChatMessage
  onTraceClick?: (traceId: string) => void
}

export function MessageBubble({ message, onTraceClick }: MessageBubbleProps) {
  const openDrawer = useUiStore((state) => state.openDrawer)
  const isUser = message.role === 'user'

  return (
    <div className={cn('flex flex-col gap-1.5', isUser ? 'items-end' : 'items-start')}>
      <div className="flex items-center gap-2 text-xs text-slate-400">
        <span>{isUser ? '用户' : '客服 Agent'}</span>
        <span className="tabular-nums">{formatTime(message.createdAt)}</span>
      </div>

      <div
        className={cn(
          'max-w-[720px] rounded-lg px-3.5 py-2.5 text-sm leading-relaxed',
          isUser
            ? 'bg-brand-600 text-white'
            : 'border border-slate-200 bg-white text-slate-700',
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : message.content ? (
          <div className="prose-sm [&_p]:my-1">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        ) : (
          <span className="text-slate-400">正在生成</span>
        )}
      </div>

      {!isUser && message.toolCalls && message.toolCalls.length > 0 ? (
        <div className="flex w-full max-w-[720px] flex-col gap-1.5">
          {message.toolCalls.map((toolCall) => (
            <ToolCallCard key={toolCall.id} toolCall={toolCall} />
          ))}
        </div>
      ) : null}

      {!isUser && message.retrieval && message.retrieval.length > 0 ? (
        <div className="flex w-full max-w-[720px] flex-wrap gap-1.5">
          {message.retrieval.map((hit) => (
            <span
              key={hit.id}
              title={`${hit.source} · 相似度 ${hit.score.toFixed(2)}`}
              className="max-w-[260px] truncate rounded-md border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs text-slate-500"
            >
              {hit.title}
            </span>
          ))}
        </div>
      ) : null}

      {!isUser && (message.intent || message.traceId) ? (
        <div className="flex items-center gap-2">
          {message.intent ? (
            <IntentBadge intent={message.intent} onClick={() => openDrawer('state')} />
          ) : null}
          {message.traceId ? (
            <span
              onClick={() => onTraceClick?.(message.traceId as string)}
              className="link text-xs"
            >
              查看 Trace
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
