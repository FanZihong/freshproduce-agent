import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Send, UserRound } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Modal } from '@/components/ui/Modal'
import { Tag } from '@/components/ui/Tag'
import { createTicket } from '@/api/tickets'
import { ROUTES } from '@/constants/routes'
import { useChatStream } from '@/hooks/useChatStream'
import { useSessionStore } from '@/stores/sessionStore'
import { useUiStore } from '@/stores/uiStore'

export function Composer() {
  const navigate = useNavigate()
  const { sending, send } = useChatStream()
  const currentSessionId = useSessionStore((state) => state.currentSessionId)
  const updateSession = useSessionStore((state) => state.updateSession)
  const pushToast = useUiStore((state) => state.pushToast)
  const messagesMap = useSessionStore((state) => state.messages)
  const [value, setValue] = useState('')
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [creating, setCreating] = useState(false)

  const disabled = !currentSessionId || sending
  const messages = useMemo(
    () => (currentSessionId ? messagesMap[currentSessionId] ?? [] : []),
    [currentSessionId, messagesMap],
  )

  const handoffContext = useMemo(() => {
    const slots = [...messages].reverse().flatMap((message) => message.slots ?? [])
    const orderId = slots.find((slot) => slot.key === 'order_id' && slot.value)?.value
    const lastUser = [...messages].reverse().find((message) => message.role === 'user')
    const lastAssistant = [...messages].reverse().find((message) => message.role === 'assistant')
    const disputeWords = ['全退', '投诉', '不行', '差评', '太差']
    const isDispute = Boolean(lastUser && disputeWords.some((word) => lastUser.content.includes(word)))
    const summary = [
      lastAssistant ? `Agent 方案：${lastAssistant.content.slice(0, 50)}` : '',
      lastUser ? `用户诉求：${lastUser.content.slice(0, 30)}` : '',
    ]
      .filter(Boolean)
      .join('；')
    return {
      orderId: orderId ?? '待补充',
      reason: isDispute ? '争议理赔转人工' : '用户主动转人工',
      priority: (isDispute ? 'high' : 'normal') as 'high' | 'normal',
      summary: summary || '会话中用户请求人工客服，已同步上下文与订单信息。',
    }
  }, [messages])

  const handleSend = async () => {
    if (!value.trim() || disabled) return
    const text = value
    setValue('')
    await send(text)
  }

  const handleHandoff = async () => {
    if (!currentSessionId) return
    setCreating(true)
    try {
      const ticket = await createTicket({
        sessionId: currentSessionId,
        orderId: handoffContext.orderId,
        reason: handoffContext.reason,
        summary: handoffContext.summary,
        priority: handoffContext.priority,
      })
      updateSession(currentSessionId, { status: 'handoff' })
      setConfirmOpen(false)
      pushToast({ title: '已创建工单', description: ticket.id, variant: 'success' })
      navigate(ROUTES.ticketDetail(ticket.id))
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="border-t border-slate-200 bg-white px-5 py-3">
      <textarea
        value={value}
        rows={3}
        disabled={disabled}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            void handleSend()
          }
        }}
        placeholder="描述您的售后问题，Enter 发送，Shift + Enter 换行"
        className="focus-ring w-full resize-none rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 transition-colors duration-150 hover:border-slate-300 focus:border-brand-600 disabled:cursor-not-allowed disabled:bg-slate-50"
      />

      <div className="mt-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {sending ? <Tag variant="warning" dot>生成中</Tag> : null}
          <span className="text-xs text-slate-400">
            意图识别、检索与工具调用过程可在右侧面板查看
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            icon={<UserRound size={15} />}
            disabled={disabled}
            onClick={() => setConfirmOpen(true)}
          >
            转人工
          </Button>
          <Button
            variant="primary"
            icon={<Send size={15} />}
            disabled={disabled || !value.trim()}
            onClick={handleSend}
          >
            发送
          </Button>
        </div>
      </div>

      <Modal
        open={confirmOpen}
        title="转人工确认"
        onClose={() => setConfirmOpen(false)}
        footer={
          <>
            <Button variant="secondary" onClick={() => setConfirmOpen(false)}>
              取消
            </Button>
            <Button variant="primary" disabled={creating} onClick={handleHandoff}>
              {creating ? '创建中' : '确认转人工'}
            </Button>
          </>
        }
      >
        <p className="text-sm text-slate-600">
          转人工后将基于当前会话创建工单，并把订单信息、已填槽位与工具调用结果一并交接给客服。
        </p>
      </Modal>
    </div>
  )
}
