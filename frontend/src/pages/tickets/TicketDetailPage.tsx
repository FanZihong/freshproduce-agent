import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Tag } from '@/components/ui/Tag'
import { Empty } from '@/components/ui/Empty'
import { TICKET_PRIORITY, TICKET_STATUS } from '@/constants/status'
import { ROUTES } from '@/constants/routes'
import { fetchTicket, updateTicketStatus } from '@/api/tickets'
import type { Ticket } from '@/types'
import { formatDateTime } from '@/utils/date'
import { useUiStore } from '@/stores/uiStore'

export function TicketDetailPage() {
  const { ticketId } = useParams<{ ticketId: string }>()
  const navigate = useNavigate()
  const pushToast = useUiStore((state) => state.pushToast)
  const [ticket, setTicket] = useState<Ticket | undefined>()
  const [loading, setLoading] = useState(true)
  const [updating, setUpdating] = useState(false)

  useEffect(() => {
    if (!ticketId) return
    let mounted = true
    setLoading(true)
    void fetchTicket(ticketId)
      .then((data) => {
        if (!mounted) return
        setTicket(data)
      })
      .catch(() => {
        if (!mounted) return
        setTicket(undefined)
      })
      .finally(() => {
        if (!mounted) return
        setLoading(false)
      })
    return () => {
      mounted = false
    }
  }, [ticketId])

  if (loading) {
    return <div className="p-5 text-sm text-slate-400">加载中...</div>
  }

  if (!ticket) {
    return (
      <div className="p-5">
        <Empty title="工单不存在" description="请返回工单列表重新选择" />
      </div>
    )
  }

  const statusMeta = TICKET_STATUS[ticket.status]
  const priorityMeta = TICKET_PRIORITY[ticket.priority]

  const handleStatus = async (status: Ticket['status']) => {
    setUpdating(true)
    await updateTicketStatus(ticket.id, status)
    setTicket({ ...ticket, status })
    setUpdating(false)
    pushToast({ title: '工单状态已更新', variant: 'success' })
  }

  return (
    <div className="flex flex-col gap-4 p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="mono text-sm font-semibold text-slate-900">{ticket.id}</span>
          <Tag variant={statusMeta.variant} dot>
            {statusMeta.label}
          </Tag>
          <Tag variant={priorityMeta.variant}>{priorityMeta.label}</Tag>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={() => navigate(ROUTES.chatSession(ticket.sessionId))}>
            查看原始对话
          </Button>
          {ticket.status !== 'processing' ? (
            <Button variant="secondary" disabled={updating} onClick={() => handleStatus('processing')}>
              标记处理中
            </Button>
          ) : null}
          {ticket.status !== 'closed' ? (
            <Button variant="primary" disabled={updating} onClick={() => handleStatus('closed')}>
              关闭工单
            </Button>
          ) : null}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <Card title="工单信息" className="col-span-2">
          <dl className="grid grid-cols-2 gap-y-3 text-[13px]">
            <dt className="text-slate-500">关联订单</dt>
            <dd className="mono text-slate-800">{ticket.orderId}</dd>
            <dt className="text-slate-500">问题类型</dt>
            <dd className="text-slate-800">{ticket.reason}</dd>
            <dt className="text-slate-500">创建时间</dt>
            <dd className="text-slate-800">{formatDateTime(ticket.createdAt)}</dd>
            <dt className="text-slate-500">处理人</dt>
            <dd className="text-slate-800">{ticket.assignee}</dd>
          </dl>
        </Card>
        <Card title="关联会话">
          <p className="text-[13px] text-slate-600">会话编号</p>
          <p className="mono mt-1 text-[13px] text-slate-800">{ticket.sessionId}</p>
          <button
            type="button"
            onClick={() => navigate(ROUTES.chatSession(ticket.sessionId))}
            className="link focus-ring mt-3 cursor-pointer bg-transparent text-[13px]"
          >
            回放对话
          </button>
        </Card>
      </div>

      <Card title="上下文摘要">
        <p className="text-[13px] leading-relaxed text-slate-600">{ticket.summary}</p>
      </Card>
    </div>
  )
}
