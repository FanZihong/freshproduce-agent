import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Table } from '@/components/ui/Table'
import type { Column } from '@/components/ui/Table'
import { Tag } from '@/components/ui/Tag'
import { Input } from '@/components/ui/Input'
import { TICKET_PRIORITY, TICKET_STATUS } from '@/constants/status'
import { ROUTES } from '@/constants/routes'
import { fetchTickets } from '@/api/tickets'
import type { Ticket, TicketStatus } from '@/types'
import { formatDateTime } from '@/utils/date'
import { cn } from '@/utils/format'

const FILTERS: Array<{ key: TicketStatus | 'all'; label: string }> = [
  { key: 'all', label: '全部' },
  { key: 'pending', label: '待处理' },
  { key: 'processing', label: '处理中' },
  { key: 'closed', label: '已关闭' },
]

export function TicketListPage() {
  const navigate = useNavigate()
  const [tickets, setTickets] = useState<Ticket[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<TicketStatus | 'all'>('all')
  const [keyword, setKeyword] = useState('')

  useEffect(() => {
    let mounted = true
    void fetchTickets().then((data) => {
      if (!mounted) return
      setTickets(data)
      setLoading(false)
    })
    return () => {
      mounted = false
    }
  }, [])

  const filtered = useMemo(() => {
    return tickets.filter((ticket) => {
      const matchStatus = filter === 'all' || ticket.status === filter
      const text = keyword.trim()
      const matchKeyword =
        !text || ticket.id.includes(text) || ticket.orderId.includes(text) || ticket.reason.includes(text)
      return matchStatus && matchKeyword
    })
  }, [filter, keyword, tickets])

  const columns: Column<Ticket>[] = [
    {
      key: 'id',
      title: '工单号',
      render: (row) => <span className="mono text-[13px] text-slate-800">{row.id}</span>,
    },
    {
      key: 'orderId',
      title: '关联订单',
      render: (row) => <span className="mono text-[13px] text-slate-500">{row.orderId}</span>,
    },
    { key: 'reason', title: '问题类型' },
    {
      key: 'priority',
      title: '优先级',
      render: (row) => {
        const meta = TICKET_PRIORITY[row.priority]
        return <Tag variant={meta.variant}>{meta.label}</Tag>
      },
    },
    {
      key: 'status',
      title: '状态',
      render: (row) => {
        const meta = TICKET_STATUS[row.status]
        return (
          <Tag variant={meta.variant} dot>
            {meta.label}
          </Tag>
        )
      },
    },
    { key: 'assignee', title: '处理人' },
    {
      key: 'createdAt',
      title: '创建时间',
      align: 'right',
      render: (row) => formatDateTime(row.createdAt),
    },
  ]

  return (
    <div className="flex flex-col gap-4 p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          {FILTERS.map((item) => (
            <button
              key={item.key}
              type="button"
              onClick={() => setFilter(item.key)}
              className={cn(
                'focus-ring cursor-pointer rounded-md px-3 py-1.5 text-[13px] transition-colors duration-150',
                filter === item.key
                  ? 'bg-brand-50 font-medium text-brand-700'
                  : 'text-slate-500 hover:bg-slate-100',
              )}
            >
              {item.label}
            </button>
          ))}
        </div>
        <div className="w-64">
          <Input
            placeholder="搜索工单号 / 订单号 / 问题类型"
            value={keyword}
            onChange={(event) => setKeyword(event.target.value)}
          />
        </div>
      </div>

      <Table
        columns={columns}
        rows={filtered}
        rowKey={(row) => row.id}
        loading={loading}
        onRowClick={(row) => navigate(ROUTES.ticketDetail(row.id))}
        emptyText="没有符合条件的工单"
      />
    </div>
  )
}
