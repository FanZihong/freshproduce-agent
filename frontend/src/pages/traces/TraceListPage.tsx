import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Table } from '@/components/ui/Table'
import type { Column } from '@/components/ui/Table'
import { Tag } from '@/components/ui/Tag'
import { MetricCard } from '@/components/business/MetricCard'
import { TRACE_STATUS } from '@/constants/status'
import { ROUTES } from '@/constants/routes'
import { fetchTraces } from '@/api/traces'
import type { TraceSummary } from '@/types'
import { formatDateTime } from '@/utils/date'
import { cn, formatCost, formatDuration, formatPercent } from '@/utils/format'

type Filter = 'all' | 'success' | 'failed'

const FILTERS: Array<{ key: Filter; label: string }> = [
  { key: 'all', label: '全部' },
  { key: 'success', label: '成功' },
  { key: 'failed', label: '失败' },
]

export function TraceListPage() {
  const navigate = useNavigate()
  const [traces, setTraces] = useState<TraceSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<Filter>('all')

  useEffect(() => {
    let mounted = true
    void fetchTraces().then((data) => {
      if (!mounted) return
      setTraces(data)
      setLoading(false)
    })
    return () => {
      mounted = false
    }
  }, [])

  const metrics = useMemo(() => {
    if (traces.length === 0) {
      return { successRate: 0, p95: 0, avgCost: 0, failureRate: 0 }
    }
    const success = traces.filter((trace) => trace.status === 'success').length
    const sorted = [...traces].sort((a, b) => a.durationMs - b.durationMs)
    const p95 = sorted[Math.floor(sorted.length * 0.95) - 1] ?? sorted[sorted.length - 1]
    const avgCost = traces.reduce((sum, trace) => sum + trace.tokenCost, 0) / traces.length
    const topFailure = traces.filter((trace) => trace.failureType).length / traces.length
    return { successRate: success / traces.length, p95: p95.durationMs, avgCost, failureRate: topFailure }
  }, [traces])

  const filtered = useMemo(
    () => traces.filter((trace) => filter === 'all' || trace.status === filter),
    [filter, traces],
  )

  const columns: Column<TraceSummary>[] = [
    { key: 'id', title: 'Trace', render: (row) => <span className="mono text-[13px]">{row.id}</span> },
    { key: 'sessionId', title: '会话', render: (row) => <span className="mono text-[13px] text-slate-500">{row.sessionId}</span> },
    { key: 'intent', title: '意图' },
    {
      key: 'status',
      title: '状态',
      render: (row) => {
        const meta = TRACE_STATUS[row.status]
        return (
          <Tag variant={meta.variant} dot>
            {meta.label}
          </Tag>
        )
      },
    },
    {
      key: 'failureType',
      title: '失败类型',
      render: (row) => row.failureType ?? '-',
    },
    {
      key: 'durationMs',
      title: '耗时',
      align: 'right',
      render: (row) => formatDuration(row.durationMs),
    },
    {
      key: 'tokenCost',
      title: '成本',
      align: 'right',
      render: (row) => formatCost(row.tokenCost),
    },
    {
      key: 'createdAt',
      title: '时间',
      align: 'right',
      render: (row) => formatDateTime(row.createdAt),
    },
  ]

  return (
    <div className="flex flex-col gap-4 p-5">
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="调用总数" value={String(traces.length)} hint="最近 24 小时" />
        <MetricCard label="成功率" value={formatPercent(metrics.successRate, 1)} target="≥ 95%" />
        <MetricCard label="P95 延迟" value={formatDuration(Math.round(metrics.p95))} target="≤ 2.5s" />
        <MetricCard label="平均单次成本" value={formatCost(metrics.avgCost)} target="≤ ¥0.005" />
      </div>

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
        <p className="text-xs text-slate-400">
          失败样本将进入 badcase 池，用于知识库补齐与意图训练集扩充
        </p>
      </div>

      <Table
        columns={columns}
        rows={filtered}
        rowKey={(row) => row.id}
        loading={loading}
        onRowClick={(row) => navigate(ROUTES.traceDetail(row.id))}
        emptyText="没有符合条件的 Trace"
      />
    </div>
  )
}
