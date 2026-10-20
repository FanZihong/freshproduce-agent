import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Empty } from '@/components/ui/Empty'
import { Tag } from '@/components/ui/Tag'
import { TraceTree } from '@/components/business/TraceTree'
import { SPAN_TYPE_LABEL, TRACE_STATUS } from '@/constants/status'
import { ROUTES } from '@/constants/routes'
import { fetchTraces, fetchTraceDetail } from '@/api/traces'
import type { TraceSpan, TraceSummary } from '@/types'
import { formatCost, formatDuration } from '@/utils/format'

export function TraceDetailPage() {
  const { traceId } = useParams<{ traceId: string }>()
  const navigate = useNavigate()
  const [spans, setSpans] = useState<TraceSpan[]>([])
  const [summary, setSummary] = useState<TraceSummary | undefined>()
  const [selected, setSelected] = useState<TraceSpan | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!traceId) return
    let mounted = true
    void Promise.all([fetchTraceDetail(traceId), fetchTraces()])
      .then(([spanList, traceList]) => {
        if (!mounted) return
        setSpans(spanList)
        setSelected(spanList[0] ?? null)
        setSummary(traceList.find((trace) => trace.id === traceId))
      })
      .catch(() => {
        if (!mounted) return
        setSpans([])
       })
      .finally(() => {
        if (!mounted) return
        setLoading(false)
      })
    return () => {
      mounted = false
    }
  }, [traceId])

  if (loading) {
    return <div className="p-5 text-sm text-slate-400">加载中...</div>
  }

  if (spans.length === 0) {
    return (
      <div className="p-5">
        <Empty title="Trace 不存在" description="请返回监控中心重新选择" />
      </div>
    )
  }

  const statusMeta = summary ? TRACE_STATUS[summary.status] : null

  return (
    <div className="flex flex-col gap-4 p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="mono text-sm font-semibold text-slate-900">{traceId}</span>
          {statusMeta ? (
            <Tag variant={statusMeta.variant} dot>
              {statusMeta.label}
            </Tag>
          ) : null}
          {summary?.failureType ? <Tag variant="danger">{summary.failureType}</Tag> : null}
        </div>
        <div className="flex items-center gap-2">
          {summary ? (
            <Button
              variant="secondary"
              onClick={() => navigate(ROUTES.chatSession(summary.sessionId))}
            >
              关联会话
            </Button>
          ) : null}
          <Button variant="secondary" onClick={() => navigate(ROUTES.traces)}>
            返回监控中心
          </Button>
        </div>
      </div>

      {summary ? (
        <div className="grid grid-cols-4 gap-4">
          <Card title="意图" bodyClassName="px-4 py-3">
            <p className="text-[13px] text-slate-800">{summary.intent}</p>
          </Card>
          <Card title="总耗时" bodyClassName="px-4 py-3">
            <p className="tabular-nums text-[13px] text-slate-800">
              {formatDuration(summary.durationMs)}
            </p>
          </Card>
          <Card title="Token 成本" bodyClassName="px-4 py-3">
            <p className="tabular-nums text-[13px] text-slate-800">
              {formatCost(summary.tokenCost)}
            </p>
          </Card>
          <Card title="关联会话" bodyClassName="px-4 py-3">
            <p className="mono text-[13px] text-slate-800">{summary.sessionId}</p>
          </Card>
        </div>
      ) : null}

      <div className="grid grid-cols-3 gap-4">
        <Card title="执行链路" className="col-span-2">
          <TraceTree spans={spans} selectedId={selected?.id} onSelect={setSelected} />
        </Card>
        <Card title="节点详情">
          {selected ? (
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-2">
                <span className="mono text-[13px] text-slate-800">{selected.name}</span>
                <Tag variant="neutral">{SPAN_TYPE_LABEL[selected.type] ?? selected.type}</Tag>
              </div>
              <div className="flex items-center gap-3 text-xs text-slate-500">
                <span>耗时 {formatDuration(selected.durationMs)}</span>
                <span>{selected.status === 'success' ? '成功' : '失败'}</span>
              </div>
              <dl className="flex flex-col gap-2">
                {Object.entries(selected.detail).map(([key, value]) => (
                  <div key={key} className="rounded-md border border-slate-100 px-3 py-2">
                    <dt className="mono text-xs text-slate-400">{key}</dt>
                    <dd className="mt-0.5 break-all text-[13px] text-slate-700">{value}</dd>
                  </div>
                ))}
              </dl>
            </div>
          ) : (
            <p className="text-xs text-slate-400">选择左侧节点查看详情</p>
          )}
        </Card>
      </div>
    </div>
  )
}
