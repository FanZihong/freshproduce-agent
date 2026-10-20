import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Table } from '@/components/ui/Table'
import type { Column } from '@/components/ui/Table'
import { MetricCard } from '@/components/business/MetricCard'
import { Empty } from '@/components/ui/Empty'
import { ROUTES } from '@/constants/routes'
import { fetchReport } from '@/api/evals'
import type { EvalComparison, EvalReport } from '@/types'
import { formatDelta } from '@/utils/format'

export function ReportPage() {
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const [report, setReport] = useState<EvalReport | undefined>()
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!runId) return
    let mounted = true
    void fetchReport(runId).then((data) => {
      if (!mounted) return
      setReport(data)
      setLoading(false)
    })
    return () => {
      mounted = false
    }
  }, [runId])

  if (loading) {
    return <div className="p-5 text-sm text-slate-400">加载中...</div>
  }

  if (!report) {
    return (
      <div className="p-5">
        <Empty title="评测报告不存在" />
      </div>
    )
  }

  const columns: Column<EvalComparison>[] = [
    { key: 'label', title: '指标' },
    { key: 'current', title: '本次版本', align: 'right' },
    { key: 'previous', title: '上一版本', align: 'right' },
    {
      key: 'delta',
      title: '变化',
      align: 'right',
      render: (row) => (
        <span className={row.delta >= 0 ? 'text-brand-600' : 'text-red-600'}>
          {formatDelta(row.delta, 2)}
        </span>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-4 p-5">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold text-slate-900">评测报告 {report.runId}</p>
          <p className="mt-1 text-xs text-slate-400">
            当前版本 {report.version} · 对比版本 {report.previousVersion}
          </p>
        </div>
        <Button variant="secondary" onClick={() => navigate(ROUTES.evals)}>
          返回评测中心
        </Button>
      </div>

      <div className="grid grid-cols-4 gap-4">
        {report.metrics.map((metric) => (
          <MetricCard
            key={metric.key}
            label={metric.label}
            value={metric.value}
            target={metric.target}
            delta={metric.delta}
          />
        ))}
      </div>

      <Card title="与上一版本对比" bodyClassName="p-0">
        <div className="px-4 py-3">
          <Table columns={columns} rows={report.comparisons} rowKey={(row) => row.label} />
        </div>
      </Card>
    </div>
  )
}
