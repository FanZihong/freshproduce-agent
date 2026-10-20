import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Tag } from '@/components/ui/Tag'
import { Empty } from '@/components/ui/Empty'
import { ROUTES } from '@/constants/routes'
import { fetchRun } from '@/api/evals'
import type { EvalRun } from '@/types'
import { formatDateTime } from '@/utils/date'

export function RunPage() {
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const [run, setRun] = useState<EvalRun | undefined>()
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!runId) return
    let mounted = true
    void fetchRun(runId).then((data) => {
      if (!mounted) return
      setRun(data)
      setLoading(false)
    })
    return () => {
      mounted = false
    }
  }, [runId])

  // 真实后端会自己推进进度，这里轮询任务状态而不是本地模拟
  useEffect(() => {
    if (!run || run.status !== 'running') return undefined
    const timer = window.setInterval(() => {
      void fetchRun(run.id)
        .then((latest) => {
          if (latest) setRun(latest)
        })
        .catch(() => undefined)
    }, 800)
    return () => window.clearInterval(timer)
  }, [run?.id, run?.status])

  if (loading) {
    return <div className="p-5 text-sm text-slate-400">加载中...</div>
  }

  if (!run) {
    return (
      <div className="p-5">
        <Empty title="评测任务不存在" />
      </div>
    )
  }

  const finished = run.status === 'finished'

  return (
    <div className="flex flex-col gap-4 p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="mono text-sm font-semibold text-slate-900">{run.id}</span>
          <Tag variant={finished ? 'success' : 'warning'} dot>
            {finished ? '已完成' : '运行中'}
          </Tag>
        </div>
        <Button
          variant={finished ? 'primary' : 'secondary'}
          disabled={!finished}
          onClick={() => navigate(ROUTES.evalReport(run.id))}
        >
          查看评测报告
        </Button>
      </div>

      <Card title="任务信息">
        <dl className="grid grid-cols-2 gap-y-3 text-[13px]">
          <dt className="text-slate-500">数据集</dt>
          <dd className="text-slate-800">{run.datasetName}</dd>
          <dt className="text-slate-500">评测版本</dt>
          <dd className="mono text-slate-800">{run.version}</dd>
          <dt className="text-slate-500">开始时间</dt>
          <dd className="text-slate-800">{formatDateTime(run.startedAt)}</dd>
          <dt className="text-slate-500">结束时间</dt>
          <dd className="text-slate-800">{run.finishedAt ? formatDateTime(run.finishedAt) : '-'}</dd>
        </dl>
      </Card>

      <Card title="执行进度">
        <div className="mb-2 flex items-center justify-between text-[13px] text-slate-600">
          <span>{finished ? '评测完成，报告已生成' : '正在执行意图识别、检索与端到端评测'}</span>
          <span className="tabular-nums">{run.progress}%</span>
        </div>
        <div className="h-2 w-full rounded-full bg-slate-100">
          <div
            className="h-2 rounded-full bg-brand-600 transition-all duration-300"
            style={{ width: `${run.progress}%` }}
          />
        </div>
      </Card>
    </div>
  )
}
