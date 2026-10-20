import { useEffect, useState } from 'react'
import { Cpu } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Input } from '@/components/ui/Input'
import { Modal } from '@/components/ui/Modal'
import { Table } from '@/components/ui/Table'
import type { Column } from '@/components/ui/Table'
import { Tag } from '@/components/ui/Tag'
import { VERSION_STATUS } from '@/constants/status'
import { fetchVersions, startTraining } from '@/api/intent'
import type { IntentVersion } from '@/types'
import { formatDateTime } from '@/utils/date'
import { cn } from '@/utils/format'
import { useUiStore } from '@/stores/uiStore'

export function IntentPage() {
  const fmt = (value: number | null | undefined, digits: number) =>
    value === null || value === undefined ? '-' : value.toFixed(digits)
  const pct = (value: number | null | undefined) =>
    value === null || value === undefined ? '-' : `${(value * 100).toFixed(1)}%`
  const pushToast = useUiStore((state) => state.pushToast)
  const [versions, setVersions] = useState<IntentVersion[]>([])
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<string[]>([])
  const [trainOpen, setTrainOpen] = useState(false)
  const [note, setNote] = useState('')
  const [training, setTraining] = useState(false)
  const [command, setCommand] = useState('')

  useEffect(() => {
    let mounted = true
    void fetchVersions().then((data) => {
      if (!mounted) return
      setVersions(data)
      setLoading(false)
    })
    return () => {
      mounted = false
    }
  }, [])

  const toggleSelect = (id: string) => {
    setSelected((previous) => {
      if (previous.includes(id)) return previous.filter((item) => item !== id)
      if (previous.length >= 2) return [previous[1], id]
      return [...previous, id]
    })
  }

  const handleTrain = async () => {
    setTraining(true)
    try {
      const version = await startTraining(note.trim() || '手动触发训练')
      setCommand(version.note)
      pushToast({ title: '训练命令已生成', description: '复制到终端执行', variant: 'info' })
    } finally {
      setTraining(false)
    }
  }

  const copyCommand = async () => {
    try {
      await navigator.clipboard.writeText(command)
      pushToast({ title: '已复制', variant: 'success' })
    } catch {
      pushToast({ title: '复制失败，请手动选中复制', variant: 'warning' })
    }
  }

  const compareItems = versions.filter((version) => selected.includes(version.id))
  // 校准与域外指标挂在对应的 LoRA 版本上，取第一个有数据的版本展示
  const calibration = versions.find((version) => version.calibration)?.calibration ?? null
  const ood = versions.find((version) => version.ood)?.ood ?? null

  const columns: Column<IntentVersion>[] = [
    {
      key: 'select',
      title: '',
      width: '40px',
      render: (row) => (
        <input
          type="checkbox"
          checked={selected.includes(row.id)}
          onChange={() => toggleSelect(row.id)}
          className="h-3.5 w-3.5 cursor-pointer accent-[#16a34a]"
          aria-label={`选择 ${row.version}`}
        />
      ),
    },
    { key: 'version', title: '版本', render: (row) => <span className="mono text-[13px]">{row.version}</span> },
    {
      key: 'status',
      title: '状态',
      render: (row) => {
        const meta = VERSION_STATUS[row.status]
        return (
          <Tag variant={meta.variant} dot>
            {meta.label}
          </Tag>
        )
      },
    },
    {
      key: 'f1',
      title: 'macro-F1',
      align: 'right',
      render: (row) => (row.f1 > 0 ? row.f1.toFixed(3) : '-'),
    },
    {
      key: 'latencyMs',
      title: 'P95 时延',
      align: 'right',
      render: (row) => (row.latencyMs > 0 ? `${row.latencyMs} ms` : '-'),
    },
    {
      key: 'costPerK',
      title: '千次成本',
      align: 'right',
      render: (row) => (row.costPerK > 0 ? `¥${row.costPerK.toFixed(2)}` : '-'),
    },
    {
      key: 'note',
      title: '说明',
      render: (row) => (
        <div className="flex flex-col">
          <span>{row.note}</span>
          {row.active ? (
            <span className="mt-0.5 text-[11px] text-brand-600">当前推理版本</span>
          ) : null}
        </div>
      ),
    },
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
        <p className="text-xs text-slate-400">
          勾选两个版本可进行指标对比；训练数据来自大模型预标注与人工校准的蒸馏流水线
        </p>
        <Button variant="primary" icon={<Cpu size={15} />} onClick={() => setTrainOpen(true)}>
          发起训练
        </Button>
      </div>

      <Table
        columns={columns}
        rows={versions}
        rowKey={(row) => row.id}
        loading={loading}
        emptyText="暂无模型版本"
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card title="置信度校准与拒答策略">
          {calibration ? (
            <div className="flex flex-col gap-3">
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: '温度 T', value: fmt(calibration.temperature, 3) },
                  {
                    label: 'ECE（校准前 → 后）',
                    value: `${fmt(calibration.eceBefore, 3)} → ${fmt(calibration.eceAfter, 3)}`,
                  },
                  { label: '放行阈值', value: fmt(calibration.acceptThreshold, 2) },
                  { label: '转人工阈值', value: fmt(calibration.handoffThreshold, 2) },
                  {
                    label: '拒答率',
                    value: pct(calibration.refusalRate),
                  },
                  {
                    label: '被拒/被接受错误率倍数',
                    value: calibration.ratio === null ? '-' : `${calibration.ratio.toFixed(1)}x`,
                  },
                ].map((item) => (
                  <div key={item.label} className="rounded-md border border-slate-100 px-3 py-2">
                    <p className="text-xs text-slate-500">{item.label}</p>
                    <p className="mono mt-0.5 text-[13px] text-slate-700">{item.value}</p>
                  </div>
                ))}
              </div>
              <p className="text-xs text-slate-400">
                拟合集 {calibration.fitSplit ?? '-'} / 验收集 {calibration.evalSplit ?? '-'}，样本{' '}
                {calibration.samples ?? '-'} 条；被拒样本的错误率应显著高于被接受样本，
                否则阈值没有意义。
              </p>
            </div>
          ) : (
            <p className="text-xs text-slate-400">
              尚未校准。跑完本地推理与拟合后会显示温度、阈值与 ECE：
              <span className="mono ml-1">scripts/fit_intent_calibration.py</span>
            </p>
          )}
        </Card>

        <Card title="域外检测">
          {ood ? (
            <div className="flex flex-col gap-3">
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: '域外召回（三路信号）', value: pct(ood.oodRecall) },
                  { label: '已知类误杀率', value: pct(ood.falseKillRate) },
                  { label: '纯模型域外召回', value: pct(ood.modelOnlyRecall) },
                  { label: '语义相似度信号', value: ood.withEmbedder ? '已启用' : '未启用' },
                ].map((item) => (
                  <div key={item.label} className="rounded-md border border-slate-100 px-3 py-2">
                    <p className="text-xs text-slate-500">{item.label}</p>
                    <p className="mono mt-0.5 text-[13px] text-slate-700">{item.value}</p>
                  </div>
                ))}
              </div>
              <p className="text-xs text-slate-400">
                判定口径：top1 置信度低、与各类语义相似度低、规则版兜底 三路信号，
                命中领域词时强制判域内（压误杀）。评测后端 {ood.backend ?? '-'}。
              </p>
            </div>
          ) : (
            <p className="text-xs text-slate-400">
              尚未评测。跑 <span className="mono">scripts/eval_ood.py</span> 后会显示域外召回与误杀率。
            </p>
          )}
        </Card>
      </div>

      {compareItems.length === 2 ? (
        <Card title="版本对比">
          <div className="grid grid-cols-3 gap-4">
            {[
              { label: 'macro-F1', format: (value: number) => (value > 0 ? value.toFixed(3) : '-') },
              { label: 'P95 时延', format: (value: number) => (value > 0 ? `${value} ms` : '-') },
              { label: '千次成本', format: (value: number) => (value > 0 ? `¥${value.toFixed(2)}` : '-') },
            ].map((metric) => {
              const left = compareItems[0]
              const right = compareItems[1]
              const leftValue =
                metric.label === 'macro-F1'
                  ? left.f1
                  : metric.label === 'P95 时延'
                    ? left.latencyMs
                    : left.costPerK
              const rightValue =
                metric.label === 'macro-F1'
                  ? right.f1
                  : metric.label === 'P95 时延'
                    ? right.latencyMs
                    : right.costPerK
              const betterLeft =
                metric.label === 'macro-F1' ? leftValue >= rightValue : leftValue <= rightValue
              return (
                <div key={metric.label} className="rounded-md border border-slate-100 px-3 py-3">
                  <p className="text-xs text-slate-500">{metric.label}</p>
                  <div className="mt-1 flex items-center justify-between text-[13px]">
                    <span className={cn('mono', betterLeft ? 'text-brand-700' : 'text-slate-500')}>
                      {left.version} · {metric.format(leftValue)}
                    </span>
                    <span className={cn('mono', !betterLeft ? 'text-brand-700' : 'text-slate-500')}>
                      {right.version} · {metric.format(rightValue)}
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        </Card>
      ) : null}

      <Modal
        open={trainOpen}
        title="发起意图模型训练"
        onClose={() => {
          setTrainOpen(false)
          setCommand('')
        }}
        footer={
          <>
            <Button
              variant="secondary"
              onClick={() => {
                setTrainOpen(false)
                setCommand('')
              }}
            >
              关闭
            </Button>
            {command ? (
              <Button variant="secondary" onClick={copyCommand}>
                复制命令
              </Button>
            ) : (
              <Button variant="primary" disabled={training} onClick={handleTrain}>
                {training ? '生成中' : '生成训练命令'}
              </Button>
            )}
          </>
        }
      >
        <div className="flex flex-col gap-4">
          <Input
            label="训练说明"
            placeholder="例如：补全提示词后的第二轮实验"
            value={note}
            onChange={(event) => setNote(event.target.value)}
          />
          {command ? (
            <div className="flex flex-col gap-2">
              <p className="text-xs text-slate-500">在 WSL 终端里执行（训练要跑数小时，先挂后台）：</p>
              <pre className="overflow-x-auto rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-[12px] leading-5 text-slate-700">
                {command}
              </pre>
              <p className="text-xs text-slate-400">
                训练完成后刷新本页即出现新版本；需通过评测门禁（关键指标不劣化）才会替换线上基线。
              </p>
            </div>
          ) : (
            <p className="text-xs text-slate-400">
              训练是数小时的任务，页面只生成命令、不在请求里执行；跑完刷新页面即可看到新版本与真实指标。
            </p>
          )}
        </div>
      </Modal>
    </div>
  )
}
