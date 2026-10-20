import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Play } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Modal } from '@/components/ui/Modal'
import { Table } from '@/components/ui/Table'
import type { Column } from '@/components/ui/Table'
import { Tag } from '@/components/ui/Tag'
import { ROUTES } from '@/constants/routes'
import { fetchDatasets, fetchRuns, fetchSamples, startRun } from '@/api/evals'
import type { EvalDataset, EvalRun, EvalSample } from '@/types'
import { formatDateTime } from '@/utils/date'
import { cn } from '@/utils/format'
import { useUiStore } from '@/stores/uiStore'

export function DatasetPage() {
  const navigate = useNavigate()
  const pushToast = useUiStore((state) => state.pushToast)
  const [datasets, setDatasets] = useState<EvalDataset[]>([])
  const [runs, setRuns] = useState<EvalRun[]>([])
  const [selectedDataset, setSelectedDataset] = useState<EvalDataset | null>(null)
  const [samples, setSamples] = useState<EvalSample[]>([])
  const [activeSample, setActiveSample] = useState<EvalSample | null>(null)
  const [starting, setStarting] = useState(false)

  useEffect(() => {
    let mounted = true
    void Promise.all([fetchDatasets(), fetchRuns()]).then(([datasetList, runList]) => {
      if (!mounted) return
      setDatasets(datasetList)
      setRuns(runList)
      if (datasetList.length > 0) {
        setSelectedDataset(datasetList[0])
      }
    })
    return () => {
      mounted = false
    }
  }, [])

  useEffect(() => {
    if (!selectedDataset) return
    void fetchSamples(selectedDataset.id).then(setSamples)
  }, [selectedDataset])

  const handleStartRun = async () => {
    if (!selectedDataset) return
    setStarting(true)
    try {
      const run = await startRun(selectedDataset.id)
      pushToast({ title: '评测任务已创建', description: run.id, variant: 'success' })
      navigate(ROUTES.evalRun(run.id))
    } finally {
      setStarting(false)
    }
  }

  const sampleColumns: Column<EvalSample>[] = [
    { key: 'input', title: '用户输入' },
    {
      key: 'expectedIntent',
      title: '期望意图',
      render: (row) => <span className="mono text-[13px] text-slate-600">{row.expectedIntent}</span>,
    },
    {
      key: 'note',
      title: '备注',
      render: (row) => row.note ?? '-',
    },
  ]

  const runColumns: Column<EvalRun>[] = [
    { key: 'id', title: '任务号', render: (row) => <span className="mono text-[13px]">{row.id}</span> },
    { key: 'datasetName', title: '数据集' },
    { key: 'version', title: '版本' },
    {
      key: 'status',
      title: '状态',
      render: (row) => (
        <Tag variant={row.status === 'finished' ? 'success' : row.status === 'running' ? 'warning' : 'danger'} dot>
          {row.status === 'finished' ? '已完成' : row.status === 'running' ? '运行中' : '失败'}
        </Tag>
      ),
    },
    {
      key: 'startedAt',
      title: '开始时间',
      align: 'right',
      render: (row) => formatDateTime(row.startedAt),
    },
  ]

  return (
    <div className="flex flex-col gap-4 p-5">
      <div className="grid grid-cols-4 gap-4">
        <Card title="评测数据集" className="col-span-1" bodyClassName="p-2">
          <ul className="flex flex-col gap-1">
            {datasets.map((dataset) => (
              <li key={dataset.id}>
                <button
                  type="button"
                  onClick={() => setSelectedDataset(dataset)}
                  className={cn(
                    'focus-ring w-full cursor-pointer rounded-md px-3 py-2.5 text-left transition-colors duration-150',
                    selectedDataset?.id === dataset.id ? 'bg-brand-50' : 'hover:bg-slate-50',
                  )}
                >
                  <p
                    className={cn(
                      'text-[13px] font-medium',
                      selectedDataset?.id === dataset.id ? 'text-brand-700' : 'text-slate-800',
                    )}
                  >
                    {dataset.name}
                  </p>
                  <p className="mt-1 text-xs text-slate-400">{dataset.sampleCount} 条样本</p>
                </button>
              </li>
            ))}
          </ul>
        </Card>

        <Card
          className="col-span-3"
          title={selectedDataset ? `${selectedDataset.name} · 样本列表` : '样本列表'}
          extra={
            <Button
              variant="primary"
              size="sm"
              icon={<Play size={14} />}
              disabled={!selectedDataset || starting}
              onClick={handleStartRun}
            >
              {starting ? '创建中' : '发起评测'}
            </Button>
          }
          bodyClassName="p-0"
        >
          <div className="px-4 py-3">
            <p className="mb-3 text-xs text-slate-400">
              {selectedDataset?.description ?? '选择左侧数据集查看样本'}
            </p>
            <Table
              columns={sampleColumns}
              rows={samples}
              rowKey={(row) => row.id}
              onRowClick={(row) => setActiveSample(row)}
              emptyText="该数据集暂无样本"
            />
          </div>
        </Card>
      </div>

      <Card title="评测任务记录" bodyClassName="p-0">
        <div className="px-4 py-3">
          <Table
            columns={runColumns}
            rows={runs}
            rowKey={(row) => row.id}
            onRowClick={(row) =>
              navigate(
                row.status === 'finished' ? ROUTES.evalReport(row.id) : ROUTES.evalRun(row.id),
              )
            }
          />
        </div>
      </Card>

      <Modal
        open={Boolean(activeSample)}
        title="样本详情"
        onClose={() => setActiveSample(null)}
      >
        {activeSample ? (
          <div className="flex flex-col gap-3 text-[13px]">
            <div>
              <p className="text-slate-500">用户输入</p>
              <p className="mt-1 text-slate-800">{activeSample.input}</p>
            </div>
            <div>
              <p className="text-slate-500">期望意图</p>
              <p className="mono mt-1 text-slate-800">{activeSample.expectedIntent}</p>
            </div>
            <div>
              <p className="text-slate-500">期望回答</p>
              <p className="mt-1 text-slate-800">{activeSample.expectedAnswer}</p>
            </div>
            {activeSample.note ? (
              <div>
                <p className="text-slate-500">备注</p>
                <p className="mt-1 text-slate-800">{activeSample.note}</p>
              </div>
            ) : null}
          </div>
        ) : null}
      </Modal>
    </div>
  )
}
