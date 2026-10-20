import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import * as echarts from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { Card } from '@/components/ui/Card'
import { MetricCard } from '@/components/business/MetricCard'
import { ROUTES } from '@/constants/routes'
import { mockFailureDistribution, mockTrend } from '@/api/mock'
import { formatPercent } from '@/utils/format'

echarts.use([LineChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer])

function TrendChart() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return undefined
    const chart = echarts.init(containerRef.current)
    chart.setOption({
      grid: { left: 40, right: 16, top: 24, bottom: 28 },
      tooltip: { trigger: 'axis' },
      legend: {
        data: ['目标达成率', '检索 recall@5'],
        right: 0,
        top: 0,
        textStyle: { color: '#475569', fontSize: 12 },
      },
      xAxis: {
        type: 'category',
        data: mockTrend.map((item) => item.date),
        axisLine: { lineStyle: { color: '#e5e7eb' } },
        axisLabel: { color: '#94a3b8', fontSize: 11 },
      },
      yAxis: {
        type: 'value',
        min: 60,
        max: 100,
        splitLine: { lineStyle: { color: '#f1f5f9' } },
        axisLabel: { color: '#94a3b8', fontSize: 11, formatter: '{value}%' },
      },
      series: [
        {
          name: '目标达成率',
          type: 'line',
          smooth: true,
          symbolSize: 6,
          data: mockTrend.map((item) => item.resolution),
          itemStyle: { color: '#16a34a' },
          areaStyle: { color: 'rgba(22,163,74,0.08)' },
        },
        {
          name: '检索 recall@5',
          type: 'line',
          smooth: true,
          symbolSize: 6,
          data: mockTrend.map((item) => item.recall),
          itemStyle: { color: '#2563eb' },
        },
      ],
    })
    const handleResize = () => chart.resize()
    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      chart.dispose()
    }
  }, [])

  return <div ref={containerRef} className="h-64 w-full" />
}

export function OverviewPage() {
  const navigate = useNavigate()
  const maxFailure = Math.max(...mockFailureDistribution.map((item) => item.value))

  return (
    <div className="flex flex-col gap-4 p-5">
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          label="端到端目标达成率"
          value="82.3%"
          target="≥ 80%"
          delta={5.4}
          onClick={() => navigate(ROUTES.evals)}
        />
        <MetricCard
          label="转人工率"
          value="21.8%"
          target="≤ 25%"
          delta={-6.2}
          onClick={() => navigate(ROUTES.tickets)}
        />
        <MetricCard
          label="检索 recall@5"
          value="0.867"
          target="≥ 0.85"
          delta={6.1}
          onClick={() => navigate(ROUTES.knowledge)}
        />
        <MetricCard
          label="P95 响应延迟"
          value="1.9s"
          target="≤ 2.5s"
          delta={-3.2}
          onClick={() => navigate(ROUTES.traces)}
        />
      </div>

      <div className="grid grid-cols-3 gap-4">
        <Card title="近 7 天核心指标趋势" className="col-span-2">
          <TrendChart />
        </Card>
        <Card title="失败类型分布">
          <ul className="flex flex-col gap-3">
            {mockFailureDistribution.map((item) => (
              <li key={item.name}>
                <div className="mb-1 flex items-center justify-between text-[13px] text-slate-600">
                  <span>{item.name}</span>
                  <span className="tabular-nums text-slate-400">{formatPercent(item.value / 100, 0)}</span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-slate-100">
                  <div
                    className="h-1.5 rounded-full bg-brand-600"
                    style={{ width: `${(item.value / maxFailure) * 100}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
        </Card>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <Card title="最近评测">
          <p className="text-[13px] text-slate-600">
            intent-v3 + rag-rrf-rerank 在黄金集 v1 上完成，意图 F1 0.912、recall@5 0.867。
          </p>
          <button
            type="button"
            onClick={() => navigate(ROUTES.evalReport('run-20260930-02'))}
            className="link focus-ring mt-2 cursor-pointer bg-transparent text-[13px]"
          >
            查看评测报告
          </button>
        </Card>
        <Card title="待处理工单">
          <p className="text-[13px] text-slate-600">当前有 1 个高优先级工单待处理，来源为争议理赔转人工。</p>
          <button
            type="button"
            onClick={() => navigate(ROUTES.tickets)}
            className="link focus-ring mt-2 cursor-pointer bg-transparent text-[13px]"
          >
            进入工单中心
          </button>
        </Card>
        <Card title="自进化闭环">
          <p className="text-[13px] text-slate-600">
            本周从 trace 中挖掘 46 条 badcase，其中 38% 为检索未命中，已进入知识库补齐流程。
          </p>
          <button
            type="button"
            onClick={() => navigate(ROUTES.traces)}
            className="link focus-ring mt-2 cursor-pointer bg-transparent text-[13px]"
          >
            查看失败样本
          </button>
        </Card>
      </div>
    </div>
  )
}
