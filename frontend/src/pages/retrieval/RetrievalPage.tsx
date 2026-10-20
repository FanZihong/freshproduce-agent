import { useState } from 'react'
import { Search } from 'lucide-react'
import { searchRetrieval } from '@/api/retrieval'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Empty } from '@/components/ui/Empty'
import { Input } from '@/components/ui/Input'
import { Modal } from '@/components/ui/Modal'
import { Skeleton } from '@/components/ui/Skeleton'
import { Tag } from '@/components/ui/Tag'
import { RetrievalHitList } from '@/components/business/RetrievalHitList'
import type { RetrievalHitItem, RetrievalSearchResult } from '@/types'
import { useUiStore } from '@/stores/uiStore'

type Kind = 'bm25' | 'vector' | 'hybrid'
type Rerank = 'off' | 'score' | 'bge'
type Strategy = 'rule' | 'paragraph'

const SELECT_CLASS =
  'focus-ring h-9 w-full cursor-pointer rounded-md border border-slate-200 bg-white px-2 text-[13px] text-slate-700 transition-colors duration-150 hover:border-slate-300 focus:border-brand-600'

const PRESETS: Array<{ label: string; query: string; context?: string; product?: string }> = [
  { label: '单轮检索', query: '车厘子怎么保存' },
  { label: '多轮指代', query: '那它呢', context: '草莓怎么保存', product: '草莓' },
  { label: '档位规则', query: '坏果六成怎么算赔付' },
  { label: '知识库没有', query: '山竹怎么保存' },
]

export function RetrievalPage() {
  const pushToast = useUiStore((state) => state.pushToast)

  const [query, setQuery] = useState('车厘子怎么保存')
  const [context, setContext] = useState('')
  const [product, setProduct] = useState('')
  const [kind, setKind] = useState<Kind>('hybrid')
  const [rerank, setRerank] = useState<Rerank>('score')
  const [strategy, setStrategy] = useState<Strategy>('rule')
  const [rrfK, setRrfK] = useState(60)
  const [topK, setTopK] = useState(5)

  const [result, setResult] = useState<RetrievalSearchResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [activeHit, setActiveHit] = useState<RetrievalHitItem | null>(null)

  const run = async (override?: { query?: string; context?: string; product?: string }) => {
    const finalQuery = (override?.query ?? query).trim()
    if (!finalQuery) {
      pushToast({ title: '请输入检索问题', variant: 'warning' })
      return
    }
    setLoading(true)
    try {
      const data = await searchRetrieval({
        query: finalQuery,
        context: override?.context ?? context,
        product: override?.product ?? product,
        topK,
        kind,
        rerank,
        strategy,
        rrfK,
      })
      setResult(data)
    } catch (error) {
      pushToast({
        title: '检索失败',
        description: error instanceof Error ? error.message : '请确认后端服务已启动',
        variant: 'danger',
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-4 p-5">
      <Card
        title="检索调试台"
        extra={
          <span className="text-xs text-slate-400">
            改写 → BM25 / 向量召回 → RRF 融合 → 精排 → 引用过滤
          </span>
        }
      >
        <div className="flex flex-col gap-3">
          <div className="flex items-end gap-2">
            <Input
              label="问题"
              placeholder="例如：车厘子怎么保存"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') void run()
              }}
            />
            <Button
              variant="primary"
              icon={<Search size={15} />}
              disabled={loading}
              onClick={() => void run()}
            >
              {loading ? '检索中' : '检索'}
            </Button>
          </div>

          <div className="grid grid-cols-7 gap-3">
            <Input
              label="上一轮问题"
              placeholder="可选，多轮用"
              value={context}
              onChange={(event) => setContext(event.target.value)}
            />
            <Input
              label="上文商品"
              placeholder="可选"
              value={product}
              onChange={(event) => setProduct(event.target.value)}
            />
            <label className="flex flex-col gap-1.5 text-[13px] font-medium text-slate-700">
              检索方式
              <select
                className={SELECT_CLASS}
                value={kind}
                onChange={(event) => setKind(event.target.value as Kind)}
              >
                <option value="hybrid">混合（RRF）</option>
                <option value="bm25">仅 BM25</option>
                <option value="vector">仅向量</option>
              </select>
            </label>
            <label className="flex flex-col gap-1.5 text-[13px] font-medium text-slate-700">
              重排方式
              <select
                className={SELECT_CLASS}
                value={rerank}
                onChange={(event) => setRerank(event.target.value as Rerank)}
              >
                <option value="score">融合分数基线</option>
                <option value="bge">CrossEncoder 精排</option>
                <option value="off">不重排</option>
              </select>
            </label>
            <label className="flex flex-col gap-1.5 text-[13px] font-medium text-slate-700">
              分块策略
              <select
                className={SELECT_CLASS}
                value={strategy}
                onChange={(event) => setStrategy(event.target.value as Strategy)}
              >
                <option value="rule">按规则条</option>
                <option value="paragraph">按段落窗口</option>
              </select>
            </label>
            <Input
              label="RRF k"
              type="number"
              value={rrfK}
              onChange={(event) => setRrfK(Number(event.target.value) || 60)}
            />
            <Input
              label="返回条数"
              type="number"
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value) || 5)}
            />
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">预设：</span>
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                type="button"
                onClick={() => {
                  setQuery(preset.query)
                  setContext(preset.context ?? '')
                  setProduct(preset.product ?? '')
                  void run(preset)
                }}
                className="focus-ring cursor-pointer rounded-md border border-slate-200 px-2 py-1 text-xs text-slate-600 transition-colors duration-150 hover:border-brand-600 hover:text-brand-700"
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
      </Card>

      {loading ? (
        <Card title="检索中">
          <Skeleton lines={4} />
        </Card>
      ) : null}

      {!result && !loading ? (
        <Empty
          title="还没有检索结果"
          description="输入问题后点击检索，或直接用上面的预设用例"
        />
      ) : null}

      {result && !loading ? (
        <>
          <div className="grid grid-cols-3 gap-4">
            <Card title="① 查询改写" className="col-span-2">
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2 text-[13px]">
                  <span className="text-slate-500">原始</span>
                  <span className="text-slate-800">{result.rewritten.original}</span>
                  <span className="text-slate-300">→</span>
                  <span className="text-slate-500">改写后</span>
                  <span className="font-medium text-brand-700">{result.rewritten.query}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Tag variant={result.rewritten.rewritten ? 'success' : 'neutral'} dot>
                    {result.rewritten.rewritten ? '已改写' : '未改写'}
                  </Tag>
                  <Tag variant="info">{result.rewritten.strategy}</Tag>
                  <span className="text-xs text-slate-400">{result.rewritten.reason}</span>
                </div>
              </div>
            </Card>
            <Card title="检索配置">
              <dl className="flex flex-col gap-1.5 text-[12px]">
                <div className="flex justify-between">
                  <dt className="text-slate-500">向量模型</dt>
                  <dd className="text-slate-700">{result.config.embedder}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-500">重排</dt>
                  <dd className="text-slate-700">{result.config.reranker}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-500">候选池 / RRF k</dt>
                  <dd className="tabular-nums text-slate-700">
                    {result.config.candidateK} / {result.config.rrfK}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-500">耗时</dt>
                  <dd className="tabular-nums text-slate-700">{result.elapsedMs} ms</dd>
                </div>
              </dl>
            </Card>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Card title={`② BM25 召回（top${result.config.topK}）`}>
              <RetrievalHitList hits={result.stages.bm25} onSelect={setActiveHit} />
            </Card>
            <Card title={`③ 向量召回（top${result.config.topK}）`}>
              <RetrievalHitList hits={result.stages.vector} onSelect={setActiveHit} />
            </Card>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Card title={`④ RRF 融合（候选池 ${result.stages.fused.length} 条）`}>
              <RetrievalHitList
                hits={result.stages.fused}
                showPathRanks
                onSelect={setActiveHit}
              />
            </Card>
            <Card title={`⑤ 精排结果（top${result.config.topK}）`}>
              <RetrievalHitList
                hits={result.stages.reranked}
                showMovement
                onSelect={setActiveHit}
              />
            </Card>
          </div>

          <Card title={`⑥ 引用（符合有据规则 ${result.citations.length} 条）`}>
            {result.citations.length === 0 ? (
              <p className="text-xs text-slate-400">
                没有符合有据规则的引用——Agent 会如实告知并转人工
              </p>
            ) : (
              <ul className="flex flex-col gap-1.5">
                {result.citations.map((citation) => (
                  <li
                    key={citation.chunk_id}
                    className="rounded-md border border-slate-200 px-3 py-2"
                  >
                    <p className="text-[13px] text-slate-700">
                      《{citation.title}》{citation.section}
                    </p>
                    <p className="mt-1 text-[12px] text-slate-400">{citation.source}</p>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </>
      ) : null}

      <Modal
        open={Boolean(activeHit)}
        title="Chunk 原文"
        width={640}
        onClose={() => setActiveHit(null)}
      >
        {activeHit ? (
          <div className="flex flex-col gap-3 text-[13px]">
            <div className="flex items-center gap-2">
              <span className="mono text-slate-600">{activeHit.chunkId}</span>
              <Tag variant="neutral">排名 {activeHit.rank}</Tag>
              <Tag variant="info">分数 {activeHit.score.toFixed(4)}</Tag>
            </div>
            <dl className="grid grid-cols-2 gap-y-1.5 text-slate-600">
              <dt className="text-slate-500">文档</dt>
              <dd>《{activeHit.docTitle}》</dd>
              <dt className="text-slate-500">小节</dt>
              <dd>{activeHit.section}</dd>
              <dt className="text-slate-500">来源</dt>
              <dd>{activeHit.source}</dd>
              <dt className="text-slate-500">两路排名</dt>
              <dd className="tabular-nums">
                BM25 {activeHit.ranks?.bm25 ?? '-'} / 向量 {activeHit.ranks?.vector ?? '-'}
              </dd>
            </dl>
            <div className="rounded-md border border-slate-100 bg-slate-50 px-3 py-2 leading-relaxed text-slate-700">
              {activeHit.text}
            </div>
          </div>
        ) : null}
      </Modal>
    </div>
  )
}
