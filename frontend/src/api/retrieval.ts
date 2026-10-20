import type { RetrievalSearchResult } from '@/types'
import { http, USE_MOCK } from './client'
import { delay } from './mock'

export interface RetrievalSearchParams {
  query: string
  context?: string
  product?: string
  topK?: number
  kind?: 'bm25' | 'vector' | 'hybrid'
  rerank?: 'off' | 'score' | 'bge'
  strategy?: 'rule' | 'paragraph'
  rrfK?: number
}

const RERANK_LABEL: Record<string, string> = {
  off: '不重排',
  score: 'fusion-score',
  bge: 'cross-encoder:BAAI/bge-reranker-base',
}

export async function searchRetrieval(
  params: RetrievalSearchParams,
): Promise<RetrievalSearchResult> {
  const topK = params.topK ?? 5
  const kind = params.kind ?? 'hybrid'
  const rerank = params.rerank ?? 'score'
  const strategy = params.strategy ?? 'rule'
  const rrfK = params.rrfK ?? 60

  if (USE_MOCK) {
    await delay(200)
    return mockResult(params.query, params.context ?? '', topK, kind, rerank, strategy, rrfK)
  }

  const { data } = await http.get<RetrievalSearchResult>('/retrieval/search', {
    params: {
      query: params.query,
      context: params.context || undefined,
      product: params.product || undefined,
      top_k: topK,
      kind,
      rerank,
      strategy,
      rrf_k: rrfK,
    },
  })
  return data
}

function mockHit(rank: number, chunkId: string, docTitle: string, section: string, text: string) {
  return {
    rank,
    chunkId,
    docTitle,
    section,
    source: '示例来源（Mock 数据）',
    text,
    score: Number((1 / (rank + 1)).toFixed(4)),
    fusedScore: Number((0.03 / rank).toFixed(6)),
    rerankScore: Number((1 / (rank + 0.2)).toFixed(4)),
    ranks: { bm25: rank, vector: rank + 1 },
    movement: 0,
  }
}

function mockResult(
  query: string,
  context: string,
  topK: number,
  kind: string,
  rerank: string,
  strategy: string,
  rrfK: number,
): RetrievalSearchResult {
  const samples = [
    ['常见问题FAQ#rule#016', '常见问题 FAQ', '4. 商品与储存', '问：车厘子怎么保存？答：冷藏 0 至 4 度，食用前再清洗，建议 3 天内吃完。'],
    ['商品储存手册#rule#001', '商品储存手册', '1. 浆果类', '草莓、蓝莓、树莓建议冷藏 0 至 4 度保存，食用前再清洗。'],
    ['商品储存手册#rule#010', '商品储存手册', '3. 叶菜与根茎', '胡萝卜、白萝卜去掉叶子后冷藏，可存放 1 周左右。'],
  ]
  const hits = samples.slice(0, topK).map((item, index) => mockHit(index + 1, item[0], item[1], item[2], item[3]))
  return {
    query,
    rewritten: {
      original: query,
      query: context ? `${context}` : query,
      rewritten: Boolean(context),
      reason: context ? 'MOCK：用上文商品替换指代词' : 'MOCK：问题自包含，无需改写',
      strategy: 'rule',
      latency_ms: 0,
    },
    config: {
      kind,
      rerank,
      strategy,
      rrfK,
      topK,
      candidateK: Math.max(topK * 4, 20),
      embedder: 'hash-512（Mock 模式）',
      reranker: RERANK_LABEL[rerank] ?? rerank,
    },
    stages: { bm25: hits, vector: hits, fused: hits, reranked: hits },
    citations: hits.slice(0, 2).map((hit) => ({
      chunk_id: hit.chunkId,
      title: hit.docTitle,
      section: hit.section,
      source: hit.source,
      snippet: hit.text.slice(0, 80),
      score: hit.score,
    })),
    elapsedMs: 12,
  }
}
