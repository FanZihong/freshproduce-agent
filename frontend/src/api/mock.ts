import type {
  ChatMessage,
  ChatSession,
  EvalDataset,
  EvalReport,
  EvalRun,
  EvalSample,
  IntentResult,
  IntentVersion,
  KnowledgeDoc,
  RetrievalHit,
  SlotItem,
  Ticket,
  ToolCall,
  TraceSpan,
  TraceSummary,
} from '@/types'

export const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

const at = (time: string, day = '2026-09-30') => `${day}T${time}:00+08:00`

export const mockSessions: ChatSession[] = [
  {
    id: 's-1001',
    title: '草莓坏果理赔',
    status: 'handoff',
    lastMessage: '已为您创建工单 TK-20260930-0042，预计 2 小时内审核',
    updatedAt: at('10:42'),
  },
  {
    id: 's-1002',
    title: '订单物流查询',
    status: 'closed',
    lastMessage: '您的包裹已于今日 09:40 签收',
    updatedAt: at('09:55'),
  },
  {
    id: 's-1003',
    title: '退款到账时间咨询',
    status: 'active',
    lastMessage: '退款一般在 1-3 个工作日到账',
    updatedAt: at('09:20'),
  },
  {
    id: 's-1004',
    title: '发票开具咨询',
    status: 'active',
    lastMessage: '电子发票会在订单完成后 24 小时内发送',
    updatedAt: at('08:47'),
  },
]

const orderToolCall: ToolCall = {
  id: 'tc-1001',
  name: 'query_order',
  label: '查询订单',
  status: 'success',
  args: { order_id: '2026093000123' },
  result: '红颜草莓 500g x1，实付 25.80 元，签收时间 09:40',
  durationMs: 312,
}

const policyToolCall: ToolCall = {
  id: 'tc-1002',
  name: 'query_policy',
  label: '查询赔付规则',
  status: 'success',
  args: { category: '浆果类', ratio: '50%' },
  result: '坏果比例 30%-50%：赔付订单金额 50%，需文字或照片凭证',
  durationMs: 486,
}

const ticketToolCall: ToolCall = {
  id: 'tc-1003',
  name: 'create_ticket',
  label: '提交工单',
  status: 'success',
  args: { order_id: '2026093000123', reason: '坏果理赔', amount: '12.90' },
  result: '工单号 TK-20260930-0042，预计 2 小时内审核',
  durationMs: 268,
}

const claimSlots: SlotItem[] = [
  { key: 'order_id', label: '订单号', value: '2026093000123', required: true },
  { key: 'damage_ratio', label: '坏果比例', value: '约 50%', required: true },
  { key: 'evidence', label: '凭证', value: '文字描述（无照片）', required: true },
  { key: 'refund_amount', label: '赔付金额', value: '12.90 元', required: false },
]

const claimRetrieval: RetrievalHit[] = [
  { id: 'r-1', title: '生鲜赔付标准表 3.2 浆果类', source: '赔付标准表.md', score: 0.91 },
  { id: 'r-2', title: '坏果理赔流程说明', source: '售后政策.md', score: 0.84 },
  { id: 'r-3', title: '无照片凭证的处理规则', source: 'FAQ.md', score: 0.72 },
]

export const mockMessages: Record<string, ChatMessage[]> = {
  's-1001': [
    {
      id: 'm-1001',
      sessionId: 's-1001',
      role: 'user',
      content: '我昨天买的草莓到了，坏了一半',
      createdAt: at('10:38'),
    },
    {
      id: 'm-1002',
      sessionId: 's-1001',
      role: 'assistant',
      content:
        '很抱歉给您带来不好的体验。我先帮您确认订单信息，请提供订单号，我马上为您处理理赔。',
      createdAt: at('10:39'),
      intent: { label: 'bad_fruit_claim', name: '坏果理赔', confidence: 0.96 },
      traceId: 'tr-0001',
    },
    {
      id: 'm-1003',
      sessionId: 's-1001',
      role: 'user',
      content: '订单号 2026093000123',
      createdAt: at('10:40'),
    },
    {
      id: 'm-1004',
      sessionId: 's-1001',
      role: 'assistant',
      content:
        '已查到您的订单：红颜草莓 500g，实付 25.80 元，今日 09:40 签收。按售后标准，坏果比例在 30%-50% 之间可赔付订单金额的 50%。请提供坏果照片，或描述一下坏果比例。',
      createdAt: at('10:40'),
      intent: { label: 'bad_fruit_claim', name: '坏果理赔', confidence: 0.94 },
      toolCalls: [orderToolCall, policyToolCall],
      retrieval: claimRetrieval,
      slots: claimSlots.map((slot) =>
        slot.key === 'evidence' ? { ...slot, value: null } : slot,
      ),
      traceId: 'tr-1002',
    },
    {
      id: 'm-1005',
      sessionId: 's-1001',
      role: 'user',
      content: '大概一半吧，照片一直传不上来',
      createdAt: at('10:41'),
    },
    {
      id: 'm-1006',
      sessionId: 's-1001',
      role: 'assistant',
      content:
        '好的，我用文字描述作为凭证为您提交：坏果约 50%、无照片。已按 50% 档位申请退款 12.90 元，工单号 TK-20260930-0042，预计 2 小时内审核完成。',
      createdAt: at('10:42'),
      intent: { label: 'bad_fruit_claim', name: '坏果理赔', confidence: 0.92 },
      toolCalls: [ticketToolCall],
      slots: claimSlots,
      traceId: 'tr-1003',
    },
    {
      id: 'm-1007',
      sessionId: 's-1001',
      role: 'user',
      content: '这个不行，我要全退',
      createdAt: at('10:42'),
    },
    {
      id: 'm-1008',
      sessionId: 's-1001',
      role: 'assistant',
      content:
        '理解您的心情。超出标准的诉求我无法直接判定，已为您转接人工客服，我们刚才的订单信息和理赔记录会一并带过去，您无需重复描述。',
      createdAt: at('10:42'),
      intent: { label: 'human_handoff', name: '转人工', confidence: 0.89 },
      traceId: 'tr-1004',
    },
  ],
  's-1002': [
    {
      id: 'm-2001',
      sessionId: 's-1002',
      role: 'user',
      content: '我的订单到哪了',
      createdAt: at('09:52'),
    },
    {
      id: 'm-2002',
      sessionId: 's-1002',
      role: 'assistant',
      content: '您的包裹已于今日 09:40 签收，签收人为本人。如未收到请告诉我，我帮您核实配送记录。',
      createdAt: at('09:55'),
      intent: { label: 'query_order', name: '订单查询', confidence: 0.95 },
      toolCalls: [
        {
          id: 'tc-2001',
          name: 'query_logistics',
          label: '查询物流',
          status: 'success',
          args: { order_id: '2026092800451' },
          result: '运输中 -> 派送中 -> 已签收（09:40）',
          durationMs: 275,
        },
      ],
      traceId: 'tr-2001',
    },
  ],
  's-1003': [
    {
      id: 'm-3001',
      sessionId: 's-1003',
      role: 'user',
      content: '退款多久能到账',
      createdAt: at('09:18'),
    },
    {
      id: 'm-3002',
      sessionId: 's-1003',
      role: 'assistant',
      content:
        '退款审核通过后一般在 1-3 个工作日到账，原路返回。如果超过 3 个工作日未收到，可以提供工单号，我帮您查询银行处理进度。',
      createdAt: at('09:20'),
      intent: { label: 'refund_consult', name: '退款咨询', confidence: 0.93 },
      retrieval: [
        { id: 'r-301', title: '退款到账时效说明', source: '售后政策.md', score: 0.88 },
      ],
      traceId: 'tr-3001',
    },
  ],
  's-1004': [
    {
      id: 'm-4001',
      sessionId: 's-1004',
      role: 'user',
      content: '怎么开电子发票',
      createdAt: at('08:45'),
    },
    {
      id: 'm-4002',
      sessionId: 's-1004',
      role: 'assistant',
      content:
        '在订单详情页点击"申请开票"，填写抬头与税号即可。电子发票会在订单完成后的 24 小时内发送到您填写的邮箱。',
      createdAt: at('08:47'),
      intent: { label: 'invoice', name: '发票开具', confidence: 0.9 },
      retrieval: [
        { id: 'r-401', title: '电子发票开具流程', source: 'FAQ.md', score: 0.86 },
      ],
      traceId: 'tr-4001',
    },
  ],
}

export const mockTickets: Ticket[] = [
  {
    id: 'TK-20260930-0042',
    sessionId: 's-1001',
    orderId: '2026093000123',
    reason: '坏果理赔（争议：用户要求全额退款）',
    summary: '草莓坏果约 50%，无照片凭证，已按 50% 档位申请退款，用户不接受要求全退，转人工。',
    status: 'pending',
    priority: 'high',
    createdAt: at('10:42'),
    assignee: '待分配',
  },
  {
    id: 'TK-20260930-0038',
    sessionId: 's-1002',
    orderId: '2026092800451',
    reason: '物流签收异常核实',
    summary: '用户反馈未收到包裹但显示已签收，需核实配送记录。',
    status: 'processing',
    priority: 'normal',
    createdAt: at('09:56'),
    assignee: '客服 03',
  },
  {
    id: 'TK-20260929-0117',
    sessionId: 's-1003',
    orderId: '2026092700882',
    reason: '退款超时未到账',
    summary: '退款已审核通过 4 个工作日仍未到账，需查询银行处理状态。',
    status: 'closed',
    priority: 'normal',
    createdAt: at('16:20', '2026-09-29'),
    assignee: '客服 07',
  },
  {
    id: 'TK-20260929-0093',
    sessionId: 's-1004',
    orderId: '2026092600315',
    reason: '发票抬头填写错误',
    summary: '用户需要重新开具发票，原发票需作废。',
    status: 'closed',
    priority: 'normal',
    createdAt: at('11:05', '2026-09-29'),
    assignee: '客服 05',
  },
]

export const mockDatasets: EvalDataset[] = [
  {
    id: 'ds-001',
    name: '生鲜售后黄金集 v1',
    description: '覆盖 10 类意图，人工标注，作为回归基线',
    sampleCount: 48,
    updatedAt: at('18:30', '2026-09-29'),
  },
  {
    id: 'ds-002',
    name: 'JDDC 抽样评测集',
    description: '公开电商客服对话抽样，用于跨分布验证',
    sampleCount: 120,
    updatedAt: at('20:10', '2026-09-28'),
  },
]

export const mockSamples: EvalSample[] = [
  {
    id: 'sp-001',
    datasetId: 'ds-001',
    input: '草莓到货坏了一半，怎么赔',
    expectedIntent: 'bad_fruit_claim',
    expectedAnswer: '按坏果比例匹配赔付档位，收集订单号与凭证后提交工单',
    note: '核心场景，必须命中',
  },
  {
    id: 'sp-002',
    datasetId: 'ds-001',
    input: '那个能退吗',
    expectedIntent: 'refund_consult',
    expectedAnswer: '需结合上一轮商品上下文做指代消解后回答退换货政策',
    note: '多轮指代，检索改写重点用例',
  },
  {
    id: 'sp-003',
    datasetId: 'ds-001',
    input: '三天了还没发货',
    expectedIntent: 'urge_logistics',
    expectedAnswer: '查询订单状态并给出预计发货时间与催单动作',
  },
  {
    id: 'sp-004',
    datasetId: 'ds-001',
    input: '重量不太够啊',
    expectedIntent: 'short_weight',
    expectedAnswer: '引导提供称重照片，按缺重比例匹配赔付档位',
  },
  {
    id: 'sp-005',
    datasetId: 'ds-001',
    input: '今天天气怎么样',
    expectedIntent: 'chitchat',
    expectedAnswer: '识别为域外问题，礼貌拒答并引导售后诉求',
    note: '域外检测用例',
  },
  {
    id: 'sp-006',
    datasetId: 'ds-002',
    input: '我要投诉你们物流太慢',
    expectedIntent: 'complaint',
    expectedAnswer: '安抚情绪，记录投诉内容并判断是否转人工',
  },
]

export const mockRuns: EvalRun[] = [
  {
    id: 'run-20260930-02',
    datasetId: 'ds-001',
    datasetName: '生鲜售后黄金集 v1',
    version: 'intent-v3 + rag-rrf-rerank',
    status: 'finished',
    progress: 100,
    startedAt: at('09:00'),
    finishedAt: at('09:06'),
  },
  {
    id: 'run-20260930-01',
    datasetId: 'ds-002',
    datasetName: 'JDDC 抽样评测集',
    version: 'intent-v2 + rag-hybrid',
    status: 'finished',
    progress: 100,
    startedAt: at('08:20', '2026-09-29'),
    finishedAt: at('08:31', '2026-09-29'),
  },
]

export const mockReport: EvalReport = {
  runId: 'run-20260930-02',
  version: 'intent-v3 + rag-rrf-rerank',
  previousVersion: 'intent-v2 + rag-hybrid',
  metrics: [
    { key: 'intent_f1', label: '意图识别 macro-F1', value: '0.912', target: '≥ 0.90', delta: 0.038 },
    { key: 'recall5', label: '检索 recall@5', value: '0.867', target: '≥ 0.85', delta: 0.061 },
    { key: 'resolution', label: '端到端达成率', value: '82.3%', target: '≥ 80%', delta: 5.4 },
    { key: 'handoff', label: '转人工率', value: '21.8%', target: '≤ 25%', delta: -6.2 },
  ],
  comparisons: [
    { label: '意图 macro-F1', current: '0.912', previous: '0.874', delta: 0.038 },
    { label: '检索 recall@5', current: '0.867', previous: '0.806', delta: 0.061 },
    { label: '检索 MRR', current: '0.741', previous: '0.688', delta: 0.053 },
    { label: '工具调用成功率', current: '96.4%', previous: '94.1%', delta: 2.3 },
    { label: '端到端达成率', current: '82.3%', previous: '76.9%', delta: 5.4 },
    { label: '平均处理轮次', current: '3.4', previous: '3.9', delta: -0.5 },
    { label: '转人工率', current: '21.8%', previous: '28.0%', delta: -6.2 },
  ],
}

export const mockTraces: TraceSummary[] = [
  {
    id: 'tr-1004',
    sessionId: 's-1001',
    intent: '转人工',
    status: 'failed',
    failureType: '争议场景升级人工',
    durationMs: 2180,
    tokenCost: 0.0043,
    createdAt: at('10:42'),
  },
  {
    id: 'tr-1003',
    sessionId: 's-1001',
    intent: '坏果理赔',
    status: 'success',
    durationMs: 3120,
    tokenCost: 0.0071,
    createdAt: at('10:42'),
  },
  {
    id: 'tr-2001',
    sessionId: 's-1002',
    intent: '订单查询',
    status: 'success',
    durationMs: 1460,
    tokenCost: 0.0028,
    createdAt: at('09:55'),
  },
  {
    id: 'tr-3001',
    sessionId: 's-1003',
    intent: '退款咨询',
    status: 'success',
    durationMs: 1180,
    tokenCost: 0.0021,
    createdAt: at('09:20'),
  },
  {
    id: 'tr-5001',
    sessionId: 's-1005',
    intent: '坏果理赔',
    status: 'failed',
    failureType: '检索未命中（赔付档位缺失）',
    durationMs: 2640,
    tokenCost: 0.0058,
    createdAt: at('11:05'),
  },
  {
    id: 'tr-5002',
    sessionId: 's-1006',
    intent: '缺斤少两',
    status: 'failed',
    failureType: '工具超时（query_policy）',
    durationMs: 5340,
    tokenCost: 0.0062,
    createdAt: at('11:18'),
  },
]

export const mockSpanTree: TraceSpan[] = [
  {
    id: 'span-1',
    name: 'langgraph.workflow',
    type: 'workflow',
    status: 'success',
    durationMs: 3120,
    detail: { session_id: 's-1001', checkpoint: 'cp-20260930-1042', nodes: '6' },
    children: [
      {
        id: 'span-2',
        name: 'intent.classify',
        type: 'intent',
        status: 'success',
        durationMs: 78,
        detail: { model: 'intent-v3-lora', label: 'bad_fruit_claim', confidence: '0.92' },
      },
      {
        id: 'span-3',
        name: 'rag.retrieve',
        type: 'retrieval',
        status: 'success',
        durationMs: 940,
        detail: { rewrite: '草莓坏果赔付标准', bm25_hits: '20', vector_hits: '20', rrf: 'k=60', rerank_top: '3' },
        children: [
          {
            id: 'span-4',
            name: 'rag.rerank',
            type: 'retrieval',
            status: 'success',
            durationMs: 412,
            detail: { model: 'bge-reranker-v2-m3', input: '40', output: '3' },
          },
        ],
      },
      {
        id: 'span-5',
        name: 'tool.query_order',
        type: 'tool',
        status: 'success',
        durationMs: 312,
        detail: { order_id: '2026093000123', http_status: '200' },
      },
      {
        id: 'span-6',
        name: 'tool.query_policy',
        type: 'tool',
        status: 'success',
        durationMs: 486,
        detail: { category: '浆果类', ratio: '50%', hit: '赔付标准表 3.2' },
      },
      {
        id: 'span-7',
        name: 'llm.generate',
        type: 'llm',
        status: 'success',
        durationMs: 1240,
        detail: { model: 'deepseek-chat', prompt_tokens: '1840', completion_tokens: '216' },
      },
    ],
  },
]

export const mockDocs: KnowledgeDoc[] = [
  { id: 'doc-01', name: '生鲜赔付标准表.md', category: '赔付规则', chunkCount: 42, status: 'indexed', updatedAt: at('17:00', '2026-09-29') },
  { id: 'doc-02', name: '售后政策总则.md', category: '售后政策', chunkCount: 68, status: 'indexed', updatedAt: at('17:05', '2026-09-29') },
  { id: 'doc-03', name: '常见问题 FAQ.md', category: 'FAQ', chunkCount: 96, status: 'indexed', updatedAt: at('17:20', '2026-09-29') },
  { id: 'doc-04', name: '物流时效说明.md', category: '物流', chunkCount: 24, status: 'indexed', updatedAt: at('18:02', '2026-09-29') },
  { id: 'doc-05', name: '商品储存手册.md', category: '商品知识', chunkCount: 35, status: 'indexing', updatedAt: at('10:12') },
  { id: 'doc-06', name: '争议订单处理规范.md', category: '售后政策', chunkCount: 0, status: 'failed', updatedAt: at('10:30') },
]

export const mockVersions: IntentVersion[] = [
  {
    id: 'iv-3',
    version: 'intent-v3-lora',
    status: 'ready',
    f1: 0.912,
    latencyMs: 78,
    costPerK: 0.12,
    note: 'Qwen 1.5B LoRA，蒸馏标注 2400 条',
    createdAt: at('09:00'),
  },
  {
    id: 'iv-2',
    version: 'intent-v2-lora',
    status: 'ready',
    f1: 0.874,
    latencyMs: 82,
    costPerK: 0.12,
    note: '训练集 1200 条，未做难例扩充',
    createdAt: at('15:30', '2026-09-26'),
  },
  {
    id: 'iv-1',
    version: 'few-shot-baseline',
    status: 'ready',
    f1: 0.836,
    latencyMs: 940,
    costPerK: 6.8,
    note: '大模型 few-shot 分类基线',
    createdAt: at('11:10', '2026-09-24'),
  },
  {
    id: 'iv-4',
    version: 'intent-v4-lora',
    status: 'training',
    f1: 0,
    latencyMs: 0,
    costPerK: 0,
    note: '扩充争议场景难例后训练中',
    createdAt: at('10:50'),
  },
]

export const mockFailureDistribution = [
  { name: '检索未命中', value: 38 },
  { name: '意图误判', value: 22 },
  { name: '工具超时', value: 16 },
  { name: '槽位缺失', value: 14 },
  { name: '其他', value: 10 },
]

export const mockTrend = [
  { date: '09-24', resolution: 71.2, recall: 78.5 },
  { date: '09-25', resolution: 73.8, recall: 79.9 },
  { date: '09-26', resolution: 75.1, recall: 81.6 },
  { date: '09-27', resolution: 76.4, recall: 82.3 },
  { date: '09-28', resolution: 78.9, recall: 83.8 },
  { date: '09-29', resolution: 80.2, recall: 85.1 },
  { date: '09-30', resolution: 82.3, recall: 86.7 },
]

interface MockReply {
  intent: IntentResult
  content: string
  toolCalls?: ToolCall[]
  slots?: SlotItem[]
  retrieval?: RetrievalHit[]
}

export function buildMockReply(text: string): MockReply {
  if (text.includes('坏') || text.includes('草莓') || text.includes('理赔')) {
    return {
      intent: { label: 'bad_fruit_claim', name: '坏果理赔', confidence: 0.93 },
      content:
        '已识别为坏果理赔。请提供订单号，我查到订单后会按赔付标准表匹配档位，并根据您提供的凭证提交工单；若凭证不足或对赔付金额有异议，我会转接人工客服。',
      toolCalls: [orderToolCall, policyToolCall],
      slots: claimSlots.map((slot) => ({ ...slot, value: null })),
      retrieval: claimRetrieval,
    }
  }
  if (text.includes('物流') || text.includes('发货') || text.includes('到哪')) {
    return {
      intent: { label: 'urge_logistics', name: '催物流', confidence: 0.91 },
      content: '我来帮您查询物流进度，请提供订单号；如果超过承诺时效，我会为您创建催单工单并跟进。',
      toolCalls: [
        {
          id: `tc-${Date.now()}`,
          name: 'query_logistics',
          label: '查询物流',
          status: 'success',
          args: { order_id: '2026093000123' },
          result: '运输中，预计今日 18:00 前送达',
          durationMs: 296,
        },
      ],
    }
  }
  if (text.includes('退') || text.includes('退款')) {
    return {
      intent: { label: 'refund_consult', name: '退款咨询', confidence: 0.9 },
      content: '退款审核通过后一般 1-3 个工作日原路到账。您提供订单号或工单号，我可以帮您查询当前处理进度。',
      retrieval: [{ id: 'r-501', title: '退款到账时效说明', source: '售后政策.md', score: 0.87 }],
    }
  }
  if (text.includes('发票')) {
    return {
      intent: { label: 'invoice', name: '发票开具', confidence: 0.92 },
      content: '在订单详情页点击"申请开票"，填写抬头与税号即可，电子发票会在 24 小时内发送到邮箱。',
    }
  }
  if (text.includes('投诉') || text.includes('人工')) {
    return {
      intent: { label: 'human_handoff', name: '转人工', confidence: 0.88 },
      content: '已为您转接人工客服，并同步本次对话的订单信息与处理记录，您无需重复描述问题。',
    }
  }
  return {
    intent: { label: 'chitchat', name: '域外问题', confidence: 0.84 },
    content:
      '这个问题超出了售后服务的范围。我可以帮您处理订单查询、坏果理赔、退款咨询、发票开具等售后问题，请告诉我具体情况。',
  }
}

export function buildMockTraceDetail(traceId: string): TraceSpan[] {
  const root: TraceSpan = {
    ...mockSpanTree[0],
    detail: { ...mockSpanTree[0].detail, trace_id: traceId },
  }
  return [root]
}
