import { useEffect, useMemo, useState } from 'react'
import { RefreshCw, Upload } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Modal } from '@/components/ui/Modal'
import { Table } from '@/components/ui/Table'
import type { Column } from '@/components/ui/Table'
import { Tag } from '@/components/ui/Tag'
import { DOC_STATUS } from '@/constants/status'
import { fetchDocs, uploadDoc } from '@/api/knowledge'
import type { KnowledgeDoc } from '@/types'
import { formatDateTime } from '@/utils/date'
import { useUiStore } from '@/stores/uiStore'

const CATEGORIES = ['全部', '赔付规则', '售后政策', 'FAQ', '物流', '商品知识']

export function KnowledgePage() {
  const pushToast = useUiStore((state) => state.pushToast)
  const [docs, setDocs] = useState<KnowledgeDoc[]>([])
  const [loading, setLoading] = useState(true)
  const [category, setCategory] = useState('全部')
  const [keyword, setKeyword] = useState('')
  const [uploadOpen, setUploadOpen] = useState(false)
  const [fileName, setFileName] = useState('')
  const [newCategory, setNewCategory] = useState('售后政策')
  const [uploading, setUploading] = useState(false)
  const [preview, setPreview] = useState<KnowledgeDoc | null>(null)

  useEffect(() => {
    let mounted = true
    void fetchDocs().then((data) => {
      if (!mounted) return
      setDocs(data)
      setLoading(false)
    })
    return () => {
      mounted = false
    }
  }, [])

  const filtered = useMemo(
    () =>
      docs.filter((doc) => {
        const matchCategory = category === '全部' || doc.category === category
        const matchKeyword = !keyword.trim() || doc.name.includes(keyword.trim())
        return matchCategory && matchKeyword
      }),
    [category, docs, keyword],
  )

  const handleUpload = async () => {
    if (!fileName.trim()) return
    setUploading(true)
    try {
      const doc = await uploadDoc(fileName.trim(), newCategory)
      setDocs((previous) => [doc, ...previous])
      setUploadOpen(false)
      setFileName('')
      pushToast({ title: '文档已上传', description: '索引构建中', variant: 'success' })
    } finally {
      setUploading(false)
    }
  }

  const columns: Column<KnowledgeDoc>[] = [
    { key: 'name', title: '文档名称' },
    { key: 'category', title: '分类' },
    {
      key: 'chunkCount',
      title: '规则块',
      align: 'right',
      render: (row) => String(row.chunkCount),
    },
    {
      key: 'chunkCountParagraph',
      title: '段落块',
      align: 'right',
      render: (row) => (row.chunkCountParagraph == null ? '-' : String(row.chunkCountParagraph)),
    },
    {
      key: 'sections',
      title: '小节',
      align: 'right',
      render: (row) => (row.sections == null ? '-' : String(row.sections)),
    },
    {
      key: 'status',
      title: '索引状态',
      render: (row) => {
        const meta = DOC_STATUS[row.status]
        return (
          <Tag variant={meta.variant} dot>
            {meta.label}
          </Tag>
        )
      },
    },
    {
      key: 'updatedAt',
      title: '更新时间',
      align: 'right',
      render: (row) => formatDateTime(row.updatedAt),
    },
  ]

  return (
    <div className="flex flex-col gap-4 p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          {CATEGORIES.map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => setCategory(item)}
              className={
                category === item
                  ? 'focus-ring cursor-pointer rounded-md bg-brand-50 px-3 py-1.5 text-[13px] font-medium text-brand-700'
                  : 'focus-ring cursor-pointer rounded-md px-3 py-1.5 text-[13px] text-slate-500 transition-colors duration-150 hover:bg-slate-100'
              }
            >
              {item}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <div className="w-56">
            <Input
              placeholder="搜索文档"
              value={keyword}
              onChange={(event) => setKeyword(event.target.value)}
            />
          </div>
          <Button
            variant="secondary"
            icon={<RefreshCw size={15} />}
            onClick={() => pushToast({ title: '已触发重建索引', variant: 'info' })}
          >
            重建索引
          </Button>
          <Button variant="primary" icon={<Upload size={15} />} onClick={() => setUploadOpen(true)}>
            上传文档
          </Button>
        </div>
      </div>

      <Table
        columns={columns}
        rows={filtered}
        rowKey={(row) => row.id}
        loading={loading}
        onRowClick={(row) => setPreview(row)}
        emptyText="没有符合条件的文档"
      />

      <Modal
        open={uploadOpen}
        title="上传知识库文档"
        onClose={() => setUploadOpen(false)}
        footer={
          <>
            <Button variant="secondary" onClick={() => setUploadOpen(false)}>
              取消
            </Button>
            <Button variant="primary" disabled={uploading || !fileName.trim()} onClick={handleUpload}>
              {uploading ? '上传中' : '确认上传'}
            </Button>
          </>
        }
      >
        <div className="flex flex-col gap-4">
          <Input
            label="文档名称"
            placeholder="例如：赔付标准表.md"
            value={fileName}
            onChange={(event) => setFileName(event.target.value)}
          />
          <div className="flex flex-col gap-1.5">
            <label htmlFor="doc-category" className="text-[13px] font-medium text-slate-700">
              文档分类
            </label>
            <select
              id="doc-category"
              value={newCategory}
              onChange={(event) => setNewCategory(event.target.value)}
              className="focus-ring h-9 cursor-pointer rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900 transition-colors duration-150 hover:border-slate-300 focus:border-brand-600"
            >
              {CATEGORIES.filter((item) => item !== '全部').map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </div>
          <p className="text-xs text-slate-400">
            上传后会进入索引流程，索引完成即可被混合检索命中。
          </p>
        </div>
      </Modal>

      <Modal open={Boolean(preview)} title="文档详情" onClose={() => setPreview(null)}>
        {preview ? (
          <div className="flex flex-col gap-3 text-[13px]">
            <div className="flex items-center justify-between">
              <span className="text-slate-800">{preview.name}</span>
              <Tag variant={DOC_STATUS[preview.status].variant} dot>
                {DOC_STATUS[preview.status].label}
              </Tag>
            </div>
            <dl className="grid grid-cols-2 gap-y-2 text-slate-600">
              <dt className="text-slate-500">文档 ID</dt>
              <dd className="mono">{preview.docId ?? '-'}</dd>
              <dt className="text-slate-500">分类</dt>
              <dd>{preview.category}</dd>
              <dt className="text-slate-500">分块数（规则 / 段落）</dt>
              <dd className="tabular-nums">
                {preview.chunkCount} / {preview.chunkCountParagraph ?? '-'}
              </dd>
              <dt className="text-slate-500">小节数</dt>
              <dd className="tabular-nums">{preview.sections ?? '-'}</dd>
              <dt className="text-slate-500">更新时间</dt>
              <dd>{formatDateTime(preview.updatedAt)}</dd>
            </dl>
            {preview.source ? (
              <div className="rounded-md border border-slate-100 bg-slate-50 px-3 py-2 text-slate-600">
                来源：{preview.source}
              </div>
            ) : null}
            <p className="text-xs text-slate-400">
              想看具体分块内容与检索过程，请到「检索调试台」按关键词检索，点击结果可查看 chunk 原文。
            </p>
          </div>
        ) : null}
      </Modal>
    </div>
  )
}
