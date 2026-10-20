import { useState } from 'react'
import type { ToolCall } from '@/types'
import { Tag } from '@/components/ui/Tag'
import { formatDuration } from '@/utils/format'

interface ToolCallCardProps {
  toolCall: ToolCall
}

const STATUS_META = {
  running: { label: '执行中', variant: 'warning' as const },
  success: { label: '成功', variant: 'success' as const },
  failed: { label: '失败', variant: 'danger' as const },
}

export function ToolCallCard({ toolCall }: ToolCallCardProps) {
  const [expanded, setExpanded] = useState(false)
  const statusMeta = STATUS_META[toolCall.status]

  return (
    <div className="rounded-md border border-slate-200 bg-slate-50">
      <button
        type="button"
        onClick={() => setExpanded((value) => !value)}
        className="focus-ring flex w-full cursor-pointer items-center justify-between gap-2 px-3 py-2 text-left"
      >
        <span className="flex items-center gap-2">
          <span className="text-[13px] font-medium text-slate-700">{toolCall.label}</span>
          <span className="mono text-xs text-slate-400">{toolCall.name}</span>
        </span>
        <span className="flex items-center gap-2">
          {toolCall.durationMs ? (
            <span className="tabular-nums text-xs text-slate-400">
              {formatDuration(toolCall.durationMs)}
            </span>
          ) : null}
          <Tag variant={statusMeta.variant} dot>
            {statusMeta.label}
          </Tag>
        </span>
      </button>
      {expanded ? (
        <div className="border-t border-slate-200 px-3 py-2">
          <p className="mb-1 text-xs font-medium text-slate-500">入参</p>
          <pre className="mono mb-2 overflow-auto rounded bg-white p-2 text-xs text-slate-600">
            {JSON.stringify(toolCall.args, null, 2)}
          </pre>
          {toolCall.result ? (
            <>
              <p className="mb-1 text-xs font-medium text-slate-500">返回</p>
              <p className="rounded bg-white p-2 text-xs text-slate-600">{toolCall.result}</p>
            </>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
