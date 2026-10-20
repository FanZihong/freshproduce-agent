import { useState } from 'react'
import type { TraceSpan } from '@/types'
import { Tag } from '@/components/ui/Tag'
import { SPAN_TYPE_LABEL } from '@/constants/status'
import { cn, formatDuration } from '@/utils/format'

interface TraceTreeProps {
  spans: TraceSpan[]
  onSelect?: (span: TraceSpan) => void
  selectedId?: string
}

interface SpanNodeProps {
  span: TraceSpan
  depth: number
  onSelect?: (span: TraceSpan) => void
  selectedId?: string
}

function SpanNode({ span, depth, onSelect, selectedId }: SpanNodeProps) {
  const isSelected = selectedId === span.id
  const hasChildren = Boolean(span.children && span.children.length > 0)
  const [expanded, setExpanded] = useState(true)

  return (
    <li>
      <div
        onClick={() => onSelect?.(span)}
        style={{ paddingLeft: depth * 16 }}
        className={cn(
          'flex cursor-pointer items-center justify-between gap-3 rounded-md px-2 py-1.5 transition-colors duration-150',
          isSelected ? 'bg-brand-50' : 'hover:bg-slate-50',
        )}
      >
        <span className="flex min-w-0 items-center gap-2">
          {hasChildren ? (
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                setExpanded((value) => !value)
              }}
              className="focus-ring w-4 cursor-pointer text-xs text-slate-400"
              aria-label={expanded ? '收起' : '展开'}
            >
              {expanded ? '-' : '+'}
            </button>
          ) : (
            <span className="w-4" />
          )}
          <span className="mono truncate text-[13px] text-slate-700" title={span.name}>
            {span.name}
          </span>
          <Tag variant="neutral">{SPAN_TYPE_LABEL[span.type] ?? span.type}</Tag>
        </span>
        <span className="flex items-center gap-2">
          <span className="tabular-nums text-xs text-slate-400">
            {formatDuration(span.durationMs)}
          </span>
          <Tag variant={span.status === 'success' ? 'success' : 'danger'} dot>
            {span.status === 'success' ? '成功' : '失败'}
          </Tag>
        </span>
      </div>
      {hasChildren && expanded ? (
        <ul className="mt-0.5">
          {span.children?.map((child) => (
            <SpanNode
              key={child.id}
              span={child}
              depth={depth + 1}
              onSelect={onSelect}
              selectedId={selectedId}
            />
          ))}
        </ul>
      ) : null}
    </li>
  )
}

export function TraceTree({ spans, onSelect, selectedId }: TraceTreeProps) {
  return (
    <ul className="flex flex-col gap-1">
      {spans.map((span) => (
        <SpanNode key={span.id} span={span} depth={0} onSelect={onSelect} selectedId={selectedId} />
      ))}
    </ul>
  )
}
