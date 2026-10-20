import type { RetrievalHitItem } from '@/types'
import { cn } from '@/utils/format'

interface RetrievalHitListProps {
  hits: RetrievalHitItem[]
  emptyText?: string
  showMovement?: boolean
  showPathRanks?: boolean
  onSelect?: (hit: RetrievalHitItem) => void
}

const MOVEMENT_CLASS = {
  up: 'text-brand-600',
  down: 'text-red-600',
  flat: 'text-slate-400',
}

function movementMeta(movement: number | null | undefined) {
  if (!movement) return { text: '—', className: MOVEMENT_CLASS.flat }
  if (movement > 0) return { text: `↑${movement}`, className: MOVEMENT_CLASS.up }
  return { text: `↓${Math.abs(movement)}`, className: MOVEMENT_CLASS.down }
}

export function RetrievalHitList({
  hits,
  emptyText = '没有结果',
  showMovement = false,
  showPathRanks = false,
  onSelect,
}: RetrievalHitListProps) {
  if (hits.length === 0) {
    return <p className="text-xs text-slate-400">{emptyText}</p>
  }

  return (
    <ul className="flex flex-col gap-1.5">
      {hits.map((hit) => {
        const movement = movementMeta(hit.movement)
        return (
          <li key={`${hit.chunkId}-${hit.rank}`}>
            <button
              type="button"
              onClick={() => onSelect?.(hit)}
              className="focus-ring w-full cursor-pointer rounded-md border border-slate-200 px-3 py-2 text-left transition-colors duration-150 hover:border-slate-300 hover:bg-slate-50"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="flex min-w-0 items-center gap-2">
                  <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-slate-100 text-[11px] font-medium tabular-nums text-slate-600">
                    {hit.rank}
                  </span>
                  <span className="ellipsis mono text-[12px] text-slate-500">{hit.chunkId}</span>
                </span>
                <span className="flex shrink-0 items-center gap-2 text-[12px] tabular-nums">
                  {showPathRanks ? (
                    <span className="text-slate-400">
                      BM25 {hit.ranks?.bm25 ?? '-'} / 向量 {hit.ranks?.vector ?? '-'}
                    </span>
                  ) : null}
                  {showMovement ? (
                    <span className={cn('font-medium', movement.className)}>{movement.text}</span>
                  ) : null}
                  <span className="text-slate-600">{hit.score.toFixed(4)}</span>
                </span>
              </div>
              <p className="ellipsis mt-1 text-[12px] text-slate-400">
                《{hit.docTitle}》{hit.section}
              </p>
              <p className="mt-1 line-clamp-2 text-[13px] leading-relaxed text-slate-700">
                {hit.text}
              </p>
            </button>
          </li>
        )
      })}
    </ul>
  )
}
