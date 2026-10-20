import type { ReactNode } from 'react'
import { cn } from '@/utils/format'
import { Empty } from './Empty'

export interface Column<T> {
  key: string
  title: string
  width?: string
  align?: 'left' | 'center' | 'right'
  render?: (row: T) => ReactNode
}

interface TableProps<T> {
  columns: Column<T>[]
  rows: T[]
  rowKey: (row: T) => string
  onRowClick?: (row: T) => void
  emptyText?: string
  loading?: boolean
}

export function Table<T>({
  columns,
  rows,
  rowKey,
  onRowClick,
  emptyText = '暂无数据',
  loading,
}: TableProps<T>) {
  if (loading) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        {Array.from({ length: 4 }).map((_, index) => (
          <div key={index} className="mb-2 h-8 animate-pulse rounded bg-slate-100" />
        ))}
      </div>
    )
  }

  if (rows.length === 0) {
    return <Empty title={emptyText} description="调整筛选条件后重试" />
  }

  return (
    <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
      <div className="max-h-[560px] overflow-auto">
        <table className="w-full border-collapse text-sm">
          <thead className="sticky top-0 z-10 bg-slate-50">
            <tr>
              {columns.map((column) => (
                <th
                  key={column.key}
                  style={{ width: column.width }}
                  className={cn(
                    'border-b border-slate-200 px-4 py-2.5 text-xs font-medium text-slate-500',
                    column.align === 'right'
                      ? 'text-right'
                      : column.align === 'center'
                        ? 'text-center'
                        : 'text-left',
                  )}
                >
                  {column.title}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={rowKey(row)}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
                className={cn(
                  'border-b border-slate-100 last:border-b-0',
                  onRowClick && 'row-hover',
                )}
              >
                {columns.map((column) => (
                  <td
                    key={column.key}
                    className={cn(
                      'px-4 py-3 align-middle text-slate-700',
                      column.align === 'right'
                        ? 'text-right tabular-nums'
                        : column.align === 'center'
                          ? 'text-center'
                          : 'text-left',
                    )}
                  >
                    {column.render
                      ? column.render(row)
                      : String((row as unknown as Record<string, unknown>)[column.key] ?? '-')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
