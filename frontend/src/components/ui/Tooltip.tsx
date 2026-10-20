import type { ReactNode } from 'react'
import { cn } from '@/utils/format'

interface TooltipProps {
  content: string
  children: ReactNode
  className?: string
}

export function Tooltip({ content, children, className }: TooltipProps) {
  return (
    <span className={cn('group relative inline-flex max-w-full', className)}>
      {children}
      <span
        role="tooltip"
        className="pointer-events-none absolute -top-8 left-0 z-30 hidden whitespace-nowrap rounded-md bg-slate-900 px-2 py-1 text-xs text-white group-hover:block"
      >
        {content}
      </span>
    </span>
  )
}
