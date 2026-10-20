import type { ReactNode } from 'react'
import { cn } from '@/utils/format'
import type { TagVariant } from '@/constants/status'

const VARIANT_CLASS: Record<TagVariant, string> = {
  neutral: 'bg-slate-100 text-slate-600',
  success: 'bg-brand-50 text-brand-700',
  danger: 'bg-red-50 text-red-600',
  warning: 'bg-amber-50 text-amber-700',
  info: 'bg-blue-50 text-blue-600',
}

const DOT_CLASS: Record<TagVariant, string> = {
  neutral: 'bg-slate-400',
  success: 'bg-brand-600',
  danger: 'bg-red-600',
  warning: 'bg-amber-500',
  info: 'bg-blue-600',
}

interface TagProps {
  variant?: TagVariant
  dot?: boolean
  children: ReactNode
  className?: string
}

export function Tag({ variant = 'neutral', dot, children, className }: TagProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-medium',
        VARIANT_CLASS[variant],
        className,
      )}
    >
      {dot ? <span className={cn('h-1.5 w-1.5 rounded-full', DOT_CLASS[variant])} /> : null}
      {children}
    </span>
  )
}
