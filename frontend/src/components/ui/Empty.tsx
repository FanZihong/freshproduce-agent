import type { ReactNode } from 'react'

interface EmptyProps {
  title: string
  description?: string
  action?: ReactNode
}

export function Empty({ title, description, action }: EmptyProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-slate-200 bg-white px-6 py-10 text-center">
      <p className="text-sm font-medium text-slate-700">{title}</p>
      {description ? <p className="text-xs text-slate-400">{description}</p> : null}
      {action ? <div className="mt-1">{action}</div> : null}
    </div>
  )
}
