import { cn } from '@/utils/format'

interface MetricCardProps {
  label: string
  value: string
  target?: string
  delta?: number
  hint?: string
  onClick?: () => void
}

export function MetricCard({ label, value, target, delta, hint, onClick }: MetricCardProps) {
  const positive = (delta ?? 0) >= 0

  return (
    <div
      onClick={onClick}
      className={cn(
        'rounded-lg border border-slate-200 bg-white p-4 shadow-sm transition-shadow duration-150',
        onClick && 'cursor-pointer hover:shadow-md',
      )}
    >
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-1 flex items-baseline gap-2">
        <span className="tabular-nums text-xl font-semibold text-slate-900">{value}</span>
        {typeof delta === 'number' ? (
          <span
            className={cn(
              'tabular-nums text-xs font-medium',
              positive ? 'text-brand-600' : 'text-red-600',
            )}
          >
            {positive ? '+' : ''}
            {delta}
          </span>
        ) : null}
      </p>
      {target ? <p className="mt-1 text-xs text-slate-400">目标 {target}</p> : null}
      {hint ? <p className="mt-1 text-xs text-slate-400">{hint}</p> : null}
    </div>
  )
}
