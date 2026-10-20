import type { InputHTMLAttributes, ReactNode } from 'react'
import { cn } from '@/utils/format'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  prefixIcon?: ReactNode
  hint?: string
}

export function Input({ label, prefixIcon, hint, className, id, ...rest }: InputProps) {
  return (
    <div className="flex w-full flex-col gap-1.5">
      {label ? (
        <label htmlFor={id} className="text-[13px] font-medium text-slate-700">
          {label}
        </label>
      ) : null}
      <div className="relative flex items-center">
        {prefixIcon ? (
          <span className="pointer-events-none absolute left-3 flex text-slate-400">
            {prefixIcon}
          </span>
        ) : null}
        <input
          id={id}
          className={cn(
            'focus-ring h-9 w-full rounded-md border border-slate-200 bg-white text-sm text-slate-900 placeholder:text-slate-400',
            'transition-colors duration-150 hover:border-slate-300 focus:border-brand-600',
            prefixIcon ? 'pl-9 pr-3' : 'px-3',
            className,
          )}
          {...rest}
        />
      </div>
      {hint ? <span className="text-xs text-slate-400">{hint}</span> : null}
    </div>
  )
}
