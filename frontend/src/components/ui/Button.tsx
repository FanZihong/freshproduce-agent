import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { cn } from '@/utils/format'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'
type Size = 'sm' | 'md'

const VARIANT_CLASS: Record<Variant, string> = {
  primary: 'border border-transparent bg-brand-600 text-white hover:bg-brand-700',
  secondary: 'border border-slate-200 bg-white text-slate-700 hover:bg-slate-50',
  ghost: 'border border-transparent bg-transparent text-slate-600 hover:bg-slate-100',
  danger: 'border border-transparent bg-red-600 text-white hover:bg-red-700',
}

const SIZE_CLASS: Record<Size, string> = {
  sm: 'h-8 px-3 text-[13px]',
  md: 'h-9 px-4 text-sm',
}

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
  icon?: ReactNode
  block?: boolean
}

export function Button({
  variant = 'secondary',
  size = 'md',
  icon,
  block,
  className,
  children,
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      type="button"
      disabled={disabled}
      className={cn(
        'focus-ring inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-colors duration-150',
        VARIANT_CLASS[variant],
        SIZE_CLASS[size],
        disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer',
        block && 'w-full',
        className,
      )}
      {...rest}
    >
      {icon}
      {children}
    </button>
  )
}
