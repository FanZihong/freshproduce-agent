import { cn } from '@/utils/format'
import { useUiStore } from '@/stores/uiStore'

const VARIANT_CLASS = {
  success: 'border-brand-100 bg-brand-50 text-brand-700',
  danger: 'border-red-100 bg-red-50 text-red-600',
  warning: 'border-amber-100 bg-amber-50 text-amber-700',
  info: 'border-blue-100 bg-blue-50 text-blue-600',
}

export function Toaster() {
  const toasts = useUiStore((state) => state.toasts)
  const removeToast = useUiStore((state) => state.removeToast)

  if (toasts.length === 0) return null

  return (
    <div className="fixed right-5 top-5 z-[60] flex w-80 flex-col gap-2">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={cn(
            'fade-in cursor-pointer rounded-lg border px-3 py-2 shadow-sm',
            VARIANT_CLASS[toast.variant],
          )}
          onClick={() => removeToast(toast.id)}
        >
          <p className="text-sm font-medium">{toast.title}</p>
          {toast.description ? (
            <p className="mt-0.5 text-xs opacity-80">{toast.description}</p>
          ) : null}
        </div>
      ))}
    </div>
  )
}
