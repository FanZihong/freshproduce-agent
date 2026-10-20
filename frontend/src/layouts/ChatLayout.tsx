import type { ReactNode } from 'react'
import { cn } from '@/utils/format'

interface ChatLayoutProps {
  sidebar: ReactNode
  children: ReactNode
  drawer?: ReactNode
  drawerOpen?: boolean
}

export function ChatLayout({ sidebar, children, drawer, drawerOpen }: ChatLayoutProps) {
  return (
    <div className="flex h-full min-h-0">
      <div className="flex w-64 shrink-0 flex-col border-r border-slate-200 bg-white">{sidebar}</div>
      <div className="flex min-w-0 flex-1 flex-col bg-white">{children}</div>
      <div
        className={cn(
          'shrink-0 overflow-hidden border-l border-slate-200 bg-white transition-all duration-200',
          drawerOpen ? 'w-80' : 'w-0',
        )}
      >
        {drawerOpen ? <div className="drawer-enter h-full w-80 overflow-auto">{drawer}</div> : null}
      </div>
    </div>
  )
}
