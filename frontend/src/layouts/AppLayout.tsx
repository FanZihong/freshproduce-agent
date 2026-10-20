import { NavLink, Outlet, useLocation } from 'react-router-dom'
import {
  Activity,
  BookOpen,
  ClipboardCheck,
  Cpu,
  LayoutDashboard,
  MessageSquare,
  Search,
  Ticket,
} from 'lucide-react'
import { ROUTES } from '@/constants/routes'
import { USE_MOCK } from '@/api/client'
import { cn } from '@/utils/format'

interface NavItem {
  to: string
  label: string
  icon: typeof MessageSquare
  group: string
}

const NAV_ITEMS: NavItem[] = [
  { to: ROUTES.chat, label: '对话体验台', icon: MessageSquare, group: '演示' },
  { to: ROUTES.overview, label: '概览', icon: LayoutDashboard, group: '运营' },
  { to: ROUTES.tickets, label: '工单中心', icon: Ticket, group: '运营' },
  { to: ROUTES.evals, label: '评测中心', icon: ClipboardCheck, group: '运营' },
  { to: ROUTES.traces, label: '监控中心', icon: Activity, group: '运营' },
  { to: ROUTES.knowledge, label: '知识库', icon: BookOpen, group: '运营' },
  { to: ROUTES.retrieval, label: '检索调试台', icon: Search, group: '运营' },
  { to: ROUTES.intent, label: '意图模型', icon: Cpu, group: '运营' },
]

const PAGE_TITLE: Array<{ match: string; title: string }> = [
  { match: '/chat', title: '对话体验台' },
  { match: '/overview', title: '概览' },
  { match: '/tickets', title: '工单中心' },
  { match: '/evals', title: '评测中心' },
  { match: '/traces', title: '监控中心' },
  { match: '/knowledge', title: '知识库' },
  { match: '/retrieval', title: '检索调试台' },
  { match: '/intent', title: '意图模型' },
]

export function AppLayout() {
  const location = useLocation()
  const currentTitle =
    PAGE_TITLE.find((item) => location.pathname.startsWith(item.match))?.title ?? '生鲜售后 Agent'
  const groups = ['演示', '运营']

  return (
    <div className="flex h-full w-full">
      <aside className="flex w-56 shrink-0 flex-col border-r border-slate-200 bg-white">
        <div className="flex h-14 items-center gap-2 px-4">
          <span className="flex h-7 w-7 items-center justify-center rounded-md bg-brand-600 text-sm font-semibold text-white">
            鲜
          </span>
          <div className="leading-tight">
            <p className="text-[13px] font-semibold text-slate-900">生鲜售后 Agent</p>
            <p className="text-[11px] text-slate-400">售后智能客服控制台</p>
          </div>
        </div>

        <nav className="flex flex-1 flex-col gap-4 overflow-auto px-3 py-2">
          {groups.map((group) => (
            <div key={group} className="flex flex-col gap-0.5">
              <p className="px-2 pb-1 text-[11px] font-medium uppercase tracking-wide text-slate-400">
                {group}
              </p>
              {NAV_ITEMS.filter((item) => item.group === group).map((item) => {
                const Icon = item.icon
                return (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    className={({ isActive }) =>
                      cn(
                        'focus-ring flex items-center gap-2 rounded-md px-2 py-2 text-[13px] transition-colors duration-150',
                        isActive
                          ? 'bg-brand-50 font-medium text-brand-700'
                          : 'text-slate-600 hover:bg-slate-100',
                      )
                    }
                  >
                    <Icon size={16} strokeWidth={1.8} />
                    {item.label}
                  </NavLink>
                )
              })}
            </div>
          ))}
        </nav>

        <div className="border-t border-slate-100 px-4 py-3">
          <p className="text-[11px] text-slate-400">
            {USE_MOCK ? '当前为 Mock 数据模式' : '已连接后端服务'}
          </p>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-slate-200 bg-white px-5">
          <h1 className="text-sm font-semibold text-slate-900">{currentTitle}</h1>
          <div className="flex items-center gap-3">
            <span className="rounded-md bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
              {USE_MOCK ? 'Mock' : 'Online'}
            </span>
            <div className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-slate-200 text-xs text-slate-600">
                运营
              </span>
              <span className="text-[13px] text-slate-600">售后运营</span>
            </div>
          </div>
        </header>

        <main className="min-h-0 flex-1 overflow-auto bg-slate-50">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
