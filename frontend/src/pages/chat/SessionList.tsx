import { useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Plus, Search } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Tag } from '@/components/ui/Tag'
import { SESSION_STATUS } from '@/constants/status'
import { ROUTES } from '@/constants/routes'
import { useSessionStore } from '@/stores/sessionStore'
import { useUiStore } from '@/stores/uiStore'
import { formatRelativeTime } from '@/utils/date'
import { cn } from '@/utils/format'

export function SessionList() {
  const navigate = useNavigate()
  const { sessionId } = useParams<{ sessionId: string }>()
  const sessions = useSessionStore((state) => state.sessions)
  const createSession = useSessionStore((state) => state.createSession)
  const pushToast = useUiStore((state) => state.pushToast)
  const [keyword, setKeyword] = useState('')

  const filtered = useMemo(() => {
    if (!keyword.trim()) return sessions
    return sessions.filter((session) => session.title.includes(keyword.trim()))
  }, [keyword, sessions])

  const handleCreate = async () => {
    const id = await createSession()
    pushToast({ title: '会话已创建', variant: 'success' })
    navigate(ROUTES.chatSession(id))
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-col gap-3 border-b border-slate-100 p-4">
        <Button variant="primary" icon={<Plus size={15} />} block onClick={handleCreate}>
          新建会话
        </Button>
        <Input
          placeholder="搜索会话"
          value={keyword}
          prefixIcon={<Search size={15} />}
          onChange={(event) => setKeyword(event.target.value)}
        />
      </div>

      <div className="flex-1 overflow-auto p-2">
        {filtered.map((session) => {
          const statusMeta = SESSION_STATUS[session.status]
          const active = sessionId === session.id
          return (
            <button
              key={session.id}
              type="button"
              onClick={() => navigate(ROUTES.chatSession(session.id))}
              className={cn(
                'focus-ring mb-1 w-full cursor-pointer rounded-md px-3 py-2.5 text-left transition-colors duration-150',
                active ? 'bg-brand-50' : 'hover:bg-slate-50',
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <span
                  className={cn(
                    'ellipsis text-[13px] font-medium',
                    active ? 'text-brand-700' : 'text-slate-800',
                  )}
                >
                  {session.title}
                </span>
                <span className="shrink-0 text-[11px] text-slate-400">
                  {formatRelativeTime(session.updatedAt)}
                </span>
              </div>
              <p className="ellipsis mt-1 text-xs text-slate-400">{session.lastMessage}</p>
              <div className="mt-1.5">
                <Tag variant={statusMeta.variant} dot>
                  {statusMeta.label}
                </Tag>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
