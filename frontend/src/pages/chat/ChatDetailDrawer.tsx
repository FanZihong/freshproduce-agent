import { useMemo } from 'react'
import { X } from 'lucide-react'
import { SlotPanel } from '@/components/business/SlotPanel'
import { ToolCallCard } from '@/components/business/ToolCallCard'
import { Tag } from '@/components/ui/Tag'
import { useSessionStore } from '@/stores/sessionStore'
import { useUiStore } from '@/stores/uiStore'
import type { DrawerTab } from '@/stores/uiStore'
import { cn } from '@/utils/format'

const TABS: Array<{ key: DrawerTab; label: string }> = [
  { key: 'state', label: '会话状态' },
  { key: 'tools', label: '工具调用' },
  { key: 'retrieval', label: '检索结果' },
]

export function ChatDetailDrawer() {
  const closeDrawer = useUiStore((state) => state.closeDrawer)
  const drawerTab = useUiStore((state) => state.drawerTab)
  const openDrawer = useUiStore((state) => state.openDrawer)
  const currentSessionId = useSessionStore((state) => state.currentSessionId)
  const messagesMap = useSessionStore((state) => state.messages)
  const messages = useMemo(
    () => (currentSessionId ? messagesMap[currentSessionId] ?? [] : []),
    [currentSessionId, messagesMap],
  )

  const latestAssistant = [...messages].reverse().find((message) => message.role === 'assistant')
  const slots = latestAssistant?.slots ?? []
  const toolCalls = latestAssistant?.toolCalls ?? []
  const retrieval = latestAssistant?.retrieval ?? []

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-slate-200 px-4">
        <span className="text-sm font-semibold text-slate-900">会话详情</span>
        <button
          type="button"
          onClick={closeDrawer}
          aria-label="关闭详情"
          className="focus-ring cursor-pointer rounded-md p-1 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
        >
          <X size={16} />
        </button>
      </header>

      <div className="flex shrink-0 gap-1 border-b border-slate-100 px-3 py-2">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            type="button"
            onClick={() => openDrawer(tab.key)}
            className={cn(
              'focus-ring cursor-pointer rounded-md px-2.5 py-1 text-[13px] transition-colors duration-150',
              drawerTab === tab.key
                ? 'bg-brand-50 font-medium text-brand-700'
                : 'text-slate-500 hover:bg-slate-100',
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-auto p-4">
        {!currentSessionId ? (
          <p className="text-xs text-slate-400">请选择一个会话</p>
        ) : drawerTab === 'state' ? (
          <div className="flex flex-col gap-3">
            {latestAssistant?.intent ? (
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-500">当前意图</span>
                <Tag variant="info" dot>
                  {latestAssistant.intent.name}
                </Tag>
              </div>
            ) : null}
            <SlotPanel slots={slots} />
          </div>
        ) : drawerTab === 'tools' ? (
          <div className="flex flex-col gap-2">
            {toolCalls.length === 0 ? (
              <p className="text-xs text-slate-400">本轮未触发工具调用</p>
            ) : (
              toolCalls.map((toolCall) => <ToolCallCard key={toolCall.id} toolCall={toolCall} />)
            )}
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {retrieval.length === 0 ? (
              <p className="text-xs text-slate-400">本轮未触发知识检索</p>
            ) : (
              retrieval.map((hit) => (
                <div key={hit.id} className="rounded-md border border-slate-100 px-3 py-2">
                  <p className="text-[13px] text-slate-700">{hit.title}</p>
                  <p className="mt-1 flex items-center justify-between text-xs text-slate-400">
                    <span>{hit.source}</span>
                    <span className="tabular-nums">score {hit.score.toFixed(2)}</span>
                  </p>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  )
}
