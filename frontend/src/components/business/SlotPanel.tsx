import type { SlotItem } from '@/types'
import { Tag } from '@/components/ui/Tag'

interface SlotPanelProps {
  slots: SlotItem[]
}

export function SlotPanel({ slots }: SlotPanelProps) {
  if (slots.length === 0) {
    return <p className="text-xs text-slate-400">当前会话暂无槽位信息</p>
  }

  return (
    <ul className="flex flex-col gap-2">
      {slots.map((slot) => (
        <li
          key={slot.key}
          className="flex items-center justify-between rounded-md border border-slate-100 px-3 py-2"
        >
          <span className="text-[13px] text-slate-600">
            {slot.label}
            {slot.required ? <span className="ml-1 text-red-500">必填</span> : null}
          </span>
          {slot.value ? (
            <span className="max-w-[60%] truncate text-[13px] text-slate-900" title={slot.value}>
              {slot.value}
            </span>
          ) : (
            <Tag variant="warning">待补充</Tag>
          )}
        </li>
      ))}
    </ul>
  )
}
