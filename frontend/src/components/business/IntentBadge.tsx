import type { IntentResult } from '@/types'
import { Tag } from '@/components/ui/Tag'
import { formatPercent } from '@/utils/format'

interface IntentBadgeProps {
  intent: IntentResult
  onClick?: () => void
}

export function IntentBadge({ intent, onClick }: IntentBadgeProps) {
  return (
    <span
      onClick={onClick}
      title={onClick ? '查看会话状态' : undefined}
      className={onClick ? 'cursor-pointer' : undefined}
    >
      <Tag variant="info" dot>
        {intent.name} {formatPercent(intent.confidence, 0)}
      </Tag>
    </span>
  )
}
