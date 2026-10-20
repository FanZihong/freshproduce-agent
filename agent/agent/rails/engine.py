"""事件循环内核：取事件 → 匹配规则 → 执行动作 → 产生新事件 → 直到队列空。

三个保证：
1. **有限终止**：`max_steps` 卡死循环，rewrite 另外有深度上限；
2. **幂等**：同一条事件（seq）上同一条规则只执行一次；
3. **旁路**：规则出错只记录，不把对话带崩。
"""

from __future__ import annotations

from collections import deque

from agent.config import get_env
from agent.rails.matcher import select_rule
from agent.rails.models import RailEvent, RailRunResult
from agent.rails.registry import RailRegistry, get_registry

DEFAULT_MAX_STEPS = 32
DEFAULT_MAX_REWRITES = 3


def _max_steps() -> int:
    raw = get_env("RAILS_MAX_STEPS", "") or str(DEFAULT_MAX_STEPS)
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_MAX_STEPS
    return max(1, value)


def _max_rewrites() -> int:
    raw = get_env("RAILS_MAX_REWRITES", "") or str(DEFAULT_MAX_REWRITES)
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_MAX_REWRITES
    return max(1, value)


class RailEngine:
    def __init__(
        self,
        registry: RailRegistry | None = None,
        *,
        max_steps: int | None = None,
        max_rewrites: int | None = None,
    ) -> None:
        self.registry = registry or get_registry()
        self.max_steps = _max_steps() if max_steps is None else max(1, int(max_steps))
        self.max_rewrites = _max_rewrites() if max_rewrites is None else max(1, int(max_rewrites))

    def run(self, events: list[RailEvent], *, text_key: str = "text") -> RailRunResult:
        result = RailRunResult()
        queue: deque[tuple[int, RailEvent]] = deque()
        next_seq = 0
        for event in events:
            queue.append((next_seq, event))
            next_seq += 1
        applied: set[tuple[int, str]] = set()

        while queue:
            if result.steps >= self.max_steps:
                result.stopped = "max_steps"
                break
            seq, event = queue.popleft()
            result.steps += 1

            try:
                selected = select_rule(self.registry.rules(event.type), event)
            except Exception as exc:  # noqa: BLE001 - 护栏不能带崩对话
                result.flags.append({"event": event.type, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if selected is None:
                result.trace.append(
                    {"step": result.steps, "event": event.type, "rule": "", "action": "", "queue": len(queue)}
                )
                continue
            rule, score, matched = selected
            result.trace.append(
                {
                    "step": result.steps,
                    "event": event.type,
                    "rule": rule.id,
                    "action": rule.action,
                    "score": round(score, 2),
                    "queue": len(queue),
                }
            )
            key = (seq, rule.id)
            if key in applied:
                continue
            applied.add(key)

            hit = {
                "rule": rule.id,
                "scope": rule.scope,
                "action": rule.action,
                "event": event.type,
                "score": round(score, 2),
                "matched": matched,
                "why": matched,
            }
            result.hits.append(hit)

            try:
                if rule.action == "flag":
                    result.flags.append(hit)
                    continue
                if rule.action == "block":
                    result.blocked = True
                    result.blocked_by = rule.id
                    result.stopped = "blocked"
                    hit["message"] = rule.message
                    break
                if rule.action == "handoff":
                    result.handoff = {
                        "rule": rule.id,
                        "message": rule.message,
                        "reason": rule.reason,
                    }
                    hit["message"] = rule.message
                    queue.append(
                        (
                            next_seq,
                            RailEvent(
                                "handoff",
                                {"rule": rule.id, "message": rule.message},
                                turn=event.turn,
                                depth=event.depth + 1,
                            ),
                        )
                    )
                    next_seq += 1
                    continue
                if rule.action == "rewrite":
                    text = str(event.payload.get(text_key) or "")
                    new_text = rule.render(text)
                    if not new_text or new_text == text:
                        continue
                    if event.depth >= self.max_rewrites:
                        result.flags.append({**hit, "reason": "rewrite_depth"})
                        continue
                    result.rewrites.append({"rule": rule.id, "before": text, "after": new_text})
                    result.final_text = new_text
                    payload = {**event.payload, text_key: new_text}
                    queue.append(
                        (
                            next_seq,
                            RailEvent(event.type, payload, turn=event.turn, depth=event.depth + 1),
                        )
                    )
                    next_seq += 1
            except Exception as exc:  # noqa: BLE001 - 单条动作失败不能带崩整轮
                result.flags.append(
                    {"rule": rule.id, "error": f"{type(exc).__name__}: {exc}"}
                )

        return result
