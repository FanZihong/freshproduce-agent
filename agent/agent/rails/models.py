"""护栏的数据模型：事件、规则、单次运行结果。

和 NeMo Guardrails 的对应关系：
- `RailEvent` 对应它的事件（UserMessage / BotMessage / ActionFinished …）；
- `RailRule` 对应它的 rail 规则（when 触发 + 动作）；
- `RailRunResult` 是一次事件循环的完整回放，直接可以写进 trace。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 事件类型：沿用项目里已有的数据流，不发明新名词
EVENT_TYPES: frozenset[str] = frozenset(
    {
        "user_message",   # 用户这一轮说了什么
        "intent",         # 意图/置信度/槽位等路由状态
        "retrieval",      # 检索结果（found / hits / 相关性）
        "tool_call",      # 工具调用前
        "tool_result",    # 工具调用后
        "reply",          # 准备发给用户的回复
        "rewrite",        # 兼容旧调用：rewrite 事件
        "handoff",        # 已决定转人工
    }
)

SCOPES: tuple[str, ...] = ("input", "dialog", "execution", "retrieval", "output")
ACTIONS: tuple[str, ...] = ("block", "rewrite", "handoff", "flag")


@dataclass(frozen=True)
class RailEvent:
    """一条待处理事件；`payload` 是只读约定，引擎会复制后改写。"""

    type: str
    payload: dict = field(default_factory=dict)
    turn: int = 0
    depth: int = 0          # 被 rewrite 的次数，用于防止规则自我循环


@dataclass(frozen=True)
class RailRule:
    """一条声明式规则；匹配逻辑见 `matcher.py`。"""

    id: str
    scope: str
    event: str
    priority: int
    action: str
    message: str = ""
    reason: str = ""
    keywords: tuple[str, ...] = ()
    regex: str = ""
    not_regex: str = ""
    conditions: tuple[dict, ...] = ()
    tags: tuple[str, ...] = ()
    min_score: float = 3.0
    enabled: bool = True
    source: str = ""

    def render(self, text: str) -> str:
        """rewrite 动作的模板：`{text}` 占位，没有占位就整段替换。"""
        return self.message.replace("{text}", text).strip()


@dataclass
class RailRunResult:
    """一次事件循环的结果；`events` 是处理过的事件序列，便于复盘。"""

    hits: list[dict] = field(default_factory=list)
    flags: list[dict] = field(default_factory=list)
    rewrites: list[dict] = field(default_factory=list)
    handoff: dict | None = None
    blocked: bool = False
    blocked_by: str = ""
    steps: int = 0
    stopped: str = "empty"     # empty / blocked / max_steps
    final_text: str = ""
    trace: list[dict] = field(default_factory=list)   # 每一步：事件 → 规则 → 动作

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "flags": self.flags,
            "rewrites": self.rewrites,
            "handoff": self.handoff,
            "blocked": self.blocked,
            "blocked_by": self.blocked_by,
            "steps": self.steps,
            "stopped": self.stopped,
            "trace": self.trace,
        }
