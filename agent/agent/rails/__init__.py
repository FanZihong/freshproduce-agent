"""规则护栏：事件队列 + 规则匹配 + 四种动作（block/rewrite/handoff/flag）。"""

from agent.rails.engine import RailEngine
from agent.rails.matcher import check_condition, score_rule, select_rule
from agent.rails.models import ACTIONS, EVENT_TYPES, SCOPES, RailEvent, RailRule, RailRunResult
from agent.rails.registry import (
    RailConfigError,
    RailRegistry,
    default_rails_dir,
    get_registry,
    load_registry,
    reset_registry,
)

__all__ = [
    "ACTIONS",
    "EVENT_TYPES",
    "SCOPES",
    "RailConfigError",
    "RailEngine",
    "RailEvent",
    "RailRegistry",
    "RailRule",
    "RailRunResult",
    "check_condition",
    "default_rails_dir",
    "get_registry",
    "load_registry",
    "reset_registry",
    "score_rule",
    "select_rule",
]
