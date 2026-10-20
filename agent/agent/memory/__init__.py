"""记忆管理：短期压缩/闲置超时 + 长期摘要（TTL、脱敏）。

存储后端见 agent.memory.store（MemoryStore 协议）：
默认 SQLite 零依赖；配置 REDIS_URL 后可切 Redis，两者行为一致。
"""

from agent.memory.long_term import load_summary, mask_pii, purge_expired, save_summary
from agent.memory.facts import list_facts, purge_expired_facts
from agent.memory.short_term import (
    IDLE_TIMEOUT_MINUTES,
    apply_idle_reset,
    compress_messages,
    idle_minutes,
    should_reset,
    touch,
)
from agent.memory.store import MemoryStore, get_store, reset_store
from agent.memory.writer import RememberStats, remember
from agent.memory.search import search_facts
from agent.memory.summarizer import (
    reset_stats as reset_summary_stats,
    set_completer as set_summary_completer,
    stats as summary_stats,
    summarize_dialogue,
)

__all__ = [
    "load_summary",
    "save_summary",
    "purge_expired",
    "mask_pii",
    "compress_messages",
    "touch",
    "idle_minutes",
    "should_reset",
    "apply_idle_reset",
    "IDLE_TIMEOUT_MINUTES",
    "get_store",
    "reset_store",
    "MemoryStore",
    "summarize_dialogue",
    "summary_stats",
    "reset_summary_stats",
    "set_summary_completer",
    "remember",
    "RememberStats",
    "search_facts",
    "list_facts",
    "purge_expired_facts",
]
