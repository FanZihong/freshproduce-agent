"""记忆维护：清理过期的事实与长期摘要（TTL 语义和 store 对齐）。

两类记忆的过期口径：
- **摘要**（`user_memory`）：存的是 `expires_at`，读时比较、`purge_summaries` 批量删；
- **事实**（`memory_facts`）：没有单独的过期列，按"最近一次被召回 / 更新 + TTL"现算，
  读时过滤 + 本脚本批量删（顺带删向量索引，避免留下幽灵向量）。

为什么不在每次召回时顺手清理：召回是热路径，删除要写库、还要扫索引；
放到启动时/定时任务里，成本可控，也不影响单轮延迟。

用法（在 agent/ 目录下执行）：
    python scripts/memory_maintenance.py                 # 真删
    python scripts/memory_maintenance.py --dry-run       # 只看会删哪些
    python scripts/memory_maintenance.py --ttl-days 45   # 临时改口径

crontab 示例（每天凌晨 3 点）：
    0 3 * * * cd /path/to/agent && .venv/bin/python scripts/memory_maintenance.py >> logs/memory.log 2>&1
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.memory import facts as facts_store  # noqa: E402
from agent.memory.long_term import purge_expired  # noqa: E402
from agent.memory.store import get_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="清理过期记忆")
    parser.add_argument("--ttl-days", type=float, default=None, help="覆盖 MEMORY_FACT_TTL_DAYS")
    parser.add_argument("--dry-run", action="store_true", help="只报告，不删除")
    args = parser.parse_args()

    ttl = facts_store.resolve_ttl_days(args.ttl_days)
    now = datetime.now()
    print(f"时间：{now:%Y-%m-%d %H:%M:%S}")
    try:
        print(f"Memory 后端：{get_store().backend}")
    except Exception as exc:  # noqa: BLE001 - 存储不可用也要把事实清理跑完
        print(f"Memory 后端：不可用（{type(exc).__name__}: {exc}）")
    print(f"事实 TTL：{ttl:g} 天" if ttl > 0 else "事实 TTL：0（永不过期）")

    expired = facts_store.list_expired_ids(ttl_days=args.ttl_days, now=now)
    print(f"过期事实：{len(expired)} 条")
    for fact_id in expired[:10]:
        item = facts_store.get_fact(fact_id)
        if item is None:
            continue
        status = facts_store.describe_fact(item, now=now)
        print(
            f"  - {fact_id[:8]}｜{item['session_id']}｜{item['fact'][:32]}…"
            f"｜{status['age_days']:g} 天未用"
        )
    if len(expired) > 10:
        print(f"  ……其余 {len(expired) - 10} 条省略")

    if args.dry_run:
        print("dry-run：未删除任何数据")
        return 0

    removed = facts_store.purge_expired_facts(ttl_days=args.ttl_days, now=now)
    summaries = purge_expired()
    print(f"已删除事实：{len(removed)} 条（含向量索引记录）")
    print(f"已删除过期摘要：{summaries} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
