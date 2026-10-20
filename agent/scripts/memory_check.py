"""Memory 后端自检：确认"当前到底在用什么存储"，并跑一遍读写往返。

为什么需要单独一个命令：`MEMORY_STORE=auto` 在 Redis 连不上时会回落 SQLite，
如果只看日志或 .env，很容易"以为在用 Redis，其实在写本地库"。这条命令把
真实后端、配置来源、往返结果一次打出来。

用法（在 agent/ 目录下执行）：
    python scripts/memory_check.py                      # 用当前配置（默认 SQLite）
    REDIS_URL=redis://localhost:6379/0 python scripts/memory_check.py   # 切 Redis
    MEMORY_STORE=redis REDIS_URL=... python scripts/memory_check.py     # 强要求 Redis

退出码：0 = 往返正常；1 = 读写失败。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.config import get_env  # noqa: E402
from agent.db import init_db  # noqa: E402
from agent.memory.store import get_store  # noqa: E402

FMT = "%Y-%m-%d %H:%M:%S"


def main() -> int:
    parser = argparse.ArgumentParser(description="Memory 后端自检")
    parser.add_argument("--session", default=None, help="自检用的会话 ID（默认按时间生成）")
    parser.add_argument("--keep", action="store_true", help="保留写入的记录，方便去库里翻")
    args = parser.parse_args()

    mode = (get_env("MEMORY_STORE", "auto") or "auto").strip().lower()
    redis_url = (get_env("REDIS_URL", "") or "").strip()
    print(f"MEMORY_STORE={mode}　REDIS_URL={'已配置' if redis_url else '未配置'}")

    try:
        store = get_store()
    except RuntimeError as exc:
        print(f"✗ 初始化失败：{exc}")
        return 1

    # SQLite 后端要先建表（Redis 不需要，表已存在时这步是幂等的）
    if store.backend == "sqlite":
        init_db()
    print(f"实际后端：{store.backend}")

    session_id = args.session or f"memory-check-{datetime.now():%H%M%S}"
    now = datetime.now().strftime(FMT)
    checks: list[tuple[str, bool, str]] = []

    try:
        store.set_activity(session_id, now, ttl_seconds=120)
        got = store.get_activity(session_id)
        checks.append(("短期活跃写读", got == now, str(got)))

        store.set_summary(session_id, "memory-check 摘要", (datetime.now() + timedelta(minutes=10)).strftime(FMT))
        loaded = store.get_summary(session_id, now)
        checks.append(("长期摘要写读", loaded == "memory-check 摘要", str(loaded)))

        expired_key = f"{session_id}-expired"
        store.set_summary(expired_key, "过期摘要", "2000-01-01 00:00:00")
        gone = store.get_summary(expired_key, now)
        checks.append(("过期摘要读即清", gone is None, str(gone)))
    except Exception as exc:  # noqa: BLE001 - 连不上就该如实报错
        print(f"✗ 往返失败：{type(exc).__name__}: {exc}")
        return 1

    for name, ok, value in checks:
        print(f"  {'✓' if ok else '✗'} {name}：{value}")

    if not args.keep:
        store.clear_activity(session_id)
        store.clear_summary(session_id)
        store.clear_summary(f"{session_id}-expired")
        print("已清理自检写入的记录（加 --keep 可保留）")

    if all(ok for _, ok, _ in checks):
        print(f"\n后端 {store.backend} 往返正常。")
        return 0
    print("\n有检查项未通过，见上面明细。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
