"""初始化本地数据库：建表并写入模拟订单。

用法（在 agent/ 目录下执行）：
    python scripts/init_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DB_PATH, connect, init_db, seed_orders  # noqa: E402


def _count(conn, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    return int(row["n"])


def main() -> None:
    init_db()
    seed_orders()

    conn = connect()
    try:
        orders_n = _count(conn, "orders")
        tickets_n = _count(conn, "tickets")
    finally:
        conn.close()

    print("初始化完成")
    print(f"数据库文件: {DB_PATH}")
    print(f"orders 表: {orders_n} 条")
    print(f"tickets 表: {tickets_n} 条")


if __name__ == "__main__":
    main()
