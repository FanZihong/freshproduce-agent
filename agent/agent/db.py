"""SQLite 连接与表结构。

订单与工单是售后场景的业务数据，先落在本地 SQLite 中；
后续的工具节点（查订单、提交工单）直接复用这里的连接与路径。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = AGENT_ROOT / "data"
DB_PATH = DATA_DIR / "after_sales.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id    TEXT PRIMARY KEY,
    phone       TEXT NOT NULL,
    product     TEXT NOT NULL,
    quantity    TEXT NOT NULL,
    amount      REAL NOT NULL,
    status      TEXT NOT NULL,
    signed_at   TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id      TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    order_id       TEXT NOT NULL,
    reason         TEXT NOT NULL,
    evidence       TEXT,
    refund_amount  REAL,
    status         TEXT NOT NULL DEFAULT 'pending',
    created_at     TEXT NOT NULL,
    idempotency_key TEXT
);

CREATE TABLE IF NOT EXISTS user_memory (
    session_id  TEXT PRIMARY KEY,
    summary     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_activity (
    session_id     TEXT PRIMARY KEY,
    last_active_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'active',
    last_message TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    message_id  TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    intent      TEXT,
    confidence  REAL,
    slots       TEXT,
    tool_calls  TEXT,
    trace_id    TEXT,
    created_at  TEXT NOT NULL
);

-- 长期事实记忆：一行一条"抽取出来的事实"，不是原始对话
CREATE TABLE IF NOT EXISTS memory_facts (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    fact             TEXT NOT NULL,
    category         TEXT NOT NULL DEFAULT 'general',
    entities         TEXT NOT NULL DEFAULT '[]',
    importance       REAL NOT NULL DEFAULT 0.5,
    content_hash     TEXT NOT NULL,
    source           TEXT NOT NULL DEFAULT 'llm',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    last_accessed_at TEXT
);
"""

SEED_ORDERS: list[dict] = [
    {
        "order_id": "2026093000123",
        "phone": "138****1234",
        "product": "红颜草莓 500g",
        "quantity": "1",
        "amount": 25.80,
        "status": "已签收",
        "signed_at": "2026-09-30 09:40",
        "created_at": "2026-09-29 18:20",
    },
    {
        "order_id": "2026092800451",
        "phone": "139****5678",
        "product": "智利车厘子 1kg",
        "quantity": "1",
        "amount": 88.00,
        "status": "运输中",
        "signed_at": None,
        "created_at": "2026-09-28 09:05",
    },
    {
        "order_id": "2026092700882",
        "phone": "137****9012",
        "product": "蓝莓 250g",
        "quantity": "2",
        "amount": 39.90,
        "status": "已签收",
        "signed_at": "2026-09-28 16:05",
        "created_at": "2026-09-27 20:14",
    },
    {
        "order_id": "2026092600315",
        "phone": "136****3456",
        "product": "牛油果 4 个装",
        "quantity": "1",
        "amount": 45.60,
        "status": "已签收",
        "signed_at": "2026-09-27 11:20",
        "created_at": "2026-09-26 13:42",
    },
    {
        "order_id": "2026092500107",
        "phone": "135****7890",
        "product": "锦绣黄桃 1kg",
        "quantity": "1",
        "amount": 32.00,
        "status": "已发货",
        "signed_at": None,
        "created_at": "2026-09-25 10:08",
    },
]


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """返回开启 row_factory 的连接，保证查询结果可按列名访问。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str = DB_PATH) -> None:
    """建表（幂等，可重复执行），并把老库的 tickets 表补到最新结构。"""
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        ensure_ticket_schema(conn)
        ensure_fact_schema(conn)
        conn.commit()
    finally:
        conn.close()


# 建单幂等键的唯一索引：同一条"会话 + 订单 + 原因"只允许一张工单
TICKET_IDEMPOTENCY_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_tickets_idempotency "
    "ON tickets(idempotency_key)"
)


def ensure_ticket_schema(conn: sqlite3.Connection) -> None:
    """老库补列 + 建唯一索引；对新库是幂等的。

    为什么要单拎出来：`CREATE TABLE IF NOT EXISTS` 不会给已存在的表加列，
    所以升级时必须在建表之后补 `ALTER TABLE`，否则建单幂等会静默失效。
    """
    # PRAGMA 的列名固定在第 2 列；不用 row["name"]，这样裸连接（没设 row_factory）也能调
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tickets)")}
    if not columns:
        return  # 表还不存在：调用方应先 init_db
    if "idempotency_key" not in columns:
        conn.execute("ALTER TABLE tickets ADD COLUMN idempotency_key TEXT")
    conn.execute(TICKET_IDEMPOTENCY_INDEX)


def ensure_fact_schema(conn: sqlite3.Connection) -> None:
    """事实记忆表的索引（表由 SCHEMA 建；老库升级时这里补齐索引）。

    唯一索引 `(session_id, content_hash)` 是"哈希去重"的数据库兜底：
    应用层已经查过一遍，但并发写同一个事实时靠它保证只留一行。
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memory_facts)")}
    if not columns:
        return
    # 老库补列：CREATE TABLE IF NOT EXISTS 不会给已存在的表加字段，
    # 而老化/过期都依赖 last_accessed_at，缺了它召回会直接报错。
    if "last_accessed_at" not in columns:
        conn.execute("ALTER TABLE memory_facts ADD COLUMN last_accessed_at TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_facts_session ON memory_facts(session_id)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_facts_hash "
        "ON memory_facts(session_id, content_hash)"
    )
    conn.commit()


def seed_orders(db_path: Path | str = DB_PATH) -> None:
    """写入模拟订单，重复执行时按主键覆盖。"""
    conn = connect(db_path)
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO orders
                (order_id, phone, product, quantity, amount, status, signed_at, created_at)
            VALUES
                (:order_id, :phone, :product, :quantity, :amount, :status, :signed_at, :created_at)
            """,
            SEED_ORDERS,
        )
        conn.commit()
    finally:
        conn.close()
