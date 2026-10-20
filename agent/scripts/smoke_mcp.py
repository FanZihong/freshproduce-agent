"""MCP 端到端 demo：列工具 → 直接调工具 → 走 MCP 跑完整理赔链路，带断言与退出码。

这条命令回答一个问题：**Agent 真的能通过 MCP 协议把工具用起来吗？**
不只是"server 起得来"，而是"图里的四个工具调用全部经过 MCP，链路照样办成事"。

用法（在 agent/ 目录下执行）：
    python scripts/smoke_mcp.py                  # 进程内 MCP（无需额外起服务，最快）
    python scripts/smoke_mcp.py --json demo.json # 顺便把检查结果写成 JSON

退出码：0 = 全部检查通过；1 = 有失败（可以放进 CI 或发布门禁）。

也可以手动起服务验证跨进程：
    fastmcp run agent/tools/mcp_server.py              # stdio
    fastmcp run agent/tools/mcp_server.py --transport http --port 8001
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import connect, init_db, seed_orders  # noqa: E402
from agent.graph.workflow import build_graph  # noqa: E402
from agent.tools.gateway import get_transport, set_transport  # noqa: E402
from agent.tools.mcp_client import SERVER_FILE, call_mcp_tool, list_tools  # noqa: E402

CLAIM_TURNS = ["草莓坏了一半，怎么赔", "2026093000123", "大概一半", "照片传不上来"]
SESSION_ID = "demo-mcp"


def section(title: str) -> None:
    print()
    print(f"---------- {title} ----------")


def ticket_count(session_id: str) -> int:
    conn = connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM tickets WHERE session_id = ?", (session_id,)
        ).fetchone()
        return int(row["n"])
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="MCP 端到端 demo")
    parser.add_argument("--json", type=Path, default=None, help="把检查结果写成 JSON")
    args = parser.parse_args()

    init_db()
    seed_orders()
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"  {'✓' if ok else '✗'} {name}" + (f"（{detail}）" if detail else ""))

    section("1. MCP server 暴露了哪些工具")
    tools = list_tools()
    names = sorted(tool["name"] for tool in tools)
    check("至少 4 个工具且包含 create_ticket", len(tools) >= 4 and "create_ticket" in names, "、".join(names))

    section("2. 通过 MCP 协议直接调工具")
    order = call_mcp_tool("query_order", {"order_id": "2026093000123"})
    check(
        "query_order 返回真实订单",
        isinstance(order, dict) and order.get("product") == "红颜草莓 500g",
        str(order)[:60],
    )
    missing = call_mcp_tool("query_order", {"order_id": "9999999999999"})
    check("查无此单返回 None（网关能判 not_found）", missing is None, repr(missing))

    section("3. 跨进程 stdio 调用（尽力而为，失败不算错）")
    try:
        stdio = call_mcp_tool("query_order", {"order_id": "2026093000123"}, target=SERVER_FILE)
        print(f"  [info] stdio 子进程调用：{'可用' if stdio else '无结果'}")
    except Exception as exc:  # noqa: BLE001 - 演示脚本：这条不通时给出替代方案
        print(f"  [info] stdio 传输不可用（{type(exc).__name__}: {exc}）")
        print("  可手动验证：fastmcp run agent/tools/mcp_server.py")

    section("4. Agent 走 MCP 跑通完整理赔链路")
    set_transport("mcp")
    try:
        print(f"  当前 transport：{get_transport()}")
        graph = build_graph()
        config = {"configurable": {"thread_id": SESSION_ID}}
        for turn in CLAIM_TURNS:
            graph.invoke({"user_input": turn, "session_id": SESSION_ID}, config=config)
        state = graph.get_state(config).values
    finally:
        set_transport("direct")

    transports = {item.get("transport") for item in (state.get("tool_results") or [])}
    reply = state.get("reply") or ""
    print(f"  Agent 回复：{reply[:80]}")
    check("理赔链路的工具调用全部经过 MCP", transports == {"mcp"}, str(transports))
    check("链路走完并建了工单", "工单" in reply, reply[:60])
    check("库里能查到这张工单", ticket_count(SESSION_ID) >= 1, f"{ticket_count(SESSION_ID)} 张")

    passed = sum(1 for item in checks if item["ok"])
    print(f"\n结果：{passed}/{len(checks)} 项通过")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "checks": checks,
                    "passed": passed == len(checks),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"结果 JSON：{args.json}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
