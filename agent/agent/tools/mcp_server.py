"""把四个工具暴露成 MCP server。

启动方式（stdio）：
    fastmcp run agent/tools/mcp_server.py

启动方式（HTTP，便于用 Inspector 调试）：
    fastmcp run agent/tools/mcp_server.py --transport http --port 8001
"""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from fastmcp import FastMCP  # noqa: E402

from agent.tools.orders import query_logistics as _query_logistics  # noqa: E402
from agent.tools.orders import query_order as _query_order  # noqa: E402
from agent.tools.policy import query_policy as _query_policy  # noqa: E402
from agent.tools.faq import search_faq as _search_faq  # noqa: E402
from agent.tools.tickets import create_ticket as _create_ticket  # noqa: E402

mcp = FastMCP(
    name="fresh-service-tools",
    instructions="生鲜售后 Agent 的工具集：查订单、查物流、查赔付规则、提交工单。",
)


@mcp.tool
def query_order(order_id: str) -> dict | None:
    """按订单号查询订单信息；订单不存在时返回 null。"""
    return _query_order(order_id)


@mcp.tool
def query_logistics(order_id: str) -> dict:
    """查询订单的物流状态；没有物流记录时返回 ok=false。"""
    return _query_logistics(order_id)


@mcp.tool
def query_policy(amount: float, ratio: float, category: str = "浆果类") -> dict:
    """按订单金额与坏果比例查询赔付档位；超出自动审核上限时 matched=false。"""
    return _query_policy(amount=amount, ratio=ratio, category=category)


@mcp.tool
def create_ticket(
    session_id: str,
    order_id: str,
    reason: str,
    evidence: str | None = None,
    refund_amount: float | None = None,
) -> dict:
    """提交售后工单并返回工单号、状态与创建时间。"""
    return _create_ticket(
        session_id=session_id,
        order_id=order_id,
        reason=reason,
        evidence=evidence,
        refund_amount=refund_amount,
    )


@mcp.tool
def search_faq(query: str, top_k: int = 3) -> dict:
    """检索售后知识库（政策、赔付规则、储存建议），返回带引用的候选。"""
    return _search_faq(query=query, top_k=top_k)


if __name__ == "__main__":
    mcp.run()
