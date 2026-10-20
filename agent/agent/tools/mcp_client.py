"""MCP 客户端：让 Agent 通过 MCP 协议调用工具，而不是直接 import 函数。

默认使用进程内（in-memory）传输，便于命令行和测试直接调用；
也可以把 target 换成 mcp_server.py 的路径或 HTTP 地址，走真实 stdio / HTTP 传输。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from agent.config import get_float

SERVER_FILE = Path(__file__).resolve().parent / "mcp_server.py"
DEFAULT_TIMEOUT_SECONDS = 60.0


def _timeout() -> float:
    """单次 MCP 调用的上限：没有它，传输层一卡就是整条测试/请求挂死。"""
    return get_float("MCP_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)


def _default_target():
    """默认连本地 server 对象；需要跨进程时改用 SERVER_FILE。"""
    from agent.tools.mcp_server import mcp

    return mcp


def _normalize(data: Any) -> Any:
    """FastMCP 会把 None 结果包装成 {'result': None}，这里还原成 None，
    保证"订单查不到"这类空结果仍能被网关判定为 not_found。"""
    if isinstance(data, dict) and set(data.keys()) == {"result"}:
        return data["result"]
    return data


async def _list_tools_async(target: Any) -> list[dict]:
    from fastmcp import Client

    async with Client(target) as client:
        tools = await asyncio.wait_for(client.list_tools(), timeout=_timeout())
        return [
            {"name": tool.name, "description": (tool.description or "").strip()}
            for tool in tools
        ]


async def _call_async(target: Any, name: str, arguments: dict) -> Any:
    from fastmcp import Client

    async with Client(target) as client:
        result = await asyncio.wait_for(
            client.call_tool(name, arguments), timeout=_timeout()
        )
        data = getattr(result, "data", None)
        if data is None:
            data = getattr(result, "structured_content", None)
        if data is None:
            return result
        return _normalize(data)


def list_tools(target: Any | None = None) -> list[dict]:
    """列出 server 暴露的工具（名称 + 描述）。"""
    return asyncio.run(_list_tools_async(target if target is not None else _default_target()))


def call_mcp_tool(name: str, arguments: dict, target: Any | None = None) -> Any:
    """通过 MCP 协议调用一个工具。"""
    return asyncio.run(
        _call_async(target if target is not None else _default_target(), name, arguments)
    )


def make_mcp_callable(tool_name: str, target: Any | None = None) -> Callable[..., Any]:
    """包装成网关可直接调用的函数（关键字参数）。"""

    def _call(**kwargs: Any) -> Any:
        return call_mcp_tool(tool_name, kwargs, target)

    _call.__name__ = f"mcp:{tool_name}"
    return _call
