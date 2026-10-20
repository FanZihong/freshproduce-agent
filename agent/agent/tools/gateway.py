"""工具网关：校验 -> 重试 -> 结构化结果 -> 降级决策。

节点不直接调用工具函数，统一走 call_tool()，让参数校验、重试、降级只有一处实现。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from agent.tools.degradation import HANDOFF, Degradation, resolve_degradation
from agent.tools.orders import query_logistics as raw_query_logistics
from agent.tools.orders import query_order as raw_query_order
from agent.tools.policy import query_policy as raw_query_policy
from agent.tools.faq import search_faq as raw_search_faq
from agent.tools.retry import ToolExecutionError, run_with_retry
from agent.tools.schemas import (
    CreateTicketInput,
    QueryLogisticsInput,
    QueryOrderInput,
    QueryPolicyInput,
    SearchFaqInput,
)
from agent.tools.tickets import create_ticket as raw_create_ticket

VALIDATION = "validation"
NOT_FOUND = "not_found"
ERROR = "error"
OK = "ok"

# 工具调用方式：direct = 直接函数调用，mcp = 走 MCP 协议
_TRANSPORT = "direct"


def set_transport(mode: str) -> None:
    """切换工具调用方式，供命令行与测试使用。"""
    global _TRANSPORT
    if mode not in {"direct", "mcp"}:
        raise ValueError("transport 只能是 direct 或 mcp")
    _TRANSPORT = mode


def get_transport() -> str:
    return _TRANSPORT


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的治理配置。

    `retry_safe` 决定网关**会不会重试**：只有只读工具、或已经做了幂等的写工具
    才能标 True。默认 False——新加一个写工具忘了标，就不会被盲目重试，
    避免"重试一次多写一条"这种事故。
    """

    schema: type[BaseModel]
    func: Callable[..., Any]
    attempts: int = 3
    base_delay: float = 0.2
    timeout: float | None = None
    retry_safe: bool = False


TOOL_REGISTRY: dict[str, ToolSpec] = {
    # 只读工具：重试天然安全
    "query_order": ToolSpec(QueryOrderInput, raw_query_order, retry_safe=True),
    "query_logistics": ToolSpec(QueryLogisticsInput, raw_query_logistics, retry_safe=True),
    "query_policy": ToolSpec(QueryPolicyInput, raw_query_policy, retry_safe=True),
    # 写工具：create_ticket 靠"会话+订单+原因"的幂等键做到重复调用不重复建单，
    # 所以可以标成 retry_safe；没做幂等的写工具不要标。
    "create_ticket": ToolSpec(CreateTicketInput, raw_create_ticket, retry_safe=True),
    "search_faq": ToolSpec(
        SearchFaqInput, raw_search_faq, attempts=2, base_delay=0.1, retry_safe=True
    ),
}


@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: Any = None
    kind: str | None = None
    message: str = ""
    action: str = OK
    reply: str | None = None
    attempts: int = 0
    duration_ms: int = 0
    degradation: Degradation | None = None
    transport: str = "direct"


def _first_error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first.get("loc", ()))
    return f"{field}: {first.get('msg', '参数不合法')}"


def _failure(
    tool: str,
    kind: str,
    message: str,
    started: float,
    attempts: int = 0,
) -> ToolResult:
    degradation = resolve_degradation(tool, kind)
    return ToolResult(
        tool=tool,
        ok=False,
        kind=kind,
        message=message,
        action=degradation.action,
        reply=degradation.reply,
        attempts=attempts,
        duration_ms=int((time.perf_counter() - started) * 1000),
        degradation=degradation,
        transport=get_transport(),
    )


def call_tool(name: str, **kwargs: Any) -> ToolResult:
    """统一入口：参数校验、重试、结构化结果与降级决策。"""
    started = time.perf_counter()
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return _failure(name, ERROR, f"未注册的工具：{name}", started)

    try:
        payload = spec.schema(**kwargs)
    except ValidationError as exc:
        return _failure(name, VALIDATION, _first_error(exc), started)

    transport = get_transport()
    func = spec.func
    if transport == "mcp":
        from agent.tools.mcp_client import make_mcp_callable

        func = make_mcp_callable(name)

    try:
        outcome = run_with_retry(
            func,
            **payload.model_dump(),
            # 非重试安全的工具只跑一次：宁可报错降级，也不要重复产生副作用
            attempts=spec.attempts if spec.retry_safe else 1,
            base_delay=spec.base_delay,
            timeout=spec.timeout,
        )
    except ToolExecutionError as exc:
        return _failure(name, ERROR, str(exc), started, attempts=exc.attempts)

    data = outcome.value
    duration_ms = int((time.perf_counter() - started) * 1000)

    if data is None:
        return _failure(name, NOT_FOUND, f"{name} 没有查到数据", started, attempts=outcome.attempts)
    if isinstance(data, dict) and data.get("matched") is False:
        reason = str(data.get("reason") or "规则未覆盖该情形")
        return _failure(name, NOT_FOUND, reason, started, attempts=outcome.attempts)
    if isinstance(data, dict) and data.get("ok") is False:
        return _failure(
            name,
            NOT_FOUND,
            str(data.get("message") or "没有查到记录"),
            started,
            attempts=outcome.attempts,
        )

    # 执行护栏：工具"成功"之后再校验一次结果（金额超限、工单号格式异常）
    if isinstance(data, dict):
        try:
            from agent.rails.policy import tool_hit

            hit = tool_hit(name, data)
        except Exception as exc:  # noqa: BLE001 - 护栏失败不能影响工具结果
            print(f"[warn] 执行护栏失败，按工具成功处理：{type(exc).__name__}: {exc}")
            hit = None
        if hit:
            degradation = Degradation(
                HANDOFF,
                str(hit.get("reason") or "执行护栏拦截"),
                str(hit.get("message") or "这一步需要人工确认，已为您转接人工客服。"),
            )
            return ToolResult(
                tool=name,
                ok=False,
                kind="rail_block",
                message=degradation.reason,
                action=HANDOFF,
                reply=degradation.reply,
                attempts=outcome.attempts,
                duration_ms=duration_ms,
                degradation=degradation,
                transport=transport,
            )

    return ToolResult(
        tool=name,
        ok=True,
        data=data,
        action=OK,
        attempts=outcome.attempts,
        duration_ms=duration_ms,
        transport=transport,
    )
