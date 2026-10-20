"""工具入参模型：参数校验的第一道关。

校验失败不抛给上层，由网关统一转成结构化错误与降级话术。
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

ORDER_ID_PATTERN = re.compile(r"^\d{10,20}$")


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class QueryOrderInput(ToolInput):
    order_id: str = Field(min_length=10, max_length=20, description="订单号，10-20 位数字")

    @field_validator("order_id")
    @classmethod
    def _check_order_id(cls, value: str) -> str:
        if not ORDER_ID_PATTERN.match(value):
            raise ValueError("订单号必须是 10-20 位数字")
        return value


class QueryLogisticsInput(QueryOrderInput):
    pass


class QueryPolicyInput(ToolInput):
    amount: float = Field(ge=0, description="订单实付金额")
    ratio: float = Field(ge=0, le=1, description="坏果比例，0-1 的小数")
    category: str = Field(default="浆果类", min_length=1, description="商品类目或商品名")


class CreateTicketInput(ToolInput):
    session_id: str = Field(min_length=1, description="会话 ID")
    order_id: str = Field(min_length=10, max_length=20, description="订单号")
    reason: str = Field(min_length=1, description="工单原因")
    evidence: str | None = Field(default=None, description="凭证描述")
    refund_amount: float | None = Field(default=None, ge=0, description="赔付金额")

    @field_validator("order_id")
    @classmethod
    def _check_order_id(cls, value: str) -> str:
        if not ORDER_ID_PATTERN.match(value):
            raise ValueError("订单号必须是 10-20 位数字")
        return value


class SearchFaqInput(ToolInput):
    query: str = Field(min_length=2, max_length=200, description="要检索的问题")
    context: dict | None = Field(default=None, description="多轮上下文，如上一轮问题与商品")
    top_k: int = Field(default=3, ge=1, le=10, description="返回的候选条数")
