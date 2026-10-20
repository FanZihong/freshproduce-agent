"""Agent 图状态定义。

这是整张 LangGraph 图的"共享数据流"：每个节点只读取和修改它，
条件路由依据其中的字段决定下一步走哪个分支。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# 坏果理赔需要收集的槽位，按收集顺序排列
CLAIM_SLOTS: list[str] = ["order_id", "damage_ratio", "evidence", "refund_amount"]

# 理赔判定前必须齐全的三个槽位（赔付金额由系统计算，不需要用户提供）
REQUIRED_SLOTS: list[str] = ["order_id", "damage_ratio", "evidence"]

# 今天先支持的 4 类意图，明日再扩展为完整 10 类
SUPPORTED_INTENTS: frozenset[str] = frozenset(
    {
        # 体系里的 10 类（见 data/intent/taxonomy.yaml）
        "query_order",
        "urge_logistics",
        "bad_fruit_claim",
        "short_weight",
        "refund_consult",
        "invoice",
        "quality_qa",
        "complaint",
        "chitchat",
        "human_handoff",
        # 规则版分类器仍在使用的粗粒度标签
        "knowledge_qa",
        "fallback",
    }
)


class AgentState(TypedDict, total=False):
    """工作流各节点共享的状态。

    total=False 表示字段不必在开始时全部存在，各节点按需写入。

    字段说明：
    - messages: 对话历史。暂用 dict 列表；接入 LangGraph 后改为
      Annotated[list[AnyMessage], add_messages]，获得自动去重与合并的 reducer。
    - intent / confidence: 意图识别结果，路由的依据。
    - slots: 多轮槽位（订单号、坏果比例、凭证、赔付金额），槽位是否齐全决定继续追问还是走理赔判定。
    - handoff / handoff_summary: 转人工标志与交接摘要。
    - summary: 长对话的滚动摘要，供上下文压缩使用。
    - loop_count: 护栏计数，同一意图反复循环超过阈值时强制转人工。
    - active_flow: 当前未完成的流程（如 bad_fruit_claim），续接多轮对话时优先于本轮意图。
    - pending_slot: 正在追问的槽位，用于判断用户是否连续两轮都没有提供有效信息。
    - reply: 本轮节点给用户的回复文本。
    - handoff_reason: 转人工的具体原因（争议、信息不全等）。
    - intent_labels: 本轮识别出的完整标签序列（多意图时第 1 个是主意图）。
    - intent_secondary: 辅助意图；单意图时为空。
    - branches: 本次需要执行的分支列表（priority 只有一条，parallel 有两条）。
    - multi_branch_inputs: 并行策略下每条分支实际拿到的子句（子句切分的结果，供 trace 复盘）。
    - multi_strategy: 多意图路由策略（priority / parallel）。
    - intent_plan: 多意图路由计划的快照，供 trace 复盘（不做业务判断）。
    - out_of_domain: 域外判定结果（三路信号 + 规则否决）。
    - ood_reason: 域外判定的理由，写进 trace 便于排查误判。
    - decision: 置信度决策结果（accept / clarify / handoff）。
    - clarify_rounds: 连续澄清次数，复用 loop_count 的思路做护栏。
    - memory_hits: 本轮从长期事实记忆里召回到的条目（含分数），供 trace 与后续分支使用。
    - skill_name / skill_prompt: 本轮命中的技能与注入块（正文已按 token 预算截断）。
    - skill_reason / skill_score: 为什么选它，供 trace 复盘与调触发词。
    - rails: 本轮命中的护栏规则（输入/对话/执行/检索/输出），供 trace 复盘。
    - blocked: 输入护栏是否直接拦截了本轮（拦截后跳过意图识别）。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str
    user_input: str
    intent: str
    confidence: float
    slots: dict[str, str | float | None]
    tool_results: list[dict]
    handoff: bool
    handoff_summary: str
    summary: str
    loop_count: int
    active_flow: str
    pending_slot: str
    reply: str
    handoff_reason: str
    citations: list[dict]
    intent_labels: tuple[str, ...]
    intent_secondary: str
    branches: list[str]
    multi_branch_inputs: dict[str, str]
    multi_strategy: str
    intent_plan: dict
    out_of_domain: bool
    ood_reason: str
    decision: str
    clarify_rounds: int
    memory_hits: list[dict]
    skill_name: str
    skill_version: str
    skill_reason: str
    skill_score: float
    skill_tokens: int
    skill_prompt: str
    skill_truncated: bool
    skill_references: list[dict]
    rails: list[dict]
    blocked: bool
    tool_count: int


def first_missing_slot(slots: dict | None) -> str | None:
    """返回第一个还没填上的必填槽位，全齐时返回 None。"""
    current = slots or {}
    for key in REQUIRED_SLOTS:
        value = current.get(key)
        if value is None or value == "":
            return key
    return None


def claim_slots_complete(slots: dict | None) -> bool:
    """理赔所需的必填槽位是否已经齐全。"""
    return first_missing_slot(slots) is None
