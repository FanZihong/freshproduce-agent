"""LangGraph 工作流组装。

图结构：

    START → rails_input ─┬─(blocked)──────────────────→ rails_output
                         └→ intent ─┬→ skill → faq / fallback ─→ rails_execution
                                    ├→ order / claim / multi ────→ rails_execution
                                    ├→ clarify ─────────────────→ rails_output
                                    └→ handoff ─────────────────→ rails_output

    rails_execution ─┬→ handoff ─→ rails_output
                     └→ rails_output → respond → compress → END

`skill` 只挂在 faq / fallback 前面：先选技能、加载正文（按 token 预算截断），
再由 faq / fallback 决定怎么用。`SKILLS_ENABLED=off` 时这条边直接消失，
图和以前完全一样。

多轮上下文由 MemorySaver 按 thread_id 持久化：
每轮只把新的 user_input 传给图，槽位、工具结果、转人工标志都从 checkpoint 恢复。
"""

from __future__ import annotations

import sqlite3

from langgraph.graph import END, START, StateGraph

try:  # langgraph >= 0.3 的正式名称
    from langgraph.checkpoint.memory import InMemorySaver as _MemorySaver
except ImportError:  # 兼容旧版本的别名
    from langgraph.checkpoint.memory import MemorySaver as _MemorySaver

from agent.graph.nodes import (
    claim_node,
    compress_node,
    faq_node,
    fallback_node,
    handoff_node,
    intent_node,
    multi_node,
    order_node,
    rails_execution_node,
    rails_input_node,
    rails_output_node,
    respond_node,
    skill_node,
)
from agent.graph.router import (
    route_after_claim,
    route_after_execution,
    route_after_input,
    route_after_node,
    route_after_skill,
    route_with_skills,
)
from agent.graph.state import AgentState
from agent.db import DATA_DIR

CHECKPOINT_DB = DATA_DIR / "checkpoints.db"

# 主路由的返回值 -> 节点名
BRANCH_MAP = {
    "order": "order",
    "claim": "claim",
    "faq": "faq",
    "multi": "multi",
    # 澄清话术由 intent_node 写好，先过输出护栏再落成 AI 消息
    "clarify": "rails_output",
    "handoff": "handoff",
    "fallback": "fallback",
}

# 理赔节点执行后的返回值 -> 下一个节点（end 表示本轮结束）
AFTER_CLAIM_MAP = {
    "handoff": "handoff",
    "end": "rails_execution",
}

# 普通节点出口：工具降级时转入人工
AFTER_NODE_MAP = {
    "handoff": "handoff",
    "end": "rails_execution",
}


def build_graph(checkpointer: _MemorySaver | None = None):
    """组装并编译图；传入 checkpointer 可替换持久化实现。"""
    builder = StateGraph(AgentState)

    builder.add_node("intent", intent_node)
    builder.add_node("order", order_node)
    builder.add_node("claim", claim_node)
    builder.add_node("handoff", handoff_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("faq", faq_node)
    builder.add_node("multi", multi_node)
    builder.add_node("respond", respond_node)
    builder.add_node("compress", compress_node)
    builder.add_node("skill", skill_node)
    builder.add_node("rails_input", rails_input_node)
    builder.add_node("rails_execution", rails_execution_node)
    builder.add_node("rails_output", rails_output_node)

    builder.add_edge(START, "rails_input")
    builder.add_conditional_edges(
        "rails_input",
        route_after_input,
        {"intent": "intent", "rails_output": "rails_output"},
    )
    # FAQ / 兜底先过技能节点（SKILLS_ENABLED=off 时直接走原分支）
    builder.add_conditional_edges(
        "intent", route_with_skills, {**BRANCH_MAP, "skill": "skill"}
    )
    builder.add_conditional_edges("skill", route_after_skill, BRANCH_MAP)
    builder.add_conditional_edges("claim", route_after_claim, AFTER_CLAIM_MAP)
    builder.add_conditional_edges("order", route_after_node, AFTER_NODE_MAP)
    builder.add_conditional_edges("fallback", route_after_node, AFTER_NODE_MAP)
    builder.add_conditional_edges("faq", route_after_node, AFTER_NODE_MAP)
    builder.add_conditional_edges("multi", route_after_node, AFTER_NODE_MAP)
    builder.add_conditional_edges(
        "rails_execution",
        route_after_execution,
        {"handoff": "handoff", "rails_output": "rails_output"},
    )

    builder.add_edge("handoff", "rails_output")
    builder.add_edge("rails_output", "respond")
    builder.add_edge("respond", "compress")
    builder.add_edge("compress", END)

    return builder.compile(checkpointer=checkpointer or _MemorySaver())


def _sqlite_saver():
    """SqliteSaver：让 checkpoint 落到磁盘，进程重启后仍能恢复。"""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        return None

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    return SqliteSaver(conn)


_graphs: dict[str, object] = {}


def get_graph(persistent: bool = True):
    """进程内单例。

    persistent=True 时用 SqliteSaver（跨进程恢复），否则用内存 saver。
    未安装 langgraph-checkpoint-sqlite 时自动退回内存实现。
    """
    key = "persistent" if persistent else "memory"
    if key not in _graphs:
        checkpointer = _sqlite_saver() if persistent else None
        if persistent and checkpointer is None:
            print("[warn] 未安装 langgraph-checkpoint-sqlite，本次使用内存 checkpoint")
        _graphs[key] = build_graph(checkpointer)
    return _graphs[key]
