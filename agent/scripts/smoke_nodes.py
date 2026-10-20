"""单独验证第二段的节点与工具（不需要安装 LangGraph）。

用法（在 agent/ 目录下执行）：
    python scripts/smoke_nodes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import connect  # noqa: E402
from agent.graph.nodes import (  # noqa: E402
    claim_node,
    fallback_node,
    handoff_node,
    intent_node,
    order_node,
)
from agent.graph.router import route_after_intent  # noqa: E402
from agent.graph.state import AgentState  # noqa: E402
from agent.tools.orders import query_logistics, query_order  # noqa: E402
from agent.tools.policy import query_policy  # noqa: E402


def section(title: str) -> None:
    print()
    print(f"---------- {title} ----------")


def new_state(session_id: str) -> AgentState:
    return {
        "session_id": session_id,
        "slots": {},
        "tool_results": [],
        "loop_count": 0,
        "handoff": False,
    }


def dispatch(state: AgentState) -> None:
    """按路由结果调用对应节点，模拟图的一次执行。"""
    state.update(intent_node(state))
    route = route_after_intent(state)
    print(f"  意图={state['intent']}({state['confidence']}) 路由={route}")

    if route == "claim":
        state.update(claim_node(state))
    elif route == "order":
        state.update(order_node(state))
    elif route == "handoff":
        state.update(handoff_node(state))
    else:
        state.update(fallback_node(state))
    print(f"  Agent：{state.get('reply', '')}")


def run_turns(session_id: str, turns: list[str]) -> AgentState:
    state = new_state(session_id)
    for turn in turns:
        state["user_input"] = turn
        print(f"用户：{turn}")
        dispatch(state)
        print(f"  槽位：{state.get('slots')}")
    return state


def main() -> None:
    section("1. 意图识别（规则版，4 类）")
    demo_texts = [
        "我的订单到哪了",
        "草莓坏了一半，怎么赔",
        "我要转人工",
        "今天天气怎么样",
    ]
    for text in demo_texts:
        state = new_state("demo-intent")
        state["user_input"] = text
        state.update(intent_node(state))
        print(f"{text} -> {state['intent']} ({state['confidence']})")

    section("2. 工具直调")
    order = query_order("2026093000123")
    print("query_order:", order)
    print("query_logistics:", query_logistics("2026093000123"))
    print("query_policy(25.80, 0.5):", query_policy(amount=25.80, ratio=0.5))

    section("3. 理赔多轮（4 轮填满槽位并建工单）")
    run_turns(
        "demo-claim",
        [
            "草莓坏了一半，怎么赔",
            "2026093000123",
            "大概一半",
            "照片传不上来，用文字描述行吗",
        ],
    )

    section("4. 争议场景转人工")
    state = run_turns("demo-handoff", ["草莓坏了一半", "2026093000123", "这个不行，我要全退"])
    print("交接摘要:", state.get("handoff_summary"))

    section("5. 查单分支（先追问订单号，再回答）")
    run_turns("demo-order", ["我的订单到哪了", "2026093000123"])

    section("6. 域外兜底")
    run_turns("demo-fallback", ["今天天气怎么样"])

    section("7. 工单落库校验")
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT ticket_id, order_id, refund_amount, status FROM tickets ORDER BY created_at"
        ).fetchall()
    finally:
        conn.close()
    for row in rows:
        print(dict(row))
    if not rows:
        print("tickets 表为空（先执行 python scripts/init_db.py 初始化）")


if __name__ == "__main__":
    main()
