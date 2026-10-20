"""命令行多轮对话：把组合好的 LangGraph 工作流跑起来。

用法（在 agent/ 目录下执行）：
    python scripts/run_cli.py                     # 交互模式，连续聊天
    python scripts/run_cli.py --demo              # 依次跑三个验收用例
    python scripts/run_cli.py --turns "草莓坏了一半|2026093000123|大概一半"
    python scripts/run_cli.py --session demo-1    # 指定会话 ID
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402

from agent.db import connect, init_db, seed_orders  # noqa: E402
from agent.graph.workflow import get_graph  # noqa: E402
from agent.memory.short_term import should_reset, touch  # noqa: E402
from agent.session_lock import SessionBusy, session_lock  # noqa: E402
from agent.trace import append_turn, build_turn_record, save_turn_to_db, trace_path  # noqa: E402
from agent.tools.gateway import get_transport, set_transport  # noqa: E402

console = Console()

NODE_LABEL = {
    "rails_input": "输入护栏",
    "intent": "意图识别",
    "skill": "技能选择",
    "order": "查单分支",
    "claim": "理赔分支",
    "handoff": "转人工分支",
    "fallback": "兜底分支",
    "rails_execution": "执行护栏",
    "rails_output": "输出护栏",
}

# 三个验收用例
DEMOS: list[tuple[str, list[str]]] = [
    ("验收 1：订单查询", ["我的订单到哪了", "2026093000123"]),
    (
        "验收 2：坏果理赔全流程",
        ["草莓坏了一半，怎么赔", "2026093000123", "大概一半", "照片传不上来，用文字描述行吗"],
    ),
    ("验收 3：争议转人工", ["草莓坏了一半", "2026093000123", "这个不行，我要全退"]),
]


def ensure_db() -> None:
    """确保表存在且有模拟订单；幂等，可反复执行。"""
    init_db()
    conn = connect()
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM orders").fetchone()
        empty = int(row["n"]) == 0
    finally:
        conn.close()
    if empty:
        seed_orders()
        console.print("[yellow]检测到 orders 为空，已写入模拟订单。[/yellow]")


def render_step(node: str, update: dict) -> None:
    """把一个节点的执行结果渲染成一块面板。"""
    label = NODE_LABEL.get(node, node)
    table = Table(show_header=False, box=None, padding=(0, 1))

    if node == "intent":
        table.add_row("意图", f"{update.get('intent')}（{update.get('confidence')}）")
        if update.get("handoff_reason"):
            table.add_row("争议", str(update["handoff_reason"]))
        if update.get("summary"):
            memory = str(update["summary"]).replace("\n", " ")
            table.add_row("记忆", memory[:60] + ("..." if len(memory) > 60 else ""))
    if update.get("slots"):
        table.add_row("槽位", str(update["slots"]))
    if update.get("tool_results"):
        names = [item.get("name", "?") for item in update["tool_results"]]
        table.add_row("工具", ", ".join(names))
    if update.get("pending_slot"):
        table.add_row("待补槽位", str(update["pending_slot"]))
    if update.get("skill_name"):
        table.add_row(
            "技能",
            f"{update['skill_name']} v{update.get('skill_version', '')}"
            f"（{update.get('skill_tokens', 0)} tokens）",
        )
    if update.get("rails"):
        table.add_row(
            "护栏", ", ".join(str(item.get("rule") or "?") for item in update["rails"])
        )
    if update.get("handoff_summary"):
        table.add_row("交接摘要", str(update["handoff_summary"]))

    console.print(Panel(table, title=f"步骤：{label}", border_style="cyan", expand=False))


def run_turn(
    graph,
    config: dict,
    text: str,
    *,
    turn_index: int = 1,
    tool_offset: int = 0,
    trace_dir: Path | None = None,
    trace_enabled: bool = True,
    backend: str = "",
    transport: str = "",
) -> dict:
    """执行一轮对话，返回本轮的最终状态，并把这一轮写进 trace。"""
    session_id = config.get("configurable", {}).get("thread_id", "")
    if session_id:
        if should_reset(session_id):
            console.print("[yellow]会话闲置超过 30 分钟，短期槽位已重置。[/yellow]")
        touch(session_id)

    console.print(f"\n[bold]用户：[/bold]{text}")
    # 同一会话串行化：CLI 虽是单线程，但后台可能还有别的进程/线程在跑同一个 session
    busy = False
    try:
        with session_lock(session_id or "cli-session"):
            for chunk in graph.stream(
                {"user_input": text, "session_id": session_id},
                config=config,
                stream_mode="updates",
            ):
                for node, update in chunk.items():
                    if isinstance(update, dict):
                        render_step(node, update)
            state = graph.get_state(config).values
    except SessionBusy:
        busy = True
        state = graph.get_state(config).values
        console.print("[yellow]上一条消息还在处理，请稍候再发。[/yellow]")
    if state.get("reply"):
        console.print(Panel(state["reply"], title="Agent", border_style="green", expand=False))

    # 落 trace：一轮一行 JSONL + 写进 sessions/messages 表（失败只告警，不影响对话）
    if trace_enabled and session_id and not busy:
        record = build_turn_record(
            state,
            session_id=session_id,
            user_input=text,
            turn=turn_index,
            backend=backend,
            transport=transport,
            tool_offset=tool_offset,
        )
        append_turn(record, directory=trace_dir)
        save_turn_to_db(record)
    return state


def run_session(
    graph,
    session_id: str,
    turns: list[str],
    *,
    trace_dir: Path | None = None,
    trace_enabled: bool = True,
    backend: str = "",
    transport: str = "",
) -> dict:
    """按给定台词跑一个会话。"""
    config = {"configurable": {"thread_id": session_id}}
    state: dict = {}
    tool_offset = 0
    for index, text in enumerate(turns, start=1):
        state = run_turn(
            graph,
            config,
            text,
            turn_index=index,
            tool_offset=tool_offset,
            trace_dir=trace_dir,
            trace_enabled=trace_enabled,
            backend=backend,
            transport=transport,
        )
        # 工具记录在状态里是累加的，下一轮只记新增的那几条
        tool_offset = len(state.get("tool_results") or [])
        if state.get("handoff"):
            console.print("[yellow]已转人工，本会话结束。[/yellow]")
            break
    return state


def run_interactive(
    graph,
    session_id: str,
    *,
    trace_dir: Path | None = None,
    trace_enabled: bool = True,
    backend: str = "",
    transport: str = "",
) -> None:
    config = {"configurable": {"thread_id": session_id}}
    console.print("输入问题开始对话；输入 exit / quit / 退出 结束。")
    turn_index = 0
    tool_offset = 0
    while True:
        try:
            text = input("用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit", "退出"}:
            break
        turn_index += 1
        state = run_turn(
            graph,
            config,
            text,
            turn_index=turn_index,
            tool_offset=tool_offset,
            trace_dir=trace_dir,
            trace_enabled=trace_enabled,
            backend=backend,
            transport=transport,
        )
        tool_offset = len(state.get("tool_results") or [])
        if state.get("handoff"):
            console.print("[yellow]已转人工，本会话结束。[/yellow]")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="生鲜售后 Agent 命令行")
    parser.add_argument("--session", default=None, help="会话 ID（默认按时间生成）")
    parser.add_argument("--turns", default=None, help="一次性跑多轮，用 | 分隔")
    parser.add_argument("--demo", action="store_true", help="依次运行三个验收用例")
    parser.add_argument("--no-trace", action="store_true", help="不写 trace（调试用）")
    parser.add_argument("--trace-dir", type=Path, default=None, help="trace 目录（默认 data/trace）")
    parser.add_argument(
        "--transport",
        choices=["direct", "mcp"],
        default="direct",
        help="工具调用方式：direct 直连函数，mcp 走 MCP 协议",
    )
    args = parser.parse_args()

    ensure_db()
    set_transport(args.transport)
    graph = get_graph(persistent=True)
    console.print(f"工具调用方式：{get_transport()}")

    # 意图后端预热：local 后端第一次分类要 import torch/peft（drvfs 上可能几十秒），
    # 放在启动阶段完成，用户的第一句话就不会"卡住"。
    from agent.intent.classifier import get_backend, get_classifier

    backend = get_backend()
    if backend != "rule":
        console.print(f"[dim]意图后端：{backend}；正在预热（首次加载较慢，请稍等）…[/dim]")
        classifier = get_classifier()
        preload = getattr(classifier, "preload", None)
        if preload is not None and not preload():
            console.print("[yellow]本地模型预热失败，本次回落规则版。[/yellow]")
        elif getattr(classifier, "name", "") == "rule-based-v1":
            console.print("[yellow]没有可用的模型 adapter，本次按规则版运行。[/yellow]")
        else:
            console.print(f"[dim]预热完成：{classifier.name}[/dim]")

    # 实际使用的分类器名字写进 trace：换了后端之后能一眼看出这批数据是哪个版本产生的
    classifier_name = get_classifier().name
    trace_kwargs = {
        "trace_dir": args.trace_dir,
        "trace_enabled": not args.no_trace,
        "backend": classifier_name,
        "transport": args.transport,
    }
    if not args.no_trace:
        console.print(
            f"[dim]trace 落盘：{trace_path(args.trace_dir)}（一轮一行 JSONL，"
            "同时写 sessions/messages 表）[/dim]"
        )

    if args.demo:
        for index, (title, turns) in enumerate(DEMOS, start=1):
            console.print()
            console.print(Rule(title))
            run_session(graph, f"demo-{index}-{datetime.now():%H%M%S}", turns, **trace_kwargs)
        return

    session_id = args.session or f"cli-{datetime.now():%H%M%S}"
    console.print(f"会话 ID：{session_id}")

    if args.turns:
        run_session(
            graph,
            session_id,
            [turn for turn in args.turns.split("|") if turn.strip()],
            **trace_kwargs,
        )
        return

    run_interactive(graph, session_id, **trace_kwargs)


if __name__ == "__main__":
    main()
