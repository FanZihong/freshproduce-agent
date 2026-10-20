"""Memory 压缩消融：规则折叠 vs LLM 摘要。

做法：把对话黄金集里的场景**前面垫几轮闲聊**，把消息数推过压缩阈值（默认 8 条），
再跑完整图；同一批场景分别用 `MEMORY_SUMMARY=rule` 和 `MEMORY_SUMMARY=llm` 跑一遍，
看"压缩发生之后的几轮"有没有变差：

- 判据通过率（复用 eval_dialogues.evaluate_checks，场景本身的确定性判据）
- 工具调用次数 / 成功率
- 转人工率
- 摘要平均长度、LLM 压缩调用次数与成本

用法（在 agent/ 目录下执行）：
    python scripts/ablate_memory.py                       # 两个变体，默认垫 5 轮
    python scripts/ablate_memory.py --padding 6 --limit 8
    python scripts/ablate_memory.py --fake-llm            # 不调模型，只验证流水线

注意：LLM 变体需要 `LLM_API_KEY`（和改写共用）；没有 key 时脚本会明确报"LLM 未启用"，
而不是悄悄拿规则版的结果冒充。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.memory.summarizer import reset_stats, set_completer, stats  # noqa: E402
from scripts.eval_dialogues import evaluate_checks, load_jsonl  # noqa: E402

DEFAULT_DATASET = AGENT_ROOT / "eval" / "datasets" / "dialogues_golden.jsonl"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "memory_summary.md"
DEFAULT_JSON = AGENT_ROOT / "eval" / "reports" / "memory_summary.json"

# 垫场闲聊：走兜底分支、不建工单不转人工，只负责把消息数推过压缩阈值
PADDING_TURNS: tuple[str, ...] = (
    "你好",
    "在吗",
    "我想问下售后",
    "稍等我看看订单",
    "还在吗",
    "麻烦帮我看下",
)

FAKE_SUMMARY = "（假摘要）用户在咨询售后，涉及订单 2026093000123，诉求与凭证待确认。"


def build_padded_row(row: dict, padding: int) -> dict:
    """在场景前面垫几轮闲聊；id 加后缀，避免和正式评测的 session 撞车。"""
    pads = [PADDING_TURNS[index % len(PADDING_TURNS)] for index in range(padding)]
    return {
        **row,
        "id": f"mem-{row['id']}",
        "turns": [{"user": text} for text in pads] + list(row.get("turns") or []),
        "padding": padding,
        "scenario": f"{row.get('scenario') or row['id']}（垫 {padding} 轮）",
    }


def _tool_stats(tool_results: list[dict]) -> dict:
    return {
        "calls": len(tool_results),
        "success": sum(1 for item in tool_results if item.get("ok")),
    }


def run_row(graph, row: dict, *, mode: str) -> dict:
    """跑一段垫过场的对话，收集判据、工具与转人工事实。"""
    session_id = f"ablate-{mode}-{row['id']}"
    config = {"configurable": {"thread_id": session_id}}
    state: dict = {}
    branches: set[str] = set()
    for turn in row["turns"]:
        state = graph.invoke(
            {"user_input": turn["user"], "session_id": session_id}, config=config
        )
        branches.update(state.get("branches") or [])

    tool_results = list(state.get("tool_results") or [])
    facts = {
        "last_reply": state.get("reply") or "",
        "tools": [entry.get("name") for entry in tool_results if entry.get("name")],
        "handoff": bool(state.get("handoff")),
        "handoff_summary": state.get("handoff_summary") or "",
        "branches": sorted(branches),
    }
    checks = evaluate_checks(row, facts)
    return {
        "id": row["id"],
        "scenario": row.get("scenario"),
        "checks": checks,
        "passed": bool(checks) and all(checks.values()),
        "tools": _tool_stats(tool_results),
        "handoff": facts["handoff"],
        "summary": state.get("summary") or "",
        "summary_len": len(state.get("summary") or ""),
        "last_reply": facts["last_reply"],
    }


def aggregate(records: list[dict], mode: str, summary_meta: dict) -> dict:
    calls = sum(item["tools"]["calls"] for item in records)
    success = sum(item["tools"]["success"] for item in records)
    total = len(records) or 1
    return {
        "mode": mode,
        "dialogues": len(records),
        "pass_rate": round(sum(1 for item in records if item["passed"]) / total, 4),
        "tool_calls": calls,
        "tool_success_rate": round(success / calls, 4) if calls else 0.0,
        "handoff_rate": round(sum(1 for item in records if item["handoff"]) / total, 4),
        "avg_summary_len": round(
            sum(item["summary_len"] for item in records) / total, 1
        ),
        "llm": summary_meta,
    }


def recommendation(rule: dict, llm: dict) -> str:
    """给出"默认用哪个"的建议；打平时选规则（省时省钱）。"""
    if llm["llm"].get("calls", 0) == 0:
        return "LLM 变体没有真正调用模型，本次对比无效"
    llm_key = (llm["pass_rate"], llm["tool_success_rate"], -llm["handoff_rate"])
    rule_key = (rule["pass_rate"], rule["tool_success_rate"], -rule["handoff_rate"])
    if llm_key > rule_key:
        return "LLM 摘要更好：可以通过 MEMORY_SUMMARY=llm 启用（注意成本与延迟）"
    if (llm["pass_rate"], llm["tool_success_rate"]) < (
        rule["pass_rate"],
        rule["tool_success_rate"],
    ):
        return "LLM 摘要更差或没有收益：保持 rule"
    return "两者打平：保持 rule（同样的结果，规则版零成本、可复现）"


def write_report(
    path: Path, *, rows: list[dict], variants: dict, padding: int, fake_llm: bool = False
) -> None:
    rule, llm = variants["rule"], variants["llm"]
    lines = [
        "# Memory 压缩消融：规则折叠 vs LLM 摘要",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 场景：{len(rows)} 段，每段前面垫 {padding} 轮闲聊（把消息数推过压缩阈值）",
        "- 压缩阈值：超过 8 条消息折叠、保留最近 4 条（两种模式一致）",
        "",
        (
            "> ⚠️ 本次是 `--fake-llm` 演练：LLM 变体用的是固定假摘要，"
            "数字只能验证流水线，不能当结论。\n"
            if fake_llm
            else ""
        ),
        "## 一、对比",
        "",
        "| 指标 | 规则折叠 | LLM 摘要 |",
        "| --- | --- | --- |",
        f"| 判据通过率 | {rule['pass_rate']:.4f} | {llm['pass_rate']:.4f} |",
        f"| 工具调用次数 | {rule['tool_calls']} | {llm['tool_calls']} |",
        f"| 工具成功率 | {rule['tool_success_rate']:.4f} | {llm['tool_success_rate']:.4f} |",
        f"| 转人工率 | {rule['handoff_rate']:.4f} | {llm['handoff_rate']:.4f} |",
        f"| 摘要平均长度 | {rule['avg_summary_len']:.1f} 字 | {llm['avg_summary_len']:.1f} 字 |",
        f"| 压缩调用次数 | 0（本地） | {llm['llm'].get('calls', 0)} |",
        f"| 压缩耗时合计 | 0 ms | {llm['llm'].get('latency_ms', 0):.0f} ms |",
        f"| 压缩成本 | ¥0 | ¥{llm['llm'].get('cost', 0.0):.6f} |",
        "",
        "## 二、结论",
        "",
        recommendation(rule, llm),
        "",
        "## 三、怎么复现",
        "",
        "```bash",
        "python scripts/ablate_memory.py            # 真跑（LLM 变体需要 LLM_API_KEY）",
        "python scripts/ablate_memory.py --fake-llm # 只验证流水线，不调模型",
        "```",
        "",
        "> 提醒：这批场景只有几段、垫场也是模板化的，结论只说明「压缩会不会伤到工具链」，",
        "> 不代表 LLM 摘要在所有长对话上都更好。默认仍是 rule，想开再开。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="规则折叠 vs LLM 摘要的小消融")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ids", nargs="*", default=None, help="只跑指定场景 id")
    parser.add_argument("--limit", type=int, default=6, help="跑几段场景（默认 6）")
    parser.add_argument("--padding", type=int, default=5, help="前面垫几轮闲聊")
    parser.add_argument("--backend", default="rule", help="意图后端（透传给图）")
    parser.add_argument("--fake-llm", action="store_true", help="用假摘要验证流水线，不调模型")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.dataset)
    if args.ids:
        wanted = set(args.ids)
        rows = [row for row in rows if row["id"] in wanted]
    rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"没有可用场景：{args.dataset}")

    os.environ["INTENT_BACKEND"] = args.backend
    if args.fake_llm:
        set_completer(
            lambda messages: (FAKE_SUMMARY, {"prompt_tokens": 220, "completion_tokens": 40})
        )

    from agent.graph.workflow import get_graph

    graph = get_graph(persistent=False)  # 消融用独立会话，不写 checkpoint
    variants: dict[str, dict] = {}
    details: dict[str, list[dict]] = {}
    for mode in ("rule", "llm"):
        os.environ["MEMORY_SUMMARY"] = mode
        reset_stats()
        padded = [build_padded_row(row, args.padding) for row in rows]
        print(f"\n[{mode}] 跑 {len(padded)} 段（每段垫 {args.padding} 轮）…")
        records = []
        for row in padded:
            record = run_row(graph, row, mode=mode)
            records.append(record)
            print(
                f"  {record['id']}: 判据{'✓' if record['passed'] else '✗'} "
                f"工具 {record['tools']['success']}/{record['tools']['calls']} "
                f"转人工={record['handoff']} 摘要 {record['summary_len']} 字"
            )
        variants[mode] = aggregate(records, mode, stats() if mode == "llm" else {})
        details[mode] = records

    if variants["llm"]["llm"].get("calls", 0) == 0 and not args.fake_llm:
        print(
            "\n[!] LLM 变体一次模型都没调（没有可用的 LLM_API_KEY，或 MEMORY_SUMMARY 没生效）。\n"
            "    这份对比等同于「规则 vs 规则」，不能拿来下结论。"
        )

    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "padding": args.padding,
        "scenarios": [row["id"] for row in rows],
        "variants": variants,
        "details": details,
        "recommendation": recommendation(variants["rule"], variants["llm"]),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_report:
        write_report(
            args.report,
            rows=rows,
            variants=variants,
            padding=args.padding,
            fake_llm=args.fake_llm,
        )
        print(f"\n报告：{args.report}")
    print(f"指标 JSON：{args.json}")
    print(f"\n结论：{payload['recommendation']}")
    if args.fake_llm:
        print("（--fake-llm 演练：LLM 变体用的是固定假摘要，这行结论不代表真实效果）")
    ok = variants["llm"]["llm"].get("calls", 0) > 0 or args.fake_llm
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
