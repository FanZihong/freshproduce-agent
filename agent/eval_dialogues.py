"""端到端对话评测：跑完对话黄金集，算工具成功率 / 转人工率 / 目标解决率。

这是评测层里唯一"从用户视角看"的评测：前面几层的指标（意图 F1、recall@k、域外召回）
都是模块级的，只有这里能回答"这段对话到底有没有把事办成"。

三个指标的分工：
- **工具成功率 / 降级率**：从每轮 `state["tool_results"]` 聚合，纯计数，不依赖模型；
- **转人工率**：分两类看——该转的转了吗（召回）、不该转的转了吗（误转）；
- **目标解决率（规则初判）**：用数据里写死的确定性判据判定（有没有建工单、有没有引用、
  有没有交接摘要、赔付金额对不对），它是第三段 LLM-as-judge 的参照物。

用法（在 agent/ 目录下执行）：
    python scripts/eval_dialogues.py                     # 规则版后端，秒级，开发用
    python scripts/eval_dialogues.py --backend local     # 本地 v1，约 3 分钟（真实数字）
    python scripts/eval_dialogues.py --strategy parallel --backend local
    python scripts/eval_dialogues.py --only dg-06 dg-07  # 只跑指定用例，调试用

环境不匹配的用例会被**跳过并说明原因**（例如需要 local 后端的多意图用例，用规则版跑没有意义），
跳过的不计入分母。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DATA_DIR  # noqa: E402

DEFAULT_DATASET = AGENT_ROOT / "eval" / "datasets" / "dialogues_golden.jsonl"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "dialogues.md"
DEFAULT_JSON = AGENT_ROOT / "eval" / "reports" / "dialogues.json"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------- 纯函数：跳过判定、判据求值、聚合（都可单测） ----------------


def skip_reason(row: dict, backend: str, strategy: str) -> str | None:
    """用例的环境要求与本次运行不匹配时，返回跳过原因。"""
    required = row.get("requires_env") or {}
    actual = {"INTENT_BACKEND": backend, "INTENT_MULTI_STRATEGY": strategy}
    mismatched = [
        f"{key}={value}（本次 {actual.get(key)}）"
        for key, value in required.items()
        if actual.get(key) != value
    ]
    if mismatched:
        return "需要 " + "、".join(mismatched)
    return None


def evaluate_checks(row: dict, facts: dict) -> dict:
    """按 check 里的确定性判据逐项判定，返回每项是否通过。

    facts 是跑完这段对话后从图状态里取出来的事实：last_reply / tools / handoff /
    handoff_summary / branches。
    """
    check = row.get("check") or {}
    results: dict[str, bool | None] = {}

    expected_tools = set(check.get("tools") or [])
    actual_tools = set(facts.get("tools") or [])
    if expected_tools:
        results["tools"] = expected_tools <= actual_tools
        facts["missing_tools"] = sorted(expected_tools - actual_tools)

    last_reply = facts.get("last_reply") or ""
    if check.get("reply_contains"):
        missing = [text for text in check["reply_contains"] if text not in last_reply]
        results["reply_contains"] = not missing
        facts["missing_in_reply"] = missing
    if check.get("reply_contains_any"):
        results["reply_contains_any"] = any(
            text in last_reply for text in check["reply_contains_any"]
        )

    if check.get("handoff_summary_contains"):
        summary = facts.get("handoff_summary") or ""
        missing = [text for text in check["handoff_summary_contains"] if text not in summary]
        results["handoff_summary_contains"] = not missing
        facts["missing_in_summary"] = missing

    if check.get("branches"):
        expected_branches = set(check["branches"])
        results["branches"] = expected_branches <= set(facts.get("branches") or [])

    return results


def dialogue_outcome(row: dict, facts: dict) -> dict:
    """把一条对话的判定结果算出来：规则初判的"是否解决" + 转人工是否正确。"""
    checks = evaluate_checks(row, facts)
    resolved = all(checks.values()) if checks else False
    expected_handoff = bool(row.get("expected_handoff"))
    actual_handoff = bool(facts.get("handoff"))
    return {
        "checks": checks,
        "resolved_predicted": resolved,
        "resolved_expected": bool(row.get("expected_goal_resolved")),
        "resolved_match": resolved == bool(row.get("expected_goal_resolved")),
        "handoff_expected": expected_handoff,
        "handoff_actual": actual_handoff,
        "handoff_ok": expected_handoff == actual_handoff,
    }


def aggregate_tools(tool_results: list[dict]) -> dict:
    """聚合工具调用情况：成功 / 重试后成功 / 降级（追问或转人工）/ 失败。"""
    stats = {
        "calls": 0,
        "success": 0,
        "retried_success": 0,
        "degraded_ask_again": 0,
        "degraded_handoff": 0,
        "failed": 0,
        "by_tool": Counter(),
    }
    for entry in tool_results or []:
        stats["calls"] += 1
        name = entry.get("name") or "unknown"
        stats["by_tool"][name] += 1
        attempts = int(entry.get("attempts") or 1)
        if entry.get("ok"):
            stats["success"] += 1
            if attempts > 1:
                stats["retried_success"] += 1
            continue
        action = entry.get("action")
        if action == "ask_again":
            stats["degraded_ask_again"] += 1
        elif action == "handoff":
            stats["degraded_handoff"] += 1
        else:
            stats["failed"] += 1
    calls = stats["calls"]
    stats["success_rate"] = round(stats["success"] / calls, 4) if calls else 0.0
    stats["degraded"] = stats["degraded_ask_again"] + stats["degraded_handoff"]
    stats["degrade_rate"] = round(stats["degraded"] / calls, 4) if calls else 0.0
    return stats


def summarize(records: list[dict]) -> dict:
    """按"预期解决的算解决率、预期不解决的算误判"两个方向统计。"""
    should_resolve = [r for r in records if r["resolved_expected"]]
    should_not = [r for r in records if not r["resolved_expected"]]
    should_handoff = [r for r in records if r["handoff_expected"]]
    should_not_handoff = [r for r in records if not r["handoff_expected"]]

    def rate(hit: int, total: int) -> float:
        return round(hit / total, 4) if total else 0.0

    return {
        "dialogues": len(records),
        "turns": sum(r["turns"] for r in records),
        # 目标解决率：该解决的有没有真解决
        "resolution_rate": rate(sum(1 for r in should_resolve if r["resolved_predicted"]), len(should_resolve)),
        "resolution_total": len(should_resolve),
        "over_claimed_rate": rate(
            sum(1 for r in should_not if r["resolved_predicted"]), len(should_not)
        ),
        "over_claimed_total": len(should_not),
        # 转人工：该转的转了吗 / 不该转的转了吗
        "handoff_recall": rate(sum(1 for r in should_handoff if r["handoff_actual"]), len(should_handoff)),
        "handoff_total": len(should_handoff),
        "handoff_false_rate": rate(
            sum(1 for r in should_not_handoff if r["handoff_actual"]), len(should_not_handoff)
        ),
        "handoff_false_total": len(should_not_handoff),
        "handoff_accuracy": rate(sum(1 for r in records if r["handoff_ok"]), len(records)),
        "check_accuracy": rate(sum(1 for r in records if r["resolved_match"]), len(records)),
    }


def group_stats(records: list[dict], key: str) -> list[tuple[str, dict]]:
    """按 chain 或 category 分组，各自算一份指标。"""
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record.get(key) or "-", []).append(record)
    return sorted(
        ((name, summarize(items)) for name, items in groups.items()),
        key=lambda item: item[0],
    )


# ---------------- 跑图（需要 langgraph，本地沙箱跑不了，交给用户环境） ----------------


def apply_env(backend: str, strategy: str, local_dir: str | None) -> None:
    """把本次运行的配置写进环境变量；切换后端后要清掉决策层缓存。"""
    os.environ["INTENT_BACKEND"] = backend
    os.environ["INTENT_MULTI_STRATEGY"] = strategy
    if local_dir:
        os.environ["INTENT_LOCAL_DIR"] = local_dir
    from agent.graph import nodes as nodes_module

    nodes_module._reset_decision_cache()


def run_dialogue(graph, row: dict, verbose: bool = False) -> dict:
    """跑完一段对话，收集判分需要的事实。"""
    session_id = f"eval-dlg-{row['id']}"
    config = {"configurable": {"thread_id": session_id}}
    state: dict = {}
    turns: list[dict] = []
    for index, turn in enumerate(row["turns"], start=1):
        state = graph.invoke(
            {"user_input": turn["user"], "session_id": session_id}, config=config
        )
        turns.append(
            {
                "index": index,
                "user": turn["user"],
                "reply": state.get("reply") or "",
                "intent": state.get("intent"),
                "branches": list(state.get("branches") or []),
                "handoff": bool(state.get("handoff")),
                "out_of_domain": bool(state.get("out_of_domain")),
                "decision": state.get("decision"),
            }
        )
        if verbose:
            print(f"    [轮 {index}] {turn['user']}")
            print(f"      → intent={state.get('intent')} branches={state.get('branches')}")
            print(f"      → {str(state.get('reply') or '')[:70]}")

    tool_results = list(state.get("tool_results") or [])
    facts = {
        "last_reply": state.get("reply") or "",
        "tools": [entry.get("name") for entry in tool_results if entry.get("name")],
        "handoff": bool(state.get("handoff")),
        "handoff_summary": state.get("handoff_summary") or "",
        "branches": sorted({b for turn in turns for b in turn["branches"]}),
        "tool_results": tool_results,
    }
    outcome = dialogue_outcome(row, facts)
    return {
        "id": row["id"],
        "scenario": row.get("scenario"),
        "chain": row.get("chain"),
        "category": row.get("category"),
        "turns": len(row["turns"]),
        "turn_log": turns,
        "facts": {
            "last_reply": facts["last_reply"],
            "tools": facts["tools"],
            "handoff": facts["handoff"],
            "handoff_summary": facts["handoff_summary"],
            "branches": facts["branches"],
        },
        "tools": aggregate_tools(tool_results),
        "resolution_criteria": row.get("resolution_criteria"),
        "note": row.get("note"),
        **outcome,
    }


# ---------------- 输出 ----------------


def print_overview(records: list[dict], backend: str, strategy: str) -> None:
    metrics = summarize(records)
    print(f"\n后端 {backend} / 策略 {strategy}：{metrics['dialogues']} 段对话、{metrics['turns']} 轮")
    print(f"  目标解决率（该解决的）：{metrics['resolution_rate']:.4f}"
          f"（{sum(1 for r in records if r['resolved_expected'] and r['resolved_predicted'])}"
          f"/{metrics['resolution_total']}）")
    print(f"  误判达成率（不该解决的）：{metrics['over_claimed_rate']:.4f}"
          f"（{sum(1 for r in records if not r['resolved_expected'] and r['resolved_predicted'])}"
          f"/{metrics['over_claimed_total']}）")
    print(f"  转人工召回：{metrics['handoff_recall']:.4f}；误转率：{metrics['handoff_false_rate']:.4f}"
          f"；转人工准确率：{metrics['handoff_accuracy']:.4f}")

    total_calls = sum(r["tools"]["calls"] for r in records)
    total_success = sum(r["tools"]["success"] for r in records)
    total_retried = sum(r["tools"]["retried_success"] for r in records)
    total_degraded = sum(r["tools"]["degraded"] for r in records)
    if total_calls:
        print(
            f"  工具调用 {total_calls} 次：成功 {total_success}（其中重试后成功 {total_retried}）、"
            f"降级 {total_degraded}、成功率 {total_success / total_calls:.4f}"
        )
    else:
        print("  本次没有触发工具调用")


def write_report(
    path: Path,
    backend: str,
    strategy: str,
    records: list[dict],
    skipped: list[dict],
    dataset: Path,
) -> None:
    metrics = summarize(records)
    total_calls = sum(r["tools"]["calls"] for r in records)
    total_success = sum(r["tools"]["success"] for r in records)
    total_retried = sum(r["tools"]["retried_success"] for r in records)
    total_degraded = sum(r["tools"]["degraded"] for r in records)

    lines = [
        "# 端到端对话评测报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据集：{dataset.name}（{metrics['dialogues']} 段对话 / {metrics['turns']} 轮，另有 {len(skipped)} 段因环境不匹配跳过）",
        f"- 运行配置：INTENT_BACKEND={backend}、INTENT_MULTI_STRATEGY={strategy}",
        "",
        "## 一、三个核心指标",
        "",
        "| 指标 | 值 | 分子/分母 | 说明 |",
        "| --- | --- | --- | --- |",
        f"| 目标解决率 | {metrics['resolution_rate']:.4f} | "
        f"{sum(1 for r in records if r['resolved_expected'] and r['resolved_predicted'])}/{metrics['resolution_total']} | "
        "预期应当解决的对话里，规则判据全部通过的比例 |",
        f"| 误判达成率 | {metrics['over_claimed_rate']:.4f} | "
        f"{sum(1 for r in records if not r['resolved_expected'] and r['resolved_predicted'])}/{metrics['over_claimed_total']} | "
        "预期不该解决的对话里，判据却全过的比例（虚报） |",
        f"| 转人工召回 | {metrics['handoff_recall']:.4f} | "
        f"{sum(1 for r in records if r['handoff_expected'] and r['handoff_actual'])}/{metrics['handoff_total']} | "
        "该转人工的真的转了 |",
        f"| 转人工误转率 | {metrics['handoff_false_rate']:.4f} | "
        f"{sum(1 for r in records if not r['handoff_expected'] and r['handoff_actual'])}/{metrics['handoff_false_total']} | "
        "不该转人工的却转了 |",
        f"| 工具成功率 | {total_success / total_calls:.4f} | {total_success}/{total_calls} | "
        f"其中重试后成功 {total_retried} 次 |" if total_calls else "| 工具成功率 | - | 0/0 | 本次没有工具调用 |",
        f"| 工具降级率 | {total_degraded / total_calls:.4f} | {total_degraded}/{total_calls} | "
        "查无此单、参数不合法、规则缺失等触发追问或转人工 |" if total_calls else "| 工具降级率 | - | 0/0 | 本次没有工具调用 |",
        "",
        "## 二、按链路分组",
        "",
        "| 链路 | 对话数 | 目标解决率 | 误判达成率 | 转人工召回 | 转人工误转率 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for chain, stats in group_stats(records, "chain"):
        lines.append(
            f"| {chain} | {stats['dialogues']} | {stats['resolution_rate']:.4f} | "
            f"{stats['over_claimed_rate']:.4f} | {stats['handoff_recall']:.4f} | "
            f"{stats['handoff_false_rate']:.4f} |"
        )
    lines += [
        "",
        "## 三、按样本性质分组",
        "",
        "（`noise`/`boundary` 是真实口语变形与模糊场景，**不并入核心解决率**——"
        "否则会把数据缺陷算成模型能力问题）",
        "",
        "| 性质 | 对话数 | 目标解决率 | 误判达成率 |",
        "| --- | --- | --- | --- |",
    ]
    for category, stats in group_stats(records, "category"):
        lines.append(
            f"| {category} | {stats['dialogues']} | {stats['resolution_rate']:.4f} | "
            f"{stats['over_claimed_rate']:.4f} |"
        )

    failed = [r for r in records if not r["resolved_match"] or not r["handoff_ok"]]
    lines += ["", f"## 四、未通过的对话（{len(failed)} 段，回放见 dialogues.json 的 turn_log）", ""]
    if not failed:
        lines.append("全部通过。")
    else:
        lines += [
            "| id | 场景 | 预期解决 | 实际解决 | 转人工(期望/实际) | 未通过的判据 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for record in failed:
            bad = [name for name, ok in record["checks"].items() if not ok]
            lines.append(
                f"| {record['id']} | {record['scenario']} | {record['resolved_expected']} | "
                f"{record['resolved_predicted']} | {record['handoff_expected']}/{record['handoff_actual']} | "
                f"{'、'.join(bad) or '-'} |"
            )
        for record in failed[:8]:
            lines += [
                "",
                f"### {record['id']} · {record['scenario']}",
                "",
                f"- 判定标准：{record['resolution_criteria']}",
                f"- 备注：{record['note'] or '-'}",
                "",
                "| 轮次 | 用户 | 意图/分支 | 回复（截断） |",
                "| --- | --- | --- | --- |",
            ]
            for turn in record["turn_log"]:
                lines.append(
                    f"| {turn['index']} | {turn['user'][:24]} | "
                    f"{turn['intent']}/{'、'.join(turn['branches']) or '-'} | "
                    f"{turn['reply'][:60].replace('|', '/')} |"
                )

    if skipped:
        lines += ["", "## 五、因环境不匹配跳过的用例", ""]
        lines += [f"- {item['id']}（{item['scenario']}）：{item['reason']}" for item in skipped]

    lines += [
        "",
        "## 六、怎么读这份报告",
        "",
        "1. **目标解决率**看「该办的办成了没有」，**误判达成率**看「有没有虚报」——两个一起看才有意义；",
        "2. **转人工召回低**通常是争议词/无据判定漏了；**误转率高**通常是情绪词太敏感或工具降级太激进；",
        "3. **工具降级率**本身不是坏事（查无此单就该追问），要看它降级后的动作对不对；",
        "4. 规则初判只做确定性核对，措辞是否得体、有没有解决用户真正的问题，由第三段 LLM-as-judge 补。",
        "",
        "复现命令：",
        "",
        "```bash",
        f"python scripts/eval_dialogues.py --backend {backend} --strategy {strategy}",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="端到端对话评测")
    parser.add_argument("--backend", choices=["rule", "local", "fewshot"], default="rule")
    parser.add_argument("--strategy", choices=["priority", "parallel"], default="priority")
    parser.add_argument("--local-dir", default=None, help="本地 adapter 目录（默认 models/v1）")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--only", nargs="*", default=None, help="只跑指定 id，调试用")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.dataset)
    if not rows:
        raise SystemExit(f"找不到对话集：{args.dataset}")
    if args.only:
        wanted = set(args.only)
        rows = [row for row in rows if row["id"] in wanted]

    print(f"对话集：{args.dataset.name}（{len(rows)} 段）")
    print(f"配置：后端 {args.backend}、多意图策略 {args.strategy}")
    apply_env(args.backend, args.strategy, args.local_dir)

    from agent.graph.workflow import build_graph
    from agent.intent.classifier import get_classifier

    classifier = get_classifier()
    preload = getattr(classifier, "preload", None)
    if preload is not None and args.backend != "rule":
        print(f"预热意图后端 {classifier.name} …")
        preload()
    print(f"意图后端实际使用：{classifier.name}")

    records: list[dict] = []
    skipped: list[dict] = []
    for row in rows:
        reason = skip_reason(row, args.backend, args.strategy)
        if reason:
            skipped.append({"id": row["id"], "scenario": row.get("scenario"), "reason": reason})
            print(f"[skip] {row['id']}：{reason}")
            continue
        print(f"[run ] {row['id']} · {row.get('scenario')}")
        records.append(run_dialogue(build_graph(), row, verbose=args.verbose))

    if not records:
        raise SystemExit("没有可跑的用例（都被环境要求跳过了），检查 --backend / --strategy")

    print_overview(records, args.backend, args.strategy)

    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset.name,
        "backend": args.backend,
        "strategy": args.strategy,
        "classifier": classifier.name,
        "metrics": summarize(records),
        "records": records,
        "skipped": skipped,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n指标 JSON：{args.json}")

    if not args.no_report:
        write_report(args.report, args.backend, args.strategy, records, skipped, args.dataset)
        print(f"报告已写入：{args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
