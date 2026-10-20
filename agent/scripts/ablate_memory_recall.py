"""记忆注入消融：同一批多轮场景，`MEMORY_RECALL=off` vs `on`，看注入长期事实有没有用。

两个组只差一个开关，其他条件完全一致：

1. **播种**：先把"这个用户以前说过什么"写成事实记忆；
2. **对照组**：`MEMORY_RECALL=off` 跑一遍（事实在库里，但不注入意图上下文）；
3. **实验组**：`MEMORY_RECALL=on` 跑一遍（`intent_node` 每轮召回 top-k 并注入）。

两种播种方式，测的是链路的两段：

- `prefix`（默认）：把每段场景**除最后一轮外**的用户话当作"上一次会话"写进事实记忆。
  最后一轮通常是最碎的一句（"一半吧""照片我传不上"），正好检验召回能不能补上上下文。
  **它测的是"注入"这一段**：写入是确定性的，变量只有开关。
- `replay`：垫 5 轮闲聊把消息推过压缩阈值，让真实的压缩节点触发事实写入。
  **它测的是"自动写入"这一段**：抽取规则能不能从真实对话里写出有用的事实。

两种都跑一遍，才能分清"没收益"是抽取出问题还是召回/注入出问题。

> 口径提醒：`prefix` 是构造出来的历史会话，不是线上真实数据。它能证明注入链路通、
> 阈值和 trace 都对，不能单独证明线上一定涨点；收益只看 on/off 两组在同一批场景上的差。
> 另外 `rule` 意图后端根本不读上下文，用 rule 跑出来的 on/off 必然打平——
> 想看到注入改变判定，要加 `--backend local`。

事实库与索引通过 `MEMORY_FACTS_DB` / `MEMORY_INDEX_PATH` 指到临时文件，不动正式数据。

用法（在 agent/ 目录下执行）：
    python scripts/ablate_memory_recall.py                      # prefix 播种，6 段多轮场景、垫 5 轮
    python scripts/ablate_memory_recall.py --seed-mode replay
    python scripts/ablate_memory_recall.py --limit 4 --padding 6
    python scripts/ablate_memory_recall.py --backend local

`prefix` 播种时默认只挑轮次 ≥2 的场景——单轮对话没有"上一次会话"可播种。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from scripts.ablate_memory import build_padded_row  # noqa: E402
from scripts.eval_dialogues import evaluate_checks, load_jsonl  # noqa: E402

DEFAULT_DATASET = AGENT_ROOT / "eval" / "datasets" / "dialogues_golden.jsonl"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "memory_recall.md"
DEFAULT_JSON = AGENT_ROOT / "eval" / "reports" / "memory_recall.json"

SEED_MODES = ("prefix", "replay")


def run_dialogue(
    graph, row: dict, session_id: str, *, thread_id: str | None = None
) -> dict:
    """跑一段对话，收集判据、工具、追问次数，以及每轮注入了哪几条事实。

    `session_id` 决定事实记忆归谁；`thread_id` 决定 LangGraph 的 checkpoint。
    播种/对照/实验三次跑必须用**不同的 thread**（否则第二次会从第一次的
    checkpoint 恢复槽位和消息），但要用**相同的 session**（否则看不到同一批事实）。
    """
    config = {"configurable": {"thread_id": thread_id or session_id}}
    state: dict = {}
    branches: set[str] = set()
    recalls: list[int] = []
    turn_log: list[dict] = []
    tool_hits: list[bool] = []
    asks: int = 0
    tool_offset: int = 0
    started = time.perf_counter()
    for turn in row["turns"]:
        state = graph.invoke(
            {"user_input": turn["user"], "session_id": session_id}, config=config
        )
        branches.update(state.get("branches") or [])
        # tool_results 是累计的：按 offset 切片，只看本轮新调用的工具
        results = list(state.get("tool_results") or [])
        called = {
            entry.get("name") for entry in results[max(0, tool_offset):] if entry.get("name")
        }
        tool_offset = len(results)
        expected_tool = turn.get("expected_tool")
        if expected_tool:
            # 数据集没有逐轮意图金标，用"该不该调这个工具"当意图路由的代理指标
            tool_hits.append(expected_tool in called)
        hits = list(state.get("memory_hits") or [])
        recalls.append(len(hits))
        turn_log.append(
            {
                "user": turn["user"],
                "recall": [
                    {
                        "id": hit.get("id"),
                        "fact": str(hit.get("fact") or "")[:40],
                        "vector_score": hit.get("vector_score"),
                        "entity_hits": hit.get("entity_hits"),
                    }
                    for hit in hits
                ],
            }
        )
        if "订单号" in (state.get("reply") or ""):
            asks += 1
    elapsed_ms = (time.perf_counter() - started) * 1000

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
        "passed": bool(checks) and all(checks.values()),
        "tools": len(tool_results),
        "tool_success": sum(1 for entry in tool_results if entry.get("ok")),
        "handoff": facts["handoff"],
        "recalls": recalls,
        "turn_log": turn_log,
        "tool_hits": tool_hits,
        "asks": asks,
        "elapsed_ms": elapsed_ms,
    }


def seed_prefix(session_id: str, row: dict) -> int:
    """把场景前几轮用户话直接写成事实，模拟"上一次会话里用户已经说过"。

    最后一轮不播种：它是这次要测的那句话，提前写进去就是泄漏。
    返回写入的事实条数。

    这里**不走 LLM 抽取**，直接落一条事实 + 一条向量：本模式测的是"注入"，
    写入必须是确定性的；抽取质量由 `--seed-mode replay` 单独测。
    """
    turns = list(row.get("turns") or [])
    prefix = [str(turn.get("user") or "").strip() for turn in turns[:-1]]
    prefix = [text for text in prefix if text]
    if not prefix:
        return 0

    from agent.memory import facts as facts_store
    from agent.memory.index import MemoryIndex
    from agent.memory.search import query_entities
    from agent.memory.writer import _default_embedder, _encode, default_index_path

    text = "用户此前反馈：" + "；".join(prefix)
    record, created = facts_store.insert_fact(
        session_id,
        text,
        category="summary",
        entities=sorted(query_entities(text)),
        source="seed",
    )
    if not created:
        print(f"[warn] 事实没写进去（可能已存在）：{session_id}｜{text[:40]}")
        return 0
    embedder = _default_embedder()
    index = MemoryIndex(default_index_path()).load()
    index.upsert(
        record["id"],
        fact=text,
        session_id=session_id,
        entities=record.get("entities") or [],
        vector=_encode(embedder, text),
    )
    index.save()
    return 1


def seed_replay(rows: list[dict], padding: int, fresh_graph) -> dict[str, int]:
    """原做法：垫场跑一遍，让压缩节点触发自动写入；返回每段写进几条事实。"""
    from agent.memory import facts as facts_store

    os.environ["MEMORY_FACTS"] = "rule"
    os.environ["MEMORY_RECALL"] = "off"
    seeded: dict[str, int] = {}
    for row in rows:
        session_id = f"ablate-recall-{row['id']}"
        padded = build_padded_row(row, padding)
        run_dialogue(fresh_graph(), padded, session_id, thread_id=f"{session_id}-seed")
        seeded[row["id"]] = len(facts_store.list_facts(session_id, limit=500))
    return seeded


def aggregate(records: list[dict], variant: str) -> dict:
    """把逐段结果压成对比用的指标；召回按"每轮平均注入条数"算，不按对话平均。"""
    total = len(records) or 1
    turns = sum(len(item["recalls"]) for item in records)
    injected = sum(sum(item["recalls"]) for item in records)
    calls = sum(item["tools"] for item in records)
    success = sum(item["tool_success"] for item in records)
    expected_tools = sum(len(item["tool_hits"]) for item in records)
    hit_tools = sum(sum(1 for hit in item["tool_hits"] if hit) for item in records)
    return {
        "variant": variant,
        "dialogues": len(records),
        "turns": turns,
        "pass_rate": round(sum(1 for item in records if item["passed"]) / total, 4),
        "tool_calls": calls,
        "tool_success_rate": round(success / calls, 4) if calls else 0.0,
        "tool_hit_rate": round(hit_tools / expected_tools, 4) if expected_tools else 0.0,
        "tool_expect": expected_tools,
        "handoff_rate": round(sum(1 for item in records if item["handoff"]) / total, 4),
        "avg_recall": round(injected / turns, 2) if turns else 0.0,
        "recall_turns": sum(1 for item in records for count in item["recalls"] if count),
        "injected_dialogues": sum(1 for item in records if any(item["recalls"])),
        "asks_total": sum(item["asks"] for item in records),
        "avg_elapsed_ms": round(sum(item["elapsed_ms"] for item in records) / total, 1),
    }


def recommendation(off: dict, on: dict, *, backend: str = "rule") -> str:
    """给出"要不要打开"的建议；先排除无效实验，再比结果。"""
    if not on["dialogues"]:
        return "没有跑出实验组数据，本次对比无效"
    if on["avg_recall"] <= 0:
        return (
            "实验组一次都没召回（事实没播种上、阈值太高、或查询和事实无关），"
            "本次对比无效：先看第二节的机制自检"
        )
    if (on["pass_rate"], on["tool_success_rate"], on["tool_hit_rate"]) > (
        off["pass_rate"],
        off["tool_success_rate"],
        off["tool_hit_rate"],
    ):
        return "记忆注入带来收益：判据/工具成功率/意图路由命中提升，可以评估把 MEMORY_RECALL 打开"
    if (on["pass_rate"], on["tool_success_rate"], on["tool_hit_rate"]) < (
        off["pass_rate"],
        off["tool_success_rate"],
        off["tool_hit_rate"],
    ):
        return "记忆注入变差：先查注入的事实是不是噪声（阈值太高/太低、top-k 太大）"
    if on["asks_total"] < off["asks_total"]:
        return (
            "判据与工具成功率打平，但重复追问更少：说明记忆补上了上下文，"
            "建议小流量打开 MEMORY_RECALL 观察"
        )
    if backend == "rule":
        return (
            "两者打平：机制跑通了，但 rule 意图后端不读上下文，注入本来就不会改变判定；"
            "保持 off，想验证收益用 --backend local"
        )
    return "两者打平：机制可用但本批数据没有收益，保持 off，把'为什么没收益'写进决策日志"


def write_report(
    path: Path,
    *,
    rows: list[dict],
    off: dict,
    on: dict,
    details: dict[str, list[dict]] | None = None,
    padding: int,
    seed_mode: str = "prefix",
    backend: str = "rule",
) -> None:
    details = details or {}
    seed_note = {
        "prefix": "把场景前几轮当历史会话写入事实（测注入链路）",
        "replay": "垫场跑一遍，由压缩节点自动写入事实（测自动写入链路）",
    }.get(seed_mode, seed_mode)
    lines = [
        "# 记忆注入消融（MEMORY_RECALL off vs on）",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 场景：{len(rows)} 段，每段垫 {padding} 轮闲聊",
        f"- 播种：{seed_mode}——{seed_note}",
        f"- 意图后端：{backend}",
        "",
        "## 一、对比",
        "",
        "| 指标 | 不注入（off） | 注入（on） |",
        "| --- | --- | --- |",
        f"| 判据通过率 | {off['pass_rate']:.4f} | {on['pass_rate']:.4f} |",
        f"| 工具调用次数 | {off['tool_calls']} | {on['tool_calls']} |",
        f"| 工具成功率 | {off['tool_success_rate']:.4f} | {on['tool_success_rate']:.4f} |",
        f"| 意图路由命中率（该调的工具调了） | {off['tool_hit_rate']:.4f} | "
        f"{on['tool_hit_rate']:.4f} |",
        f"| 转人工率 | {off['handoff_rate']:.4f} | {on['handoff_rate']:.4f} |",
        f"| 平均每轮注入事实数 | {off['avg_recall']} | {on['avg_recall']} |",
        f"| 追问订单号总次数 | {off['asks_total']} | {on['asks_total']} |",
        f"| 平均耗时 | {off['avg_elapsed_ms']:.0f} ms | {on['avg_elapsed_ms']:.0f} ms |",
        "",
        "## 二、机制自检（先确认注入真的发生，再看收益）",
        "",
        f"- 实验组发生注入的对话：{on['injected_dialogues']}/{on['dialogues']}",
        f"- 实验组发生注入的轮次：{on['recall_turns']}/{on['turns']}",
        f"- 对照组注入轮次：{off['recall_turns']}（开关关着，应为 0）",
        "",
    ]
    if on["recall_turns"] == 0:
        lines += [
            "> 没有任何一轮发生注入，这次 on/off 的差不能归因于记忆。"
            "先查播种是否写进库、相似度阈值是否过高、或场景本身与历史事实无关。",
            "",
        ]
    if backend == "rule":
        lines += [
            "> rule 意图后端按关键词判定，不读注入的上下文；用它跑出的 on/off 一定一样。"
            "要验证注入对判定的影响，用 `--backend local`。",
            "",
        ]
    elif (os.environ.get("LOCAL_INTENT_CONTEXT", "off") or "off").lower() not in {
        "on", "1", "true", "yes",
    }:
        lines += [
            "> 本地 SFT 模型默认不拼上下文（训练是单轮的）；本次 on/off 预期打平。"
            "要验证注入收益，先给带上下文的样本再训一版，或临时 `LOCAL_INTENT_CONTEXT=on`。",
            "",
        ]

    if details:
        lines += [
            "## 三、逐段明细",
            "",
            "| id | 每轮注入条数（on） | 追问订单号（on） | 判据 | 注入的事实 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for record in details.get("on", []):
            counts = "/".join(str(count) for count in record["recalls"])
            facts = "；".join(
                f"{hit['fact']}（{hit.get('vector_score')}）"
                for turn in record["turn_log"]
                for hit in turn["recall"]
            ) or "-"
            lines.append(
                f"| {record['id']} | {counts} | {record['asks']} | "
                f"{'通过' if record['passed'] else '未通过'} | {facts} |"
            )
        lines.append("")

    lines += [
        "## 四、结论",
        "",
        recommendation(off, on, backend=backend),
        "",
        "> 口径：对话集没有逐轮意图金标，所以用「判据通过率 + 工具成功率 + 重复追问次数」"
        "代替「意图命中率」——它们是同一件事的行为侧证据。",
        "> 收益结论只对这批场景成立；`prefix` 播种是构造的历史会话，不是线上分布。",
        "",
        "## 五、怎么复现",
        "",
        "```bash",
        "python scripts/ablate_memory_recall.py                     # 默认 prefix 播种",
        "python scripts/ablate_memory_recall.py --seed-mode replay   # 测自动写入",
        "python scripts/ablate_memory_recall.py --backend local      # 让注入真正影响判定",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="记忆注入消融")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--padding", type=int, default=5)
    parser.add_argument("--backend", default="rule")
    parser.add_argument("--seed-mode", choices=SEED_MODES, default="prefix")
    parser.add_argument("--sim", type=float, default=None, help="覆盖 MEMORY_RECALL_SIM")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.dataset)
    if args.ids:
        wanted = set(args.ids)
        rows = [row for row in rows if row["id"] in wanted]
    elif args.seed_mode == "prefix":
        # prefix 只对多轮场景有意义：单轮没有"上一次会话"可播种
        rows = [row for row in rows if len(row.get("turns") or []) >= 2]
    rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"没有可用场景：{args.dataset}")
    if args.backend == "local" and (
        os.environ.get("LOCAL_INTENT_CONTEXT", "off") or "off"
    ).lower() not in {"on", "1", "true", "yes"}:
        print(
            "[提示] LOCAL_INTENT_CONTEXT=off：本地模型不读注入的上下文，"
            "on/off 预期打平；这不是注入没生效，先看报告的机制自检。"
        )

    tmp = Path(tempfile.mkdtemp(prefix="memory-recall-ablation-"))
    os.environ["INTENT_BACKEND"] = args.backend
    os.environ["MEMORY_FACTS_DB"] = str(tmp / "facts.db")
    os.environ["MEMORY_INDEX_PATH"] = str(tmp / "memory_index.json")
    if args.sim is not None:
        os.environ["MEMORY_RECALL_SIM"] = str(args.sim)

    from agent.graph.workflow import get_graph

    def fresh_graph():
        return get_graph(persistent=False)

    def padded(row: dict) -> dict:
        item = build_padded_row(row, args.padding)
        item["id"] = row["id"]  # 报告里沿用原 id，不让 mem- 前缀干扰阅读
        return item

    variants: dict[str, dict] = {}
    details: dict[str, list[dict]] = {}
    seeded: dict[str, int] = {}

    # ① 播种：两种方式，写进同一个临时事实库
    if args.seed_mode == "prefix":
        print(f"[播种] prefix：{len(rows)} 段，用场景前几轮当历史会话…")
        for row in rows:
            seeded[row["id"]] = seed_prefix(f"ablate-recall-{row['id']}", row)
    else:
        print(f"[播种] replay：{len(rows)} 段，垫 {args.padding} 轮让压缩节点自动写入…")
        seeded = seed_replay(rows, args.padding, fresh_graph)
    empty = [row_id for row_id, count in seeded.items() if not count]
    print(f"  写入事实：{sum(seeded.values())} 条；没有写进去的场景：{empty or '无'}")

    # ② 对照组：事实存在但不注入
    os.environ["MEMORY_FACTS"] = "off"
    os.environ["MEMORY_RECALL"] = "off"
    print("[对照] MEMORY_RECALL=off …")
    details["off"] = [
        run_dialogue(
            fresh_graph(),
            padded(row),
            f"ablate-recall-{row['id']}",
            thread_id=f"ablate-recall-{row['id']}-off",
        )
        for row in rows
    ]
    variants["off"] = aggregate(details["off"], "off")

    # ③ 实验组：每轮召回并注入
    os.environ["MEMORY_RECALL"] = "on"
    print("[实验] MEMORY_RECALL=on …")
    details["on"] = [
        run_dialogue(
            fresh_graph(),
            padded(row),
            f"ablate-recall-{row['id']}",
            thread_id=f"ablate-recall-{row['id']}-on",
        )
        for row in rows
    ]
    variants["on"] = aggregate(details["on"], "on")

    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "padding": args.padding,
        "seed_mode": args.seed_mode,
        "backend": args.backend,
        "seeded": seeded,
        "variants": variants,
        "details": details,
        "recommendation": recommendation(variants["off"], variants["on"], backend=args.backend),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_report:
        write_report(
            args.report,
            rows=rows,
            off=variants["off"],
            on=variants["on"],
            details=details,
            padding=args.padding,
            seed_mode=args.seed_mode,
            backend=args.backend,
        )
        print(f"报告：{args.report}")
    print(f"JSON：{args.json}")
    print(f"\n结论：{payload['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
