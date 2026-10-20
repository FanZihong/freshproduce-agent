"""从对话 trace 里挖候选 badcase，产出 badcase 池（自进化飞轮的入口）。

挖的是**候选**，不是结论。信号只负责把"值得人看一眼"的轮次挑出来，
真值仍然要人工标——池子里每一行都留了 `expected_intent` / `expected_action` / `note` 三列
等着填，填完就能喂给意图层的 `evolve_intent.py`（训练集反哺），
或者喂给知识库/reranker 的反哺脚本。

八个信号各自回答一个问题：
- 空回复 / 工具失败 / 护栏触发：这一轮**没办成事**，用户大概率不满意；
- 判为域外 / 转人工：走的是兜底路径，要复核是"真该兜底"还是"判据太敏感"；
- 低置信澄清 / 低置信放行：**模型自己不确定**的地方，是意图短板的富矿；
- 多意图：一轮里命中多条分支，路由与回复顺序值得复核。

一个已知的正常现象（首次跑 trace 就踩到）：**补槽位轮不该算低置信**。用户在填槽时回的是
「2026093000123」「大概一半」这种碎片，分类器（尤其规则版）把它打成 chitchat/fallback 是正常的，
而且这些轮次的路由由"续接未完成流程"接管，标签根本不影响结果。这类轮次默认从候选里剔除，
只在统计里报个数；想连它们一起看就加 `--include-slot-turns`。

用法（在 agent/ 目录下执行）：
    python scripts/mine_traces.py                                  # 挖 data/trace 下全部
    python scripts/mine_traces.py --session cli-101530             # 只看某个会话
    python scripts/mine_traces.py --min-confidence 0.6 --limit 50
    python scripts/mine_traces.py --print-transcript cli-101530    # 原样回放某个会话

池子里的行去重按用户原话（大小写/首尾空格归一），重复出现只在 `seen` 里累加计数，
免得同一句话刷屏。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.trace import DEFAULT_TRACE_DIR, load_traces, session_transcript, sessions_of  # noqa: E402

DEFAULT_OUT = AGENT_ROOT / "eval" / "datasets" / "trace_badcase_pool.jsonl"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "trace_mining.md"


@dataclass(frozen=True)
class Signal:
    """一个挖掘信号：key 用于机器判断，label/reason 用于给人看。"""

    key: str
    label: str
    reason: str
    severity: int


SIGNALS: tuple[Signal, ...] = (
    Signal("empty_reply", "空回复", "这一轮没有给出任何回复，链路可能中断", 5),
    Signal("tool_failed", "工具失败/降级", "工具调用没成功，用户没拿到结果", 4),
    Signal(
        "loop_guard",
        "护栏触发",
        "追问次数触到上限，说明槽位或澄清没收敛",
        4,
    ),
    Signal("ood", "判为域外", "走的是兜底分支：可能真域外，也可能是领域词没命中被误杀", 3),
    Signal("handoff", "转人工", "要复核是真该转，还是判据过敏导致的误转", 3),
    Signal("clarify", "低置信澄清", "系统自己都不确定，这句是模型短板的高发区", 3),
    Signal(
        "low_confidence",
        "低置信放行",
        "原始置信度低于阈值、决策层仍放行：模型自己不确定的轮次，误判高发区"
        "（阈值用 --min-confidence 调，补槽位轮已自动排除）",
        2,
    ),
    Signal("multi_intent", "多意图", "一轮命中多条分支，路由和回复顺序值得复核", 2),
)

SIGNAL_BY_KEY = {signal.key: signal for signal in SIGNALS}

# 补槽位轮里出现这两种标签是正常现象（碎片句的兜底标签）
SLOT_TURN_LABELS: frozenset[str] = frozenset({"chitchat", "fallback"})


def is_slot_continuation(record: dict) -> bool:
    """这一轮是不是"填槽位的碎片句"。

    判据：会话里已经有槽位（说明上一轮在补槽）、本轮标签是兜底类、且没有转人工。
    这种轮次的标签不影响路由（续接流程优先），低置信不代表模型有短板。
    """
    label = str(record.get("intent") or "")
    return (
        bool(record.get("slots"))
        and label in SLOT_TURN_LABELS
        and not record.get("handoff")
    )


def slot_continuation_count(records: list[dict], **kwargs) -> int:
    """统计有多少轮是"补槽位轮里的低置信"（报告里要如实写这条被跳过多少）。"""
    return sum(
        1
        for record in records
        if is_slot_continuation(record)
        and 0 < float(record.get("confidence") or 0.0) < kwargs.get("min_confidence", 0.5)
        and (record.get("decision") or "accept") == "accept"
        and not record.get("out_of_domain")
    )


def signals_of(
    record: dict,
    *,
    min_confidence: float = 0.5,
    loop_limit: int = 3,
    clarify_limit: int = 2,
    include_slot_turns: bool = False,
) -> list[Signal]:
    """判断一轮对话命中了哪些信号。纯函数，方便单测。"""
    hits: list[Signal] = []
    tools = record.get("tools") or []
    confidence = float(record.get("confidence") or 0.0)
    decision = record.get("decision") or "accept"
    out_of_domain = bool(record.get("out_of_domain"))

    if not str(record.get("reply") or "").strip():
        hits.append(SIGNAL_BY_KEY["empty_reply"])
    if any(not item.get("ok") for item in tools):
        hits.append(SIGNAL_BY_KEY["tool_failed"])
    if int(record.get("loop_count") or 0) >= loop_limit or int(
        record.get("clarify_rounds") or 0
    ) >= clarify_limit:
        hits.append(SIGNAL_BY_KEY["loop_guard"])
    if out_of_domain:
        hits.append(SIGNAL_BY_KEY["ood"])
    if record.get("handoff"):
        hits.append(SIGNAL_BY_KEY["handoff"])
    if decision == "clarify":
        hits.append(SIGNAL_BY_KEY["clarify"])
    # 低置信"放行"才可疑：走澄清的已经在 clarify 信号里了，域外也不看置信度；
    # 补槽位轮里标签不影响路由，低置信属正常，默认不算（见 is_slot_continuation）
    slot_turn = is_slot_continuation(record)
    if (
        0 < confidence < min_confidence
        and decision == "accept"
        and not out_of_domain
        and (include_slot_turns or not slot_turn)
    ):
        hits.append(SIGNAL_BY_KEY["low_confidence"])
    if len(record.get("branches") or []) > 1:
        hits.append(SIGNAL_BY_KEY["multi_intent"])
    return hits


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).lower()


def build_candidate(record: dict, hits: list[Signal], transcript: list[dict]) -> dict:
    """把命中信号的轮次整理成一行可人工标注的池子记录。"""
    return {
        "id": f"trace-{record.get('session_id')}-t{record.get('turn')}",
        "source": "trace",
        "created_at": record.get("created_at") or "",
        "session_id": record.get("session_id") or "",
        "turn": int(record.get("turn") or 0),
        "text": record.get("user_input") or "",
        "signal_keys": [signal.key for signal in hits],
        "signals": [signal.label for signal in hits],
        "severity": max(signal.severity for signal in hits),
        "reason": "；".join(signal.reason for signal in hits),
        "observed": {
            "intent": record.get("intent") or "",
            "confidence": float(record.get("confidence") or 0.0),
            "branches": list(record.get("branches") or []),
            "decision": record.get("decision") or "",
            "out_of_domain": bool(record.get("out_of_domain")),
            "ood_reason": record.get("ood_reason") or "",
            "handoff": bool(record.get("handoff")),
            "handoff_reason": record.get("handoff_reason") or "",
            "tools": list(record.get("tools") or []),
            "reply": record.get("reply") or "",
            "backend": record.get("backend") or "",
        },
        # 上下文：人工判断"客服是不是答非所问"必须看到之前几轮
        "transcript": [
            {"turn": item.get("turn"), "user": item.get("user_input"), "reply": item.get("reply")}
            for item in transcript
        ],
        # 下面三列是留给人工的
        "expected_intent": "",
        "expected_action": "",
        "note": "",
        "status": "pending",
        "seen": 1,
        "sessions": [record.get("session_id") or ""],
    }


def mine(
    records: list[dict],
    *,
    min_confidence: float = 0.5,
    loop_limit: int = 3,
    clarify_limit: int = 2,
    session: str | None = None,
    backend: str | None = None,
    include_slot_turns: bool = False,
    limit: int = 0,
) -> list[dict]:
    """按会话聚合后逐轮判信号，去重、排序，返回候选清单。"""
    filtered = [
        item
        for item in records
        if (not session or item.get("session_id") == session)
        and (not backend or item.get("backend") == backend)
    ]
    by_key: dict[str, dict] = {}
    order: list[str] = []

    for session_id in sessions_of(filtered):
        transcript = session_transcript(filtered, session_id)
        for record in transcript:
            hits = signals_of(
                record,
                min_confidence=min_confidence,
                loop_limit=loop_limit,
                clarify_limit=clarify_limit,
                include_slot_turns=include_slot_turns,
            )
            if not hits:
                continue
            candidate = build_candidate(record, hits, transcript)
            key = normalize_text(candidate["text"])
            if key in by_key:
                existing = by_key[key]
                existing["seen"] += 1
                if candidate["session_id"] not in existing["sessions"]:
                    existing["sessions"].append(candidate["session_id"])
                continue
            by_key[key] = candidate
            order.append(key)

    candidates = [by_key[key] for key in order]
    candidates.sort(key=lambda item: (-item["severity"], item["created_at"]))
    return candidates[:limit] if limit else candidates


def signal_counts(records: list[dict], **kwargs) -> Counter:
    """统计每个信号在多少轮里出现（按轮计，不按去重后的池子计）。"""
    counter: Counter = Counter()
    for record in records:
        for signal in signals_of(record, **kwargs):
            counter[signal.key] += 1
    return counter


def write_pool(path: Path, candidates: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(item, ensure_ascii=False) for item in candidates)
    path.write_text(text + ("\n" if candidates else ""), encoding="utf-8")


def load_pool(path: Path) -> list[dict]:
    """读已有的候选池；文件不存在或读不出来就当空的。"""
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    except (OSError, json.JSONDecodeError):
        return []
    return rows


HUMAN_FIELDS: tuple[str, ...] = ("expected_intent", "expected_action", "note")


def is_labelled(row: dict) -> bool:
    """这行有没有人工标注的痕迹（标注是人工成本，重新挖掘时不能丢）。"""
    return row.get("status") == "labeled" or any(row.get(field) for field in HUMAN_FIELDS)


def merge_pool(existing: list[dict], candidates: list[dict]) -> tuple[list[dict], dict]:
    """把本次候选合并进旧池子，而不是直接覆盖。

    规则：
    - 同一 id（同一会话同一轮）以**本次的事实**为准，但把旧行里人工填的三列搬过来；
    - 旧池子里"已标注、这次没挖到"的行留下——人已经花过时间，不能因为阈值调了一下就丢；
    - 旧池子里没标注、这次也没挖到的行丢掉，避免池子越滚越大都是陈数据。
    """
    old_by_id = {row.get("id"): row for row in existing}
    merged: list[dict] = []
    added = carried = kept_labelled = 0
    for item in candidates:
        old = old_by_id.pop(item.get("id"), None)
        if old:
            filled = False
            for field in HUMAN_FIELDS:
                if old.get(field):
                    item[field] = old[field]
                    filled = True
            if old.get("status") == "labeled":
                item["status"] = "labeled"
                filled = True
            carried += 1 if filled else 0
        else:
            added += 1
        merged.append(item)
    for row in old_by_id.values():
        if is_labelled(row):
            merged.append(row)
            kept_labelled += 1
    return merged, {"added": added, "carried": carried, "kept_labelled": kept_labelled}


def print_transcript(records: list[dict], session_id: str) -> int:
    """原样回放一个会话：每轮的意图/分支/工具/是否转人工/回复。"""
    turns = session_transcript(records, session_id)
    if not turns:
        print(f"trace 里没有会话 {session_id}（现有：{'、'.join(sessions_of(records)[:10]) or '空'}）")
        return 1
    print(f"=== 会话 {session_id}（{len(turns)} 轮）===")
    for item in turns:
        tools = item.get("tools") or []
        tool_text = "、".join(
            f"{tool.get('name')}{'✓' if tool.get('ok') else '✗'}" for tool in tools
        )
        print(f"\n[第 {item.get('turn')} 轮] {item.get('created_at')}")
        print(f"  用户：{item.get('user_input')}")
        print(
            f"  意图：{item.get('intent')}（{item.get('confidence')}）"
            f" 分支：{'/'.join(item.get('branches') or []) or '-'}"
            f" 决策：{item.get('decision')} 域外：{item.get('out_of_domain')}"
        )
        if tool_text:
            print(f"  工具：{tool_text}")
        if item.get("handoff"):
            print(f"  转人工：{item.get('handoff_reason')} | 交接摘要：{item.get('handoff_summary')}")
        print(f"  客服：{item.get('reply')}")
    return 0


def write_report(
    path: Path,
    *,
    trace_path: Path,
    records: list[dict],
    candidates: list[dict],
    counts: Counter,
    backends: Counter | None = None,
    slot_skipped: int = 0,
    backend: str | None = None,
    scoped_turns: int | None = None,
    merge_stats: dict | None = None,
    **kwargs,
) -> None:
    sessions = sessions_of(records)
    scope = f"- 本次范围：{scoped_turns} 轮" if scoped_turns is not None else ""
    if scope and scoped_turns != len(records):
        scope += f"（全量 {len(records)} 轮；筛选项：" + (
            f"backend={backend}" if backend else "session"
        ) + "）"
    lines = [
        "# trace 挖掘报告（候选 badcase 池）",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- trace 来源：{trace_path}",
        f"- 规模：{len(sessions)} 个会话、{len(records)} 轮对话",
        scope,
        f"- 候选：{len(candidates)} 条（按用户原话去重后）",
        "- 后端分布："
        + (
            "、".join(f"{name} {count} 轮" for name, count in (backends or Counter()).items())
            or "未记录"
        )
        + (f"（本次只看 {backend}）" if backend else ""),
        (
            f"- 合并进池子：新增 {merge_stats.get('added', 0)} 条、"
            f"带回原有人工标注 {merge_stats.get('carried', 0)} 条、"
            f"保留已标注但本次未挖到 {merge_stats.get('kept_labelled', 0)} 条"
            if merge_stats
            else ""
        ),
        "",
        f"> 补槽位轮里的「低置信」已跳过 {slot_skipped} 轮：用户回的是订单号、比例这类碎片，"
        "标签不影响路由（续接流程优先），不算模型短板。想连它们一起看就加 `--include-slot-turns`。",
        "",
        "> 候选池是**合并**写入的：同一条以本次事实为准、人工填过的三列会搬过来，"
        "已标注但这次没挖到的行也保留。所以池子为空只说明「本次范围内没有候选」，"
        "不代表把之前的标注弄丢了。",
        "",
        "## 一、信号分布（按轮统计，一轮可以命中多个）",
        "",
        "| 信号 | 命中轮数 | 说明 |",
        "| --- | --- | --- |",
    ]
    for signal in SIGNALS:
        lines.append(f"| {signal.label} | {counts.get(signal.key, 0)} | {signal.reason} |")

    lines += [
        "",
        "## 二、候选清单（按严重度排序，最多 30 条）",
        "",
        "| id | 用户原话 | 信号 | 意图(置信度) | 客服回复（截断） |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in candidates[:30]:
        observed = item["observed"]
        reply = (observed["reply"] or "").replace("\n", " ").replace("|", "/")
        lines.append(
            f"| {item['id']} | {item['text'][:30]} | {'、'.join(item['signals'])} | "
            f"{observed['intent']}（{observed['confidence']:.2f}） | {reply[:60]} |"
        )

    lines += [
        "",
        "## 三、怎么用这一池候选",
        "",
        "1. 打开 `eval/datasets/trace_badcase_pool.jsonl`，逐条填三列：`expected_intent`（正确意图）、"
        "`expected_action`（正确的处置，比如「该转人工」「该追问订单号」）、`note`（一句话理由），"
        "填完把 `status` 改成 `labeled`；",
        "2. 只想复核某一条的上下文时，用 `--print-transcript <session_id>` 原样回放整个会话；",
        "3. 标好的行分成三路：意图写错的进意图训练集（`evolve_intent.py`），"
        "知识库答不出的进知识库补条目，检索没命中的进 reranker 迭代。",
        "",
        "> 信号只负责「把可疑的挑出来」，不负责下结论。凡是这一池里被判为「没问题」的样本，"
        "正好可以用来收紧信号阈值（比如域外误杀率就是这么做出来的）。",
        "",
        "复现命令：",
        "",
        "```bash",
        f"python scripts/mine_traces.py --trace {trace_path}"
        + (f" --backend {backend}" if backend else "")
        + "".join(
            (
                f" --{key.replace('_', '-')}"
                if value is True
                else f" --{key.replace('_', '-')} {value}"
            )
            for key, value in kwargs.items()
            if value not in (None, 0, 0.5, 3, 2) and value is not False
        ),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="从对话 trace 里挖候选 badcase")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE_DIR, help="trace 文件或目录")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="候选池输出（jsonl）")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--session", default=None, help="只看某个会话")
    parser.add_argument("--backend", default=None, help="只看某个后端的轮次，例如 local-lora")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="低置信阈值")
    parser.add_argument("--loop-limit", type=int, default=3, help="loop_count 护栏阈值")
    parser.add_argument("--clarify-limit", type=int, default=2, help="澄清轮次护栏阈值")
    parser.add_argument(
        "--include-slot-turns",
        action="store_true",
        help="把补槽位轮的低置信也收进候选（默认跳过，它们属于正常现象）",
    )
    parser.add_argument("--limit", type=int, default=0, help="最多输出多少条（0=全部）")
    parser.add_argument("--print-transcript", default=None, help="回放指定会话后退出")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    records = load_traces(args.trace)

    if args.print_transcript:
        return print_transcript(records, args.print_transcript)

    if not records:
        print(
            f"没有读到任何 trace：{args.trace}\n"
            "先去跑几轮对话把数据攒起来，例如：\n"
            "  python scripts/run_cli.py --demo\n"
            "  python scripts/run_cli.py --turns \"草莓坏了一半|2026093000123|大概一半\""
        )
        return 1

    kwargs = {
        "min_confidence": args.min_confidence,
        "loop_limit": args.loop_limit,
        "clarify_limit": args.clarify_limit,
        "include_slot_turns": args.include_slot_turns,
    }
    scoped = [
        item
        for item in records
        if (not args.session or item.get("session_id") == args.session)
        and (not args.backend or item.get("backend") == args.backend)
    ]
    candidates = mine(records, session=args.session, backend=args.backend, limit=args.limit, **kwargs)
    counts = signal_counts(scoped, **kwargs)
    slot_skipped = slot_continuation_count(scoped, min_confidence=args.min_confidence)
    backends: Counter = Counter(str(item.get("backend") or "未记录") for item in scoped)

    sessions = sessions_of(records)
    print(f"trace：{args.trace}")
    print(f"读到 {len(records)} 轮对话、{len(sessions)} 个会话")
    if len(scoped) != len(records):
        scope_desc = f"backend={args.backend}" if args.backend else "session 筛选"
        print(f"本次范围：{len(scoped)} 轮（全量 {len(records)} 轮，{scope_desc}）")
    print("后端分布：" + "、".join(f"{name} {count} 轮" for name, count in backends.items()))
    print(
        "信号命中（轮）："
        + "、".join(f"{signal.label} {counts.get(signal.key, 0)}" for signal in SIGNALS)
    )
    if slot_skipped:
        print(
            f"（其中补槽位轮的低置信 {slot_skipped} 轮已跳过——碎片句标签不影响路由，"
            "要收进来加 --include-slot-turns）"
        )
    print(f"候选 badcase：{len(candidates)} 条（去重后）")

    if not candidates and len(scoped) != len(records):
        # 最常见的困惑：筛了一个后端，而候选全在另一个后端里
        unscoped = mine(records, limit=args.limit, **{k: v for k, v in kwargs.items()})
        if unscoped:
            print(
                f"提示：去掉筛选后全量数据里有 {len(unscoped)} 条候选——"
                "换后端采的那批候选在另一份数据里，别用 --backend/--session 把范围切没了。"
            )

    existing = load_pool(args.out)
    merged, merge_stats = merge_pool(existing, candidates)
    if merged:
        write_pool(args.out, merged)
        print(
            f"候选池已写入：{args.out}（本次新增 {merge_stats['added']}、"
            f"带回原标注 {merge_stats['carried']}、保留已标注未挖到 "
            f"{merge_stats['kept_labelled']}；池子共 {len(merged)} 条）"
        )
    else:
        if existing:
            print(f"本次没有可用候选，原池子保持不动：{args.out}（{len(existing)} 条仍在）")
        else:
            print(f"本次没有可用候选，未创建池子文件：{args.out}")
    if not args.no_report:
        write_report(
            args.report,
            trace_path=args.trace,
            records=records,
            candidates=candidates,
            counts=counts,
            backends=backends,
            slot_skipped=slot_skipped,
            backend=args.backend,
            scoped_turns=len(scoped),
            merge_stats=merge_stats,
            session=args.session,
            limit=args.limit,
        )
        print(f"报告已写入：{args.report}")
    if candidates:
        print(
            "\n下一步：打开候选池，填 expected_intent / expected_action / note 三列"
            "（status 改成 labeled），再用 --print-transcript 回放上下文复核。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
