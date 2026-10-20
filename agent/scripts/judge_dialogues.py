"""端到端对话的 LLM-as-judge 打分 + 人工抽样校准。

为什么需要这一层：`eval_dialogues.py` 的判据是**确定性**的（调没调工具、回复里有没有
订单号、交接摘要里有没有带槽位），它只回答"行为对不对"，回答不了"话说到点子上没有、
用户的事到底办成没有"。这一层用大模型按 rubric 读完整段对话，给出
`达成 / 部分达成 / 未达成` + 一句理由，再由人工抽样复核，算 judge 与人的一致率。

两个刻意的设计：
1. **提示里不放我们的判据结论**（checks / resolved_predicted 都不给），只给用户话、
   客服回复、工具记录、转人工与交接摘要。否则 judge 只是在抄规则版的答案，
   一致率再高也没有信息量。
2. **人工校准必须真的做**：judge 的一致率只有对着人的标注才有意义，所以脚本把
   "抽样 → 人工填判定 → 算 kappa" 做成两条命令，样本优先选规则版与标注分歧的那几条。

用法（在 agent/ 目录下执行）：
    python scripts/judge_dialogues.py --dry-run                  # 不调 API，先验证流水线
    python scripts/judge_dialogues.py                            # 正式打分（默认读 dialogues.json）
    python scripts/judge_dialogues.py --input eval/reports/dialogues_local.json
    python scripts/judge_dialogues.py --export-review eval/reports/judge_review.jsonl --sample 12
    python scripts/judge_dialogues.py --review eval/reports/judge_review.jsonl

成本：默认并发 4、每条一次调用，30 段对话 ≈ 30 次调用、几毛钱；
缓存 `eval/reports/judge_cache.json` 的键含 (模型, 提示词版本, 对话内容)，
改 rubric 时把 PROMPT_VERSION 一起改掉，缓存自动失效。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from hashlib import sha256
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.config import get_env, get_float  # noqa: E402
from eval.metrics import (  # noqa: E402
    cohens_kappa,
    cost_per_1k,
    estimate_cost,
    label_agreement,
    latency_stats,
)

DEFAULT_INPUT = AGENT_ROOT / "eval" / "reports" / "dialogues.json"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "dialogues_judge.md"
DEFAULT_JSON = AGENT_ROOT / "eval" / "reports" / "dialogues_judge.json"
DEFAULT_CACHE = AGENT_ROOT / "eval" / "reports" / "judge_cache.json"

# 三档判定：达成 / 部分达成 / 未达成。改 rubric 时同步改版本号，缓存才会失效。
VERDICTS: tuple[str, ...] = ("达成", "部分达成", "未达成")
PROMPT_VERSION = "judge-v2"

# 判定阈值：人工一致率低于这个数就说明 rubric 还得改（第三段验收线）
AGREEMENT_TARGET = 0.8

SYSTEM_PROMPT = """你是生鲜电商售后客服的质检员。给你一段客服对话的完整记录，以及这条对话的"判定标准"，请你判定用户的目标有没有被达成。

判定以**这条对话的「判定标准」**为准，只看用户能不能拿到结果、会不会被误导；不看篇幅长短、语气是否客气、有没有多说安慰话。

三档含义：
- 达成：判定标准里写明的要求都做到了（给了结论 / 建了工单 / 答了带依据的答案）；或者标准写明这一步本来就该是"追问缺失信息 / 礼貌兜底 / 转人工"，而客服确实这么做了。
- 部分达成：只办了一部分——用户一次说了两件事只办了其中一件；给了方向但没给结论；标准要求给结果，客服只追问了；整段对话结束时事情仍然悬着。
- 未达成：答非所问、编造订单或政策信息、该办的事没办、该转人工没转、把工具报错当成答案糊弄用户。

三条补充口径（用它们解决标准写得不够细时的歧义，务必按顺序套用）：
1. **「理想 / 最好 / 若能 / 建议」是加分项，不是门槛**。标准里这类措辞描述的部分没做到，不影响判"达成"；反过来，标准里写明的要求只要有一条没做到，就不能判"达成"。例：标准写"不硬答、不误建工单；理想是确认是否还要办理"，客服做到了前两项、没做第三项 → 仍判"达成"。
2. **追问 ≠ 给结果**。标准把"追问缺失信息"写成期望行为时，追问了就是"达成"（例如查单缺订单号，先问订单号本来就是正确处置）；标准要求的是给出结论时，只追问没给结论最多算"部分达成"。
3. **会让用户误解关键信息时降档**。用户在一条消息里提了两件以上的事、而回复没有指明正在处理哪一件（用户无法判断在问哪一单、哪件商品），最高只能判"部分达成"。注意与口径 1 的边界：标准里写成"理想 / 最好"的改进项不据此降档。

如果对话中途被转交人工、且判定标准认为转人工本身就是正确处置，算"达成"。

只输出一个 JSON 对象，不要解释、不要 Markdown 代码块，格式严格为：
{"verdict": "达成", "reason": "一句话说清依据，不超过 40 字"}
verdict 只能取 "达成" / "部分达成" / "未达成" 三者之一。
"""


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------- 纯函数：渲染、解析、判分口径（都可单测） ----------------


def render_dialogue(record: dict) -> str:
    """把一条对话渲染成 judge 的输入。

    只给"用户话 / 客服回复 / 工具记录 / 转人工与交接摘要"，**不给**我们的判据结论
    （checks、resolved_predicted、分支、意图这些内部字段都不出现），避免 judge 抄答案。
    """
    lines = [f"【场景】{record.get('scenario') or record.get('id')}"]
    criteria = record.get("resolution_criteria")
    if criteria:
        lines.append(f"【本条判定标准】{criteria}")
    if record.get("note"):
        lines.append(f"【已知说明】{record['note']}")

    lines.append("【对话记录】")
    for index, turn in enumerate(record.get("turn_log") or [], start=1):
        lines.append(f"第 {index} 轮")
        lines.append(f"用户：{turn.get('user') or ''}")
        lines.append(f"客服：{turn.get('reply') or ''}")

    tools = record.get("tools") or {}
    if tools.get("calls"):
        by_tool = "、".join(f"{name}×{count}" for name, count in (tools.get("by_tool") or {}).items())
        lines.append(
            f"【工具调用】共 {tools['calls']} 次（{by_tool or '未记录'}）："
            f"成功 {tools.get('success', 0)}、降级（改为追问或转人工）{tools.get('degraded', 0)}、"
            f"失败 {tools.get('failed', 0)}"
        )
    else:
        lines.append("【工具调用】无")

    facts = record.get("facts") or {}
    if facts.get("handoff"):
        summary = (facts.get("handoff_summary") or "").strip() or "（未生成交接摘要）"
        lines.append(f"【是否转人工】是；交接摘要：{summary}")
    else:
        lines.append("【是否转人工】否")
    return "\n".join(lines)


def build_messages(record: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_dialogue(record)},
    ]


def parse_verdict(raw: str) -> tuple[str, str]:
    """解析 judge 输出；返回 (verdict, reason)，解析失败抛 ValueError。

    容错两类常见偏差：```json 代码块包裹；verdict 写成"完全达成""未达成。"这种带修饰的串。
    注意匹配顺序：先"未达成"再"部分达成"最后"达成"，否则"部分达成"会被"达成"抢先命中。
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"模型未返回 JSON：{text[:80]}")
    payload = json.loads(text[start : end + 1])

    raw_verdict = str(payload.get("verdict") or "").strip()
    verdict = raw_verdict if raw_verdict in VERDICTS else ""
    if not verdict:
        verdict = next((item for item in ("未达成", "部分达成", "达成") if item in raw_verdict), "")
    if not verdict:
        raise ValueError(f"判定不在允许集合内：{raw_verdict!r}")
    return verdict, str(payload.get("reason") or "")[:120]


def verdict_to_binary(verdict: str) -> str:
    """把三档压成二值，好和规则版初判（判据是否全过）比对。"""
    return "达成" if verdict == "达成" else "未达成"


def dry_run_verdict(record: dict) -> tuple[str, str]:
    """离线模式：直接用确定性判据推一个结果，只为验证流水线，不代表 judge 的判断。

    没装 openai / 没配 key 时也能把报告跑出来，是这个模式存在的唯一理由。
    """
    checks = record.get("checks") or {}
    if checks and all(checks.values()):
        return "达成", "dry-run：确定性判据全部通过"
    if any(checks.values()):
        return "部分达成", "dry-run：部分判据未通过"
    return "未达成", "dry-run：判据未通过或没有判据"


def cache_key(record: dict, model: str) -> str:
    """缓存键：模型 + 提示词版本 + 这条对话的渲染结果。"""
    payload = json.dumps(
        [PROMPT_VERSION, model, render_dialogue(record)],
        ensure_ascii=False,
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


# ---------------- 调用 ----------------


def call_llm(messages: list[dict], model: str, timeout: float, max_tokens: int = 200) -> tuple[str, int, int]:
    """调一次 judge；返回 (文本, prompt_tokens, completion_tokens)。"""
    from openai import OpenAI

    client = OpenAI(
        api_key=get_env("LLM_API_KEY", ""),
        base_url=get_env("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        timeout=timeout,
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    usage = getattr(response, "usage", None)
    return (
        response.choices[0].message.content or "",
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


def judge_record(
    record: dict,
    *,
    model: str,
    timeout: float,
    cache: dict | None = None,
    dry_run: bool = False,
    price_in: float = 2.0,
    price_out: float = 8.0,
    complete_fn=call_llm,
) -> dict:
    """给一条对话打分，返回一条完整记录（含 token / 时延 / 成本）。

    解析失败或调用异常都不抛出去：写进 `error` 字段，让上层统计"失败条数"而不是中断整轮评测。
    """
    result = {
        "id": record.get("id"),
        "verdict": "",
        "reason": "",
        "latency_ms": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": 0.0,
        "model": "dry-run" if dry_run else model,
        "prompt_version": PROMPT_VERSION,
        "cached": False,
        "error": "",
        "raw": "",
    }

    key = cache_key(record, model)
    if cache is not None and key in cache:
        cached = dict(cache[key])
        cached["cached"] = True
        cached["id"] = record.get("id")
        return cached

    if dry_run:
        verdict, reason = dry_run_verdict(record)
        result.update({"verdict": verdict, "reason": reason})
    else:
        started = time.perf_counter()
        try:
            raw, prompt_tokens, completion_tokens = complete_fn(
                build_messages(record), model, timeout
            )
        except Exception as exc:  # noqa: BLE001 - 单条失败不该中断整轮
            result["error"] = f"{type(exc).__name__}: {exc}"
            return result
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        result["prompt_tokens"] = prompt_tokens
        result["completion_tokens"] = completion_tokens
        result["raw"] = raw
        result["cost"] = estimate_cost(prompt_tokens, completion_tokens, price_in, price_out)
        try:
            verdict, reason = parse_verdict(raw)
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"解析失败：{exc}"
            return result
        result.update({"verdict": verdict, "reason": reason})

    if cache is not None:
        cache[key] = {**result, "cached": False}
    return result


def run_judge(
    records: list[dict],
    *,
    model: str,
    timeout: float,
    concurrency: int,
    cache: dict | None = None,
    dry_run: bool = False,
    price_in: float = 2.0,
    price_out: float = 8.0,
    complete_fn=call_llm,
    progress_every: int = 10,
) -> list[dict]:
    """批量打分：并发跑，但结果顺序按输入顺序返回，保证报告可复现。"""
    results: list[dict | None] = [None] * len(records)
    done = 0

    def work(index: int) -> tuple[int, dict]:
        return index, judge_record(
            records[index],
            model=model,
            timeout=timeout,
            cache=cache,
            dry_run=dry_run,
            price_in=price_in,
            price_out=price_out,
            complete_fn=complete_fn,
        )

    if concurrency <= 1 or len(records) <= 1:
        for index in range(len(records)):
            _, payload = work(index)
            results[index] = payload
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"    已打分 {done}/{len(records)} 条")
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(work, index) for index in range(len(records))]
            for future in as_completed(futures):
                index, payload = future.result()
                results[index] = payload
                done += 1
                if progress_every and done % progress_every == 0:
                    print(f"    已打分 {done}/{len(records)} 条")
    return [item for item in results if item is not None]


# ---------------- 指标 ----------------


def summarize_judge(records: list[dict], results: list[dict]) -> dict:
    """judge 结果汇总：分布、与规则版初判的一致率、差异清单、成本。"""
    by_id = {record.get("id"): record for record in records}
    judged = [item for item in results if item.get("verdict")]
    failed = [item for item in results if not item.get("verdict")]

    counts = {verdict: sum(1 for item in judged if item["verdict"] == verdict) for verdict in VERDICTS}
    total = len(judged)

    def rate(value: int) -> float:
        return round(value / total, 4) if total else 0.0

    # 与规则版初判（确定性判据是否全过）逐条比对。口径不同是正常的，
    # 差异大的那几条正是"rubric 或标注该改"的线索。
    comparable = [item for item in judged if item["id"] in by_id]
    judge_labels = [verdict_to_binary(item["verdict"]) for item in comparable]
    rule_labels = [
        "达成" if by_id[item["id"]].get("resolved_predicted") else "未达成" for item in comparable
    ]

    rule_said_ok = [
        item["id"]
        for item in comparable
        if item["verdict"] != "达成" and by_id[item["id"]].get("resolved_predicted")
    ]
    judge_said_ok = [
        item["id"]
        for item in comparable
        if item["verdict"] == "达成" and not by_id[item["id"]].get("resolved_predicted")
    ]

    latencies = [item["latency_ms"] for item in judged if not item.get("cached") and item["latency_ms"]]
    costs = [item["cost"] for item in judged if not item.get("cached")]
    latency = latency_stats(latencies)

    return {
        "dialogues": len(results),
        "judged": total,
        "failed": len(failed),
        "cached": sum(1 for item in results if item.get("cached")),
        "verdict_counts": counts,
        "resolved_rate": rate(counts["达成"]),
        "partial_rate": rate(counts["部分达成"]),
        "unresolved_rate": rate(counts["未达成"]),
        "judge_vs_rule_agreement": round(label_agreement(judge_labels, rule_labels), 4)
        if comparable
        else 0.0,
        "judge_vs_rule_kappa": cohens_kappa(judge_labels, rule_labels) if comparable else 0.0,
        "rule_ok_judge_not": rule_said_ok,
        "judge_ok_rule_not": judge_said_ok,
        "latency_p95_ms": latency.get("p95_ms", 0.0),
        "latency_avg_ms": latency.get("mean_ms", 0.0),
        "cost_total": round(sum(costs), 6),
        "cost_per_1k": cost_per_1k(costs),
    }


def group_judge(records: list[dict], results: list[dict], key: str) -> list[tuple[str, dict]]:
    """按链路/样本性质分组统计 judge 结果。"""
    by_id = {record.get("id"): record for record in records}
    groups: dict[str, list[dict]] = {}
    for item in results:
        if not item.get("verdict") or item["id"] not in by_id:
            continue
        groups.setdefault(by_id[item["id"]].get(key) or "-", []).append(item)

    rows: list[tuple[str, dict]] = []
    for name in sorted(groups):
        items = groups[name]
        counts = {verdict: sum(1 for item in items if item["verdict"] == verdict) for verdict in VERDICTS}
        agreed = sum(
            1
            for item in items
            if verdict_to_binary(item["verdict"])
            == ("达成" if by_id[item["id"]].get("resolved_predicted") else "未达成")
        )
        rows.append(
            (
                name,
                {
                    "dialogues": len(items),
                    **counts,
                    "judge_vs_rule_agreement": round(agreed / len(items), 4),
                },
            )
        )
    return rows


# ---------------- 人工校准 ----------------


def sample_for_review(
    records: list[dict],
    results: list[dict],
    sample: int,
    seed: int = 42,
) -> list[dict]:
    """挑人工复核样本：优先"规则版与标注分歧"的对话，其余按链路分层补齐。

    优先挑分歧样本，是因为一致率高不高取决于"难的那几条"；只抽简单样本，
    一致性会虚高。分层补样是为了让每条链路都被看到。
    """
    if sample <= 0:
        return []
    verdicts = {item["id"]: item for item in results}
    picked: list[dict] = []
    picked_ids: set[str] = set()

    def take(record: dict) -> None:
        if record["id"] in picked_ids:
            return
        picked.append(record)
        picked_ids.add(record["id"])

    # ① 判据未通过：真失败一定要人看
    for record in records:
        if any(not ok for ok in (record.get("checks") or {}).values()):
            take(record)
    # ② 判据全过但与标注口径不一致（报告里的"标注待复核"）
    for record in records:
        if len(picked) >= sample:
            break
        checks = record.get("checks") or {}
        checks_ok = all(checks.values()) if checks else False
        if checks_ok and (not record.get("resolved_match") or not record.get("handoff_ok")):
            take(record)

    # ③ 其余按链路分层，用固定 seed 保证可复现
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for record in records:
        if record["id"] in picked_ids:
            continue
        buckets.setdefault(record.get("chain") or "-", []).append(record)
    for items in buckets.values():
        rng.shuffle(items)
    chains = sorted(buckets)
    while len(picked) < sample and any(buckets[name] for name in chains):
        for name in chains:
            if len(picked) >= sample:
                break
            if buckets[name]:
                take(buckets[name].pop())

    # ④ 给每条补上 judge 的判定与判分依据，方便人工对照
    rows = []
    for record in picked:
        item = verdicts.get(record["id"]) or {}
        rows.append(
            {
                "id": record["id"],
                "scenario": record.get("scenario"),
                "chain": record.get("chain"),
                "category": record.get("category"),
                "resolution_criteria": record.get("resolution_criteria"),
                "note": record.get("note"),
                "dialogue": [
                    {"user": turn.get("user"), "reply": turn.get("reply")}
                    for turn in (record.get("turn_log") or [])
                ],
                "rule_verdict": "达成" if record.get("resolved_predicted") else "未达成",
                "rule_note": "判据全过" if record.get("resolved_predicted") else "判据未过",
                "judge_verdict": item.get("verdict", ""),
                "judge_reason": item.get("reason", ""),
                "human_verdict": "",
                "human_note": "",
            }
        )
    return rows


def calibrate(review_rows: list[dict]) -> dict:
    """算 judge 与人工的一致率 / Cohen's kappa，并列出不一致的 case。"""
    pairs = [
        row
        for row in review_rows
        if row.get("human_verdict") in VERDICTS and row.get("judge_verdict") in VERDICTS
    ]
    judge = [row["judge_verdict"] for row in pairs]
    human = [row["human_verdict"] for row in pairs]
    disagreements = [
        {
            "id": row["id"],
            "scenario": row.get("scenario"),
            "human_verdict": row["human_verdict"],
            "judge_verdict": row["judge_verdict"],
            "judge_reason": row.get("judge_reason", ""),
            "human_note": row.get("human_note", ""),
        }
        for row in pairs
        if row["human_verdict"] != row["judge_verdict"]
    ]
    agreement = label_agreement(judge, human) if pairs else 0.0
    return {
        "review_rows": len(review_rows),
        "compared": len(pairs),
        "agreement": round(agreement, 4),
        "kappa": cohens_kappa(judge, human) if pairs else 0.0,
        "passed": bool(pairs) and agreement >= AGREEMENT_TARGET,
        "target": AGREEMENT_TARGET,
        "disagreements": disagreements,
        "pending": len(review_rows) - len(pairs),
    }


def refresh_review(review_rows: list[dict], results: list[dict]) -> tuple[list[dict], int]:
    """把复核文件里的 judge 判定刷成**本次**跑出来的结果，人工判定原样保留。

    为什么需要：改完 rubric 重跑 judge 之后，复核文件里还存着上一版 judge 的判定，
    直接拿它算一致率等于拿旧结果校准新 rubric。人工填的 human_verdict / human_note
    是花时间换来的，必须留住，所以只刷 judge 那几列。
    返回 (刷新后的行, 刷新的条数)。
    """
    latest = {item["id"]: item for item in results if item.get("verdict")}
    refreshed = 0
    rows: list[dict] = []
    for row in review_rows:
        item = latest.get(row.get("id"))
        if item is None:
            rows.append(row)
            continue
        if row.get("judge_verdict") != item["verdict"] or row.get("judge_reason") != item.get("reason"):
            row = {
                **row,
                "judge_verdict": item["verdict"],
                "judge_reason": item.get("reason", ""),
            }
            refreshed += 1
        rows.append(row)
    return rows, refreshed


# ---------------- 输出 ----------------


def print_summary(metrics: dict, model: str) -> None:
    print(
        f"\njudge 后端 {model}：{metrics['judged']}/{metrics['dialogues']} 条打分成功"
        f"（失败 {metrics['failed']}、命中缓存 {metrics['cached']}）"
    )
    print(
        f"  达成 {metrics['verdict_counts']['达成']}（{metrics['resolved_rate']:.4f}）、"
        f"部分达成 {metrics['verdict_counts']['部分达成']}（{metrics['partial_rate']:.4f}）、"
        f"未达成 {metrics['verdict_counts']['未达成']}（{metrics['unresolved_rate']:.4f}）"
    )
    print(
        f"  与规则版初判一致率 {metrics['judge_vs_rule_agreement']:.4f}"
        f"（kappa {metrics['judge_vs_rule_kappa']:.4f}）"
    )
    if metrics["rule_ok_judge_not"]:
        print(f"  规则说过了、judge 说没达成：{'、'.join(metrics['rule_ok_judge_not'])}")
    if metrics["judge_ok_rule_not"]:
        print(f"  judge 说达成、规则判据没过：{'、'.join(metrics['judge_ok_rule_not'])}")
    print(
        f"  P95 时延 {metrics['latency_p95_ms']:.0f}ms；本次成本 ¥{metrics['cost_total']:.4f}"
        f"（千次 ¥{metrics['cost_per_1k']:.4f}）"
    )


def write_report(
    path: Path,
    *,
    input_path: Path,
    model: str,
    dry_run: bool,
    records: list[dict],
    results: list[dict],
    metrics: dict,
    calibration: dict | None,
) -> None:
    by_id = {record.get("id"): record for record in records}
    judge_line = (
        f"- judge：{model}（rubric 版本 {PROMPT_VERSION}，temperature 0）"
        + ("；**dry-run：结果由确定性判据推导，不代表 judge 的真实判断**" if dry_run else "")
    )
    lines = [
        "# 端到端对话 LLM-as-judge 报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 输入：{input_path.name}（{len(records)} 段对话）",
        judge_line,
        "",
        "## 一、judge 判定分布",
        "",
        "| 判定 | 条数 | 占比 |",
        "| --- | --- | --- |",
        f"| 达成 | {metrics['verdict_counts']['达成']} | {metrics['resolved_rate']:.4f} |",
        f"| 部分达成 | {metrics['verdict_counts']['部分达成']} | {metrics['partial_rate']:.4f} |",
        f"| 未达成 | {metrics['verdict_counts']['未达成']} | {metrics['unresolved_rate']:.4f} |",
        f"| 打分失败 | {metrics['failed']} | - |",
        "",
        f"成本：本次 ¥{metrics['cost_total']:.4f}，千次约 ¥{metrics['cost_per_1k']:.4f}；"
        f"P95 时延 {metrics['latency_p95_ms']:.0f}ms。",
        "",
        "## 二、按链路分组",
        "",
        "| 链路 | 对话数 | 达成 | 部分达成 | 未达成 | 与规则版初判一致率 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for chain, stats in group_judge(records, results, "chain"):
        lines.append(
            f"| {chain} | {stats['dialogues']} | {stats['达成']} | {stats['部分达成']} | "
            f"{stats['未达成']} | {stats['judge_vs_rule_agreement']:.4f} |"
        )

    lines += [
        "",
        "## 三、与规则版初判的差异",
        "",
        f"一致率 {metrics['judge_vs_rule_agreement']:.4f}（kappa {metrics['judge_vs_rule_kappa']:.4f}）。",
        "两者口径本来就不一样：规则版看「行为是不是都做对了」，judge 看「用户的事办成没有」。",
        "差异集中出现的形态就是该改 rubric 或该改标注的地方。",
        "",
        "> 读法提醒：一致率高不代表两层在测同一件事。如果 judge 的判定几乎全是「达成」"
        "（分布一边倒），kappa 会退化成 0——因为「随机一致」本身接近 1，做不出扣减，"
        "这时看一致率就够了。真正说明 judge 有价值的，是它挑出「判据全过、但用户没拿到结果」的那几条。",
        "> "
        "成本/时延显示 0 通常是本次全部命中缓存（缓存键含 rubric 版本，改 rubric 后会重新调用）。",
        "",
    ]
    for title, ids in (
        ("规则说判据全过、judge 说没达成", metrics["rule_ok_judge_not"]),
        ("judge 说达成、规则判据没过", metrics["judge_ok_rule_not"]),
    ):
        lines.append(f"**{title}**（{len(ids)} 条）")
        lines.append("")
        if not ids:
            lines += ["无。", ""]
            continue
        lines += ["| id | 场景 | judge 判定 | judge 理由 |", "| --- | --- | --- | --- |"]
        for item_id in ids:
            item = next((row for row in results if row["id"] == item_id), {})
            record = by_id.get(item_id) or {}
            lines.append(
                f"| {item_id} | {record.get('scenario')} | {item.get('verdict')} | "
                f"{item.get('reason', '')} |"
            )
        lines.append("")

    lines += ["## 四、逐条明细", "", "| id | 链路 | 判定 | 规则判据 | 理由 |", "| --- | --- | --- | --- | --- |"]
    for item in results:
        record = by_id.get(item["id"]) or {}
        rule = "全过" if record.get("resolved_predicted") else "未过"
        verdict = item.get("verdict") or f"失败（{item.get('error')}）"
        lines.append(
            f"| {item['id']} | {record.get('chain')} | {verdict} | {rule} | "
            f"{item.get('reason', '') or '-'} |"
        )

    lines += ["", "## 五、人工校准", ""]
    if calibration is None:
        lines += [
            "本次没有跑人工校准。两步就能补上：",
            "",
            "```bash",
            "python scripts/judge_dialogues.py --export-review eval/reports/judge_review.jsonl --sample 12",
            "# 打开 jsonl，给每条填上 human_verdict（达成/部分达成/未达成）与 human_note",
            "python scripts/judge_dialogues.py --review eval/reports/judge_review.jsonl",
            "```",
        ]
    else:
        lines += [
            f"- 抽样 {calibration['review_rows']} 条，其中 {calibration['compared']} 条已填人工判定"
            f"（未填 {calibration['pending']} 条，不参与一致性计算）",
            f"- judge 与人工一致率 **{calibration['agreement']:.4f}**"
            f"（门槛 {calibration['target']:.2f}：{'通过' if calibration['passed'] else '未通过，先改 rubric 再跑'}）",
            f"- Cohen's kappa：{calibration['kappa']:.4f}",
            "",
        ]
        if calibration["disagreements"]:
            lines += [
                "不一致的 case（先看是 judge 错还是标注/标准本身有歧义）：",
                "",
                "| id | 场景 | 人工 | judge | judge 理由 | 人工备注 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for row in calibration["disagreements"]:
                lines.append(
                    f"| {row['id']} | {row['scenario']} | {row['human_verdict']} | "
                    f"{row['judge_verdict']} | {row['judge_reason']} | {row['human_note'] or '-'} |"
                )
        else:
            lines.append("没有不一致的 case。")

    lines += [
        "",
        "## 六、怎么读这份报告",
        "",
        "1. **judge 分布**看「整段对话办成事的比例」，它和规则版的判据通过率不是一个东西；",
        "2. **与规则版的差异**里，「规则全过但 judge 说没达成」通常意味着判据写得太表面"
        "（比如回复里只要出现订单号就算过）；反向差异通常是 judge 被客气的措辞骗了，"
        "两条都值得拿具体 case 去改 rubric；",
        "3. **人工一致率**才是这一层的验收指标：低于 0.8 就先改 rubric，不要拿 judge 的数字下结论；",
        "4. dry-run 的结果不能当结论，只能用来验证流水线通了。",
        "",
        "复现命令：",
        "",
        "```bash",
        f"python scripts/judge_dialogues.py --input {input_path}",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(
    path: Path,
    *,
    input_path: Path,
    model: str,
    dry_run: bool,
    metrics: dict,
    results: list[dict],
    calibration: dict | None,
) -> None:
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(input_path),
        "model": "dry-run" if dry_run else model,
        "prompt_version": PROMPT_VERSION,
        "dry_run": dry_run,
        "metrics": metrics,
        "calibration": calibration,
        "records": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_review_file(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="端到端对话的 LLM-as-judge 打分与人工校准")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="eval_dialogues.py 产出的 JSON")
    parser.add_argument("--model", default="", help="覆盖 JUDGE_MODEL（默认沿用 LABEL_MODEL）")
    parser.add_argument("--concurrency", type=int, default=int(get_env("JUDGE_CONCURRENCY", "4") or 4))
    parser.add_argument("--timeout", type=float, default=get_float("JUDGE_TIMEOUT", 60.0))
    parser.add_argument("--limit", type=int, default=0, help="最多打分多少条（0=全部）")
    parser.add_argument("--only", nargs="*", default=None, help="只打分指定 id，调试用")
    parser.add_argument("--dry-run", action="store_true", help="不调 API，用判据推一个结果验证流水线")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--export-review", type=Path, default=None, help="导出人工复核样本（jsonl）")
    parser.add_argument("--sample", type=int, default=12, help="人工复核样本条数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review", type=Path, default=None, help="填好 human_verdict 后的复核文件")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"找不到对话评测结果：{args.input}\n"
            "先跑：python scripts/eval_dialogues.py --backend local "
            "--json eval/reports/dialogues_local.json --report eval/reports/dialogues_local.md"
        )
    payload = load_json(args.input)
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not records:
        raise SystemExit(f"{args.input} 里没有对话记录")
    if args.only:
        wanted = set(args.only)
        records = [record for record in records if record.get("id") in wanted]
    if args.limit:
        records = records[: args.limit]

    model = args.model or get_env("JUDGE_MODEL", "") or get_env("LABEL_MODEL", "deepseek-chat")
    price_in = get_float("JUDGE_PRICE_IN", 2.0)
    price_out = get_float("JUDGE_PRICE_OUT", 8.0)

    print(f"对话集：{args.input.name}（{len(records)} 段）")
    print(f"judge 模型：{'dry-run（不调用 API）' if args.dry_run else model}；rubric 版本 {PROMPT_VERSION}；并发 {args.concurrency}")
    if not args.dry_run:
        print(f"计价假设：输入 ¥{price_in}/百万、输出 ¥{price_out}/百万（可用 JUDGE_PRICE_* 覆盖）")

    cache: dict | None = None
    if not args.dry_run:
        if args.no_cache:
            print("缓存：已禁用（--no-cache）")
        else:
            cache = {}
            if args.cache.exists():
                cache = load_json(args.cache)
            print(f"缓存：{args.cache}（已有 {len([k for k in cache if not k.startswith('_')])} 条）")

    results = run_judge(
        records,
        model=model,
        timeout=args.timeout,
        concurrency=args.concurrency,
        cache=cache,
        dry_run=args.dry_run,
        price_in=price_in,
        price_out=price_out,
    )

    if cache is not None and not args.no_cache:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"缓存已更新：{args.cache}")

    metrics = summarize_judge(records, results)
    print_summary(metrics, "dry-run" if args.dry_run else model)

    calibration = None
    if args.review:
        review_rows = load_jsonl(args.review)
        review_rows, refreshed = refresh_review(review_rows, results)
        if refreshed:
            # 改过 rubric 之后重跑：复核文件里的 judge 列会过期，这里连同文件一起刷新
            write_review_file(args.review, review_rows)
            print(f"\n复核文件里的 judge 判定已刷成本次结果（{refreshed} 条）：{args.review}")
        calibration = calibrate(review_rows)

    if args.export_review:
        rows = sample_for_review(records, results, args.sample, args.seed)
        write_review_file(args.export_review, rows)
        print(
            f"\n人工复核样本已写入：{args.export_review}（{len(rows)} 条）\n"
            "  打开它给每条填 human_verdict（达成/部分达成/未达成）与 human_note，然后跑：\n"
            f"  python scripts/judge_dialogues.py --review {args.export_review}"
        )

    if not args.no_report:
        write_report(
            args.report,
            input_path=args.input,
            model=model,
            dry_run=args.dry_run,
            records=records,
            results=results,
            metrics=metrics,
            calibration=calibration,
        )
        print(f"\n报告已写入：{args.report}")
    write_json(
        args.json,
        input_path=args.input,
        model=model,
        dry_run=args.dry_run,
        metrics=metrics,
        results=results,
        calibration=calibration,
    )
    print(f"指标 JSON：{args.json}")

    if calibration is not None:
        if calibration["passed"]:
            print(f"人工校准：一致率 {calibration['agreement']:.4f} ≥ {AGREEMENT_TARGET}，通过")
        else:
            print(
                f"人工校准：一致率 {calibration['agreement']:.4f} 未达标"
                f"（已比 {calibration['compared']} 条）；先看不一致明细，改 rubric 或改标注，再跑一遍"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
