"""发布门禁：一条命令决定"这个版本能不能上"，全绿才切 active。

串起三道检查：
1. **意图门禁**：候选版本 vs 现役版本，macro-F1 与三个关键类（quality_qa / chitchat /
   human_handoff）都不劣化才放行——只看 macro-F1 会让"高频类涨分、短板悄悄退化"的版本蒙混过关；
2. **全链路回归**：`scripts/regression.py`，检索/对话/judge 等指标都不超容差；
3. 两道全过 → 写 `data/intent/active.json`：运行时按它解析 adapter 目录，
   `GET /api/intent/versions` 与前端按它标 active。

红了不切：旧版本继续生效，退出码 1，报告里写清是哪一项拦下的。
把"记得跑、记得看、记得回滚"写成脚本，才有门禁可言。

用法（在 agent/ 目录下执行）：
    python scripts/release.py --candidate data/intent/models/v3               # 完整流程
    python scripts/release.py --candidate data/intent/models/v3 --skip-eval    # 评测行已存在
    python scripts/release.py --candidate data/intent/models/v3 --dry-run      # 只出决策卡
    python scripts/release.py --candidate data/intent/models/v2 --skip-eval --skip-regression
        # ↑ 负样本演练：验证门禁会拒绝 v2（历史事实：v2 的 human_handoff F1 从 0.6176 掉到 0.3111）
    python scripts/release.py --rollback                                       # 回滚到默认版本

退出码：0 = 放行（或 dry-run 判定可放行）；1 = 拒绝；2 = 环境/参数问题。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DATA_DIR  # noqa: E402
from agent.intent.active import ACTIVE_PATH, clear_active, load_active, set_active  # noqa: E402
from scripts.evolve_intent import evaluate_gate  # noqa: E402

REPORTS = AGENT_ROOT / "eval" / "reports"
DEFAULT_BASELINE_JSON = REPORTS / "intent_baseline.json"
DEFAULT_REGRESSION_JSON = REPORTS / "regression.json"
DEFAULT_REPORT = REPORTS / "release.md"
DEFAULT_JSON = REPORTS / "release.json"
DEFAULT_CANDIDATE = DATA_DIR / "intent" / "models" / "v3"
DEFAULT_BASELINE = DATA_DIR / "intent" / "models" / "v1"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


# ---------------- 找评测行：路径写法要宽容 ----------------


def _same_dir(left: str, right: str) -> bool:
    """两个路径是否指同一个 adapter 目录。

    评测 JSON 里存的是相对路径（data/intent/models/v1），命令行里可能传绝对路径
    （/mnt/d/.../models/v1），直接字符串比较会匹配不上——按"父目录名 + 目录名"兜底。
    """
    if not left or not right:
        return False
    a = Path(str(left).rstrip("/"))
    b = Path(str(right).rstrip("/"))
    if str(a) == str(b):
        return True
    return (a.parent.name, a.name) == (b.parent.name, b.name) and a.name not in {"", ".", ".."}


def find_row(payload: dict, local_dir: Path | str, split: str = "test") -> dict | None:
    for row in payload.get("backends") or []:
        if row.get("backend") != "local" or row.get("split") != split:
            continue
        if _same_dir(str(row.get("local_dir") or ""), str(local_dir)):
            return row
    return None


# ---------------- 跑子命令：实时透传输出 ----------------


def run_command(command: list[str], *, timeout: float | None = None) -> tuple[int, list[str]]:
    """跑子进程并把输出实时打出来；返回 (退出码, 最后若干行)。

    踩过的坑：用 capture_output 一次性收完，几十分钟的任务看起来像"卡死"。
    """
    print("  $ " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(AGENT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    tail: list[str] = []
    lock = threading.Lock()

    def pump() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip("\n")
            with lock:
                tail.append(text)
                if len(tail) > 200:
                    tail.pop(0)
            print("    " + text, flush=True)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        code = -1
        with lock:
            tail.append(f"（超时 {timeout}s，已终止）")
    reader.join(timeout=2)
    return code, list(tail)


# ---------------- 决策 ----------------


def decide(gate: dict, regression: dict | None, *, regression_required: bool) -> dict:
    """把两道检查合成一个结论；宁可拒绝，也不要"看起来跑完了"就放行。"""
    reasons: list[str] = []
    gate_ok = bool(gate.get("rows")) and bool(gate.get("passed"))
    if not gate.get("rows"):
        reasons.append(gate.get("reason") or "意图门禁缺少可比对的评测行")
    elif not gate.get("passed"):
        reasons.append(gate.get("reason") or "意图门禁不通过")
    # 门禁已经挂了就不用再提"没跑回归"——那只是省时间，不是新的问题
    if regression_required and gate_ok:
        if regression is None:
            reasons.append("没有跑全链路回归")
        elif not regression.get("passed"):
            failed = [
                row.get("label")
                for row in regression.get("comparison") or []
                if row.get("status") == "退化"
            ]
            reasons.append("全链路回归有红项：" + "、".join(str(item) for item in failed))
    return {"approved": not reasons, "reasons": reasons}


def regression_summary(payload: dict) -> dict:
    rows = payload.get("comparison") or []
    counts = Counter(str(row.get("status") or "未知") for row in rows)
    improved = [
        {
            "label": row.get("label"),
            "baseline": row.get("baseline"),
            "current": row.get("current"),
            "delta": row.get("delta"),
        }
        for row in rows
        if row.get("status") == "改善"
    ]
    failed = [row.get("label") for row in rows if row.get("status") == "退化"]
    return {"counts": dict(counts), "improved": improved, "failed": failed}


# ---------------- 输出 ----------------


def render_card(
    *,
    candidate: Path,
    baseline: Path,
    gate: dict,
    regression: dict | None,
    regression_source: str,
    decision: dict,
    active: dict | None,
    dry_run: bool,
) -> str:
    lines = [
        "# 发布决策卡",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 候选版本：{candidate}",
        f"- 现役版本：{baseline}",
        f"- 判定：{'✅ 放行' if decision['approved'] else '❌ 拒绝（保留现役版本）'}"
        + ("（dry-run，未实际切换）" if dry_run and decision["approved"] else ""),
    ]
    if decision["reasons"]:
        lines.append("- 拦下原因：" + "；".join(decision["reasons"]))

    lines += [
        "",
        "## 一、意图门禁（macro-F1 + 三个关键类都不劣化）",
        "",
    ]
    if gate.get("rows"):
        lines += [
            "| 指标 | 现役 | 候选 | Δ | 是否达标 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in gate["rows"]:
            lines.append(
                f"| {row['metric']} | {row['baseline']:.4f} | {row['candidate']:.4f} | "
                f"{row['delta']:+.4f} | {'✓' if row['ok'] else '✗'} |"
            )
    else:
        lines.append(f"无法比对：{gate.get('reason')}")

    lines += ["", "## 二、全链路回归", ""]
    if regression is None:
        lines.append(
            "本次没有跑全链路回归（不构成完整放行证据）"
            + ("：意图门禁已不通过，没必要再花时间跑。" if not gate.get("passed") else "。")
        )
    else:
        summary = regression_summary(regression)
        counts = summary["counts"]
        source_note = (
            f"（沿用 {regression.get('created_at') or '时间未知'} 的回归结果，不是本次新跑）"
            if regression_source == "reused"
            else ""
        )
        lines.append(
            f"- 指标状态{source_note}："
            + "、".join(f"{name} {count}" for name, count in sorted(counts.items()))
        )
        if summary["improved"]:
            lines.append(
                "- 改善项："
                + "、".join(
                    f"{row['label']}（{row['delta']:+.4f}）" for row in summary["improved"]
                )
            )
        if summary["failed"]:
            lines.append("- **退化项**：" + "、".join(str(item) for item in summary["failed"]))
        lines.append(f"- 结论：{'通过' if regression.get('passed') else '未通过'}")

    lines += ["", "## 三、切换与回滚", ""]
    if active:
        lines += [
            f"已把 **{active['version']}** 记为当前生效版本（{active['activated_at']}）。",
            "",
            "```json",
            json.dumps(active, ensure_ascii=False, indent=2),
            "```",
        ]
    elif decision["approved"]:
        lines.append("dry-run：本可以切换，但没有写入 active.json。")
    else:
        lines.append("没有切换：active.json 保持原样，现役版本继续服务。")
    lines += [
        "",
        "回滚（任何一次发布都可以用这一条回去）：",
        "",
        "```bash",
        "python scripts/release.py --rollback",
        "```",
        "",
        "> 口径提醒：这里的「影子对比」= 新旧版本跑同一份评测出对比卡，提供的是决策依据；"
        "它不等于真灰度的流量隔离（单机没有分流设施）。",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="发布门禁：意图门禁 + 全链路回归，全绿才切 active")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE, help="候选 adapter 目录")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help="现役 adapter 目录")
    parser.add_argument("--split", default="test", help="门禁用哪个集合（默认 test）")
    parser.add_argument("--tolerance", type=float, default=0.0, help="意图门禁允许的劣化幅度")
    parser.add_argument("--intent-baseline", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--regression-json", type=Path, default=DEFAULT_REGRESSION_JSON)
    parser.add_argument("--skip-eval", action="store_true", help="候选已有评测行，不重跑 eval_intent")
    parser.add_argument("--skip-regression", action="store_true", help="不跑全链路回归（只能配 --dry-run）")
    parser.add_argument(
        "--regression-only",
        nargs="*",
        default=None,
        help="只跑指定的回归阶段（透传给 regression.py --only）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只出决策卡，不写 active.json")
    parser.add_argument("--rollback", action="store_true", help="删掉 active.json，回滚到默认版本")
    parser.add_argument("--active-file", type=Path, default=ACTIVE_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    if args.rollback:
        previous = load_active(args.active_file) or {}
        removed = clear_active(args.active_file)
        if removed:
            print(f"已回滚：删掉 active.json（原版本 {previous.get('version') or '未知'}）")
            print(f"  {args.active_file}")
        else:
            print(f"没有可回滚的 active.json（{args.active_file}）")
        return 0

    candidate = args.candidate
    if not (candidate / "adapter_config.json").exists():
        print(f"[!] 候选目录里没有 adapter_config.json：{candidate}")
        print("    确认版本号对不对，或者先把它训出来。")
        return 2

    print(f"候选：{candidate}")
    print(f"现役：{args.baseline}（{args.split} 集）")

    # 1) 候选评测：没有可比对的评测行，门禁就是空转
    if not args.skip_eval:
        print("\n[1/3] 跑候选版本的意图评测（首次对新 adapter 约 10~20 分钟）…")
        code, _ = run_command(
            [
                sys.executable,
                str(AGENT_ROOT / "scripts" / "eval_intent.py"),
                "--backends",
                "local",
                "--local-dir",
                str(candidate),
                "--splits",
                args.split,
                "--report",
                str(REPORTS / "intent_baseline.md"),
                "--json",
                str(args.intent_baseline),
            ]
        )
        if code != 0:
            print("[!] 候选评测失败，发布中止")
            return 2
    else:
        print("\n[1/3] 跳过候选评测（--skip-eval）")

    # 2) 意图门禁
    print("\n[2/3] 意图门禁…")
    payload = load_json(args.intent_baseline)
    baseline_row = find_row(payload, args.baseline, args.split)
    candidate_row = find_row(payload, candidate, args.split)
    gate = evaluate_gate(
        baseline_row or {}, candidate_row or {}, tolerance=args.tolerance
    )
    for row in gate.get("rows") or []:
        print(
            f"  {'✓' if row['ok'] else '✗'} {row['metric']:<20} "
            f"{row['baseline']:.4f} → {row['candidate']:.4f}（{row['delta']:+.4f}）"
        )
    if not gate.get("rows"):
        print(f"  ✗ {gate.get('reason')}")

    # 3) 全链路回归
    regression: dict | None = None
    regression_source = "none"
    if not gate.get("passed"):
        # 意图门禁已经不通过，没必要再花几十分钟跑全链路回归——直接出拒绝卡
        print("\n[3/3] 意图门禁未通过，跳过全链路回归")
    elif args.skip_regression:
        # 复用上一次的回归结果：卡片里必须写清是哪一次的，否则等于拿旧证据放行新版本
        regression = load_json(args.regression_json) or None
        if regression:
            regression_source = "reused"
            print(
                f"\n[3/3] 沿用已有回归结果（{regression.get('created_at') or '时间未知'}）："
                f"{'通过' if regression.get('passed') else '未通过'}"
            )
        else:
            print("\n[3/3] 跳过全链路回归，且找不到可复用的回归结果")
    else:
        print("\n[3/3] 全链路回归（这一步最慢，域外阶段会占大头）…")
        command = [sys.executable, str(AGENT_ROOT / "scripts" / "regression.py")]
        if args.regression_only is not None:
            command += ["--only", *args.regression_only]
        code, _ = run_command(command)
        regression = load_json(args.regression_json) or None
        if code != 0:
            print(f"  ✗ 回归退出码 {code}（报告见 {args.regression_json}）")
        if regression:
            regression_source = "fresh"

    decision = decide(gate, regression, regression_required=True)

    active = None
    if decision["approved"] and not args.dry_run:
        active = set_active(
            candidate.name,
            candidate,
            evidence={
                "baseline": str(args.baseline),
                "split": args.split,
                "tolerance": args.tolerance,
                "gate": {"passed": gate.get("passed"), "rows": gate.get("rows")},
                "regression": regression_summary(regression) if regression else None,
                "regression_headline": (regression or {}).get("created_at"),
            },
            path=args.active_file,
        )

    card = render_card(
        candidate=candidate,
        baseline=args.baseline,
        gate=gate,
        regression=regression,
        regression_source=regression_source,
        decision=decision,
        active=active,
        dry_run=args.dry_run,
    )
    print("\n" + card)

    payload_out = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidate": str(candidate),
        "baseline": str(args.baseline),
        "decision": decision,
        "gate": gate,
        "regression": regression_summary(regression) if regression else None,
        "regression_source": regression_source,
        "active": active,
        "dry_run": args.dry_run,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(card, encoding="utf-8")
        print(f"决策卡：{args.report}")
    print(f"决策 JSON：{args.json}")

    if not decision["approved"]:
        print("\n结论：拒绝放行，现役版本继续生效。")
        return 1
    print(
        "\n结论：放行。"
        + ("（dry-run，未切换）" if args.dry_run else f"当前生效版本已切到 {candidate.name}。")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
