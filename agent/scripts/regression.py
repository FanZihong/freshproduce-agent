"""自动回归：一条命令跑完全部评测，和上一次的结果逐项对比，红了就指出是哪几个文件变的。

它不重算任何指标，只做三件事：
1. **串流程**：按顺序调用已有的评测脚本（检索 → 意图 → 域外 → 多意图 → 对话 → judge），
   每个脚本照常写自己的报告，回归脚本只读它们产出的 JSON；
2. **对齐口径**：把各层 JSON 里关心的指标抓成一个扁平表，附上方向（越高越好 / 越低越好）
   与容差；
3. **对比门禁**：和 `regression_baseline.json` 逐项比，超出容差就是红项，并给出
   "这次改动了哪些文件 / 哪个 commit" 的归因信息。

为什么要"容差"而不是"一模一样"：大模型、向量检索、并发调度都带随机性，
同代码两次跑出来的 macro-F1 也会差个零点几个点。容差就是"这个范围内的波动不算退化，
超过了才当真"。默认值写在 METRIC_SPECS 里，每条都有理由。

用法（在 agent/ 目录下执行）：
    python scripts/regression.py --dry-run              # 只打印将要执行的命令，不跑
    python scripts/regression.py                        # 全套（judge 走缓存，秒级）
    python scripts/regression.py --no-judge             # 跳过花钱的 judge 层
    python scripts/regression.py --only retrieval intent
    python scripts/regression.py --update-baseline      # 确认没有退化后，把本次记为基线

退出码：0 = 全部指标在容差内；1 = 有红项或某个阶段跑挂了（CI / pre-commit 可以直接用）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

REPORTS = AGENT_ROOT / "eval" / "reports"
DEFAULT_JSON = REPORTS / "regression.json"
DEFAULT_REPORT = REPORTS / "regression.md"
DEFAULT_BASELINE = REPORTS / "regression_baseline.json"

# 归因用的代码指纹范围：这些目录下的 .py 都算"代码改动"
FINGERPRINT_DIRS: tuple[str, ...] = ("agent", "scripts", "tests")


# ---------------- 指标口径 ----------------


@dataclass(frozen=True)
class MetricSpec:
    """一个指标的判读口径。

    direction：up=越高越好、down=越低越好、info=只记录不进门槛（成本、时延这类看情况定的）。
    tolerance / rel_tolerance：允许的波动，取两者中更宽松的那个（绝对值与相对值都支持，
    因为准确率适合绝对容差、时延适合相对容差）。
    """

    key: str
    label: str
    stage: str
    direction: str
    tolerance: float = 0.0
    rel_tolerance: float = 0.0
    note: str = ""


METRIC_SPECS: tuple[MetricSpec, ...] = (
    # 检索：30 条问句，recall@5 已经打满，所以只看"能不能守住"
    MetricSpec(
        "retrieval.hybrid+rw+score.recall@5",
        "检索 recall@5（生产口径：混合+改写+分数重排）",
        "retrieval",
        "up",
        0.01,
    ),
    MetricSpec(
        "retrieval.hybrid+rw+score.recall@1",
        "检索 recall@1（生产口径）",
        "retrieval",
        "up",
        0.01,
    ),
    MetricSpec("retrieval.hybrid+rw+score.mrr", "检索 MRR（生产口径）", "retrieval", "up", 0.01),
    # 下面两行是"不加重排"的对照口径。**它和上面三行数值相等是正常的**：
    # 生产用的 ScoreReranker 就是"按融合分数排序"，而 RRF 的输出本来已经按融合分数降序，
    # 所以"加重排"等价于"不加重排"。保留两行不是冗余——两者一旦不等，
    # 说明有人改了重排器的语义，这是需要的哨兵。
    MetricSpec("retrieval.hybrid+rw.recall@5", "检索 recall@5（混合+改写）", "retrieval", "up", 0.01),
    MetricSpec("retrieval.hybrid+rw.recall@1", "检索 recall@1（混合+改写）", "retrieval", "up", 0.01),
    MetricSpec("retrieval.hybrid+rw.mrr", "检索 MRR（混合+改写）", "retrieval", "up", 0.01),
    # 意图：test 集 701 条，1 个点大约 7 条，容差就取 1 个点
    MetricSpec("intent.local.macro_f1", "意图 macro-F1（local）", "intent", "up", 0.01),
    MetricSpec("intent.local.accuracy", "意图准确率（local）", "intent", "up", 0.01),
    MetricSpec("intent.local.p95_ms", "意图 P95 时延（local）", "intent", "info", 0.0, 0.25),
    MetricSpec("intent.local.cached", "意图命中缓存条数（local）", "intent", "info"),
    # 域外：50 条，1 条 = 0.02
    MetricSpec("ood.with_signals.recall", "域外召回（三路信号+护栏）", "ood", "up", 0.02),
    MetricSpec("ood.with_signals.false_kill_rate", "已知类误杀率", "ood", "down", 0.01),
    # 多意图：50 条，1 条 = 0.02
    MetricSpec("multi.parallel.both_covered_rate", "多意图双诉求覆盖（parallel）", "multi", "up", 0.02),
    MetricSpec("multi.multilabel_macro", "多标签 macro-F1（识别层）", "multi", "up", 0.02),
    # 对话：30 段，1 段 ≈ 0.033
    MetricSpec("dialogues.behavior_ok_rate", "判据通过率", "dialogues", "up", 0.01),
    MetricSpec("dialogues.resolution_rate", "目标解决率", "dialogues", "up", 0.034),
    MetricSpec("dialogues.handoff_false_rate", "转人工误转率", "dialogues", "down", 0.034),
    MetricSpec("dialogues.tool_success_rate", "工具成功率", "dialogues", "up", 0.034),
    # judge：30 段，1 段 ≈ 0.033；人工一致率是这一层的验收线，容差放宽到 0.05
    MetricSpec("judge.resolved_rate", "judge 达成率", "judge", "up", 0.034),
    MetricSpec("judge.agreement", "judge 与人工一致率", "judge", "up", 0.05),
    MetricSpec("judge.cost_total", "judge 本次成本（¥）", "judge", "info"),
)


# ---------------- 各层 JSON -> 扁平指标 ----------------


def _get(payload: dict, *path, default=None):
    """按路径安全取值：任何一层缺失都返回 default，不让缺字段把整轮回归打断。"""
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def extract_retrieval(payload: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for item in payload.get("configs") or []:
        name = item.get("config")
        values = item.get("metrics") or {}
        for field in ("recall@1", "recall@3", "recall@5", "mrr"):
            if field in values:
                metrics[f"retrieval.{name}.{field}"] = float(values[field])
    return metrics


def extract_intent(payload: dict, backend: str = "local", split: str = "test") -> dict[str, float]:
    """挑出指定后端 + 指定集合的**最后一条**记录。

    坑（2026-10-15 首次回归暴露）：eval_intent.py 的 JSON 会把历次结果累加，
    同一个 local 后端会有 dev / test / 不同 adapter 目录好几行。取"第一条 backend==local"
    会拿到 dev 集的数字（704 条）当成 test（701 条）报出去，指标名字对、数字是别的集合的。
    这里按 split 过滤再取最后一条。
    """
    entries = [
        item
        for item in payload.get("backends") or []
        if item.get("backend") == backend and (split is None or item.get("split") == split)
    ]
    if not entries:
        return {}
    entry = entries[-1]
    queries = float(entry.get("queries") or 0)
    cached = float(entry.get("cached") or 0)
    metrics = {
        f"intent.{backend}.macro_f1": float(entry.get("macro_f1") or 0.0),
        f"intent.{backend}.accuracy": float(entry.get("accuracy") or 0.0),
        f"intent.{backend}.local_dir": entry.get("local_dir") or "",
        f"intent.{backend}.split": entry.get("split") or "",
        f"intent.{backend}.queries": queries,
        f"intent.{backend}.cached": cached,
        f"intent.{backend}.errors": float(entry.get("errors") or 0),
        f"intent.{backend}.cost_per_1k": float(entry.get("cost_per_1k") or 0.0),
    }
    # 全部命中缓存时没有真实推理，时延字段是 0——直接报出去会被读成"时延变成 0 了"。
    # 这种情况干脆不产出这个指标（表格里显示"缺失"，不影响门禁，因为它是 info 项）。
    if not (queries and cached >= queries):
        metrics[f"intent.{backend}.p95_ms"] = float(
            _get(entry, "latency", "p95_ms", default=0.0) or 0.0
        )
    return metrics


def extract_ood(payload: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for scope in ("model_only", "with_signals", "with_signals_no_veto"):
        block = payload.get(scope) or {}
        if not block:
            continue
        metrics[f"ood.{scope}.recall"] = float(block.get("ood_recall") or 0.0)
        metrics[f"ood.{scope}.false_kill_rate"] = float(block.get("false_kill_rate") or 0.0)
    return metrics


def extract_multi(payload: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, block in (payload.get("strategies") or {}).items():
        metrics[f"multi.{name}.primary_hit_rate"] = float(block.get("primary_hit_rate") or 0.0)
        metrics[f"multi.{name}.secondary_coverage"] = float(block.get("secondary_coverage") or 0.0)
        metrics[f"multi.{name}.both_covered_rate"] = float(block.get("both_covered_rate") or 0.0)
    # 识别层（与策略无关）单独挂一个键，方便看"是模型退化了还是路由退化了"
    multilabel = _get(payload, "strategies", "parallel", "multilabel", "macro")
    if multilabel is not None:
        metrics["multi.multilabel_macro"] = float(multilabel)
    return metrics


def extract_dialogues(payload: dict) -> dict[str, float]:
    metrics = payload.get("metrics") or {}
    records = payload.get("records") or []
    calls = sum(_get(record, "tools", "calls", default=0) or 0 for record in records)
    success = sum(_get(record, "tools", "success", default=0) or 0 for record in records)
    extracted = {
        "dialogues.dialogues": float(metrics.get("dialogues") or 0),
        "dialogues.behavior_ok_rate": float(metrics.get("behavior_ok_rate") or 0.0),
        "dialogues.resolution_rate": float(metrics.get("resolution_rate") or 0.0),
        "dialogues.expected_unresolved_ok_rate": float(
            metrics.get("expected_unresolved_ok_rate") or 0.0
        ),
        "dialogues.handoff_recall": float(metrics.get("handoff_recall") or 0.0),
        "dialogues.handoff_false_rate": float(metrics.get("handoff_false_rate") or 0.0),
    }
    if calls:
        extracted["dialogues.tool_success_rate"] = round(success / calls, 4)
    return extracted


def extract_judge(payload: dict) -> dict[str, float]:
    metrics = payload.get("metrics") or {}
    extracted = {
        "judge.resolved_rate": float(metrics.get("resolved_rate") or 0.0),
        "judge.partial_rate": float(metrics.get("partial_rate") or 0.0),
        "judge.unresolved_rate": float(metrics.get("unresolved_rate") or 0.0),
        "judge.vs_rule_agreement": float(metrics.get("judge_vs_rule_agreement") or 0.0),
        "judge.failed": float(metrics.get("failed") or 0),
        "judge.cost_total": float(metrics.get("cost_total") or 0.0),
    }
    calibration = payload.get("calibration") or {}
    if calibration:
        extracted["judge.agreement"] = float(calibration.get("agreement") or 0.0)
        extracted["judge.compared"] = float(calibration.get("compared") or 0)
    return extracted


# ---------------- 阶段定义 ----------------


@dataclass(frozen=True)
class Stage:
    """一个评测阶段：跑哪个脚本、要哪些参数、结果 JSON 在哪、怎么抽取指标。"""

    name: str
    title: str
    script: str
    args: tuple[str, ...]
    json_path: Path
    extract: str
    timeout: int = 1800


def build_stages(*, with_judge: bool, review_file: Path) -> list[Stage]:
    stages = [
        Stage(
            "retrieval",
            "检索（30 条问句）",
            "eval_retrieval.py",
            (
                "--configs",
                "bm25+rw",
                "hybrid+rw",
                "hybrid+rw+score",
                "--report",
                str(REPORTS / "retrieval_baseline.md"),
                "--json",
                str(REPORTS / "retrieval_baseline.json"),
            ),
            REPORTS / "retrieval_baseline.json",
            "retrieval",
        ),
        Stage(
            "intent",
            "意图（local，走缓存秒级）",
            "eval_intent.py",
            (
                "--backends",
                "local",
                "--splits",
                "test",
                "--report",
                str(REPORTS / "intent_baseline.md"),
                "--json",
                str(REPORTS / "intent_baseline.json"),
            ),
            REPORTS / "intent_baseline.json",
            "intent",
        ),
        Stage(
            "ood",
            "域外检测（50 条域外 + 域内对照）",
            "eval_ood.py",
            (
                "--backend",
                "local",
                "--with-embedder",
                "--report",
                str(REPORTS / "intent_ood.md"),
                "--json",
                str(REPORTS / "intent_ood.json"),
            ),
            REPORTS / "intent_ood.json",
            "ood",
        ),
        Stage(
            "multi",
            "多意图（50 条金标）",
            "eval_multi_intent.py",
            (
                "--backend",
                "local",
                "--report",
                str(REPORTS / "intent_multi.md"),
                "--json",
                str(REPORTS / "intent_multi.json"),
            ),
            REPORTS / "intent_multi.json",
            "multi",
        ),
        Stage(
            "dialogues",
            "端到端对话（30 段，local/priority）",
            "eval_dialogues.py",
            (
                "--backend",
                "local",
                "--strategy",
                "priority",
                "--report",
                str(REPORTS / "dialogues_local.md"),
                "--json",
                str(REPORTS / "dialogues_local.json"),
            ),
            REPORTS / "dialogues_local.json",
            "dialogues",
        ),
    ]
    if with_judge:
        judge_args = [
            "--input",
            str(REPORTS / "dialogues_local.json"),
            "--report",
            str(REPORTS / "dialogues_judge.md"),
            "--json",
            str(REPORTS / "dialogues_judge.json"),
        ]
        if review_file.exists():
            # 复核文件在的时候顺手把一致率也算上（judge 走缓存，基本不花钱）
            judge_args += ["--review", str(review_file)]
        stages.append(
            Stage(
                "judge",
                "LLM-as-judge（30 段，走缓存）",
                "judge_dialogues.py",
                tuple(judge_args),
                REPORTS / "dialogues_judge.json",
                "judge",
                timeout=900,
            )
        )
    return stages


EXTRACTORS = {
    "retrieval": extract_retrieval,
    "intent": extract_intent,
    "ood": extract_ood,
    "multi": extract_multi,
    "dialogues": extract_dialogues,
    "judge": extract_judge,
}


# ---------------- 归因信息：git 与代码指纹 ----------------


def git_info(root: Path) -> dict:
    """当前 commit（有 git 才有）；不是仓库时如实说明，不编造。"""
    if shutil.which("git") is None:
        return {"available": False, "note": "环境里没有 git"}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if head.returncode != 0:
            return {"available": False, "note": "当前目录不是 git 仓库"}
        subject = subprocess.run(
            ["git", "log", "-1", "--pretty=%s"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "available": True,
            "sha": head.stdout.strip(),
            "subject": subject.stdout.strip(),
            "dirty": bool(status.stdout.strip()),
        }
    except Exception as exc:  # noqa: BLE001 - 归因信息拿不到不该影响回归
        return {"available": False, "note": f"{type(exc).__name__}: {exc}"}


def commits_between(root: Path, base: str, head: str) -> list[str]:
    """基线 commit 到当前 commit 之间的提交列表（拿不到就返回空）。"""
    if not base or not head or base == head:
        return []
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"{base}..{head}"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        return [line for line in result.stdout.splitlines() if line.strip()][:10]
    except Exception:  # noqa: BLE001
        return []


def code_fingerprint(root: Path, dirs: tuple[str, ...] = FINGERPRINT_DIRS) -> dict[str, str]:
    """代码指纹：每个 .py 文件的 sha256 前 12 位。

    没有 git 的时候（本项目的工作副本就不是仓库），这就是"改动归因"的替代方案：
    基线里存一份指纹，对比就能列出"这次跑了哪些文件不同"。
    """
    fingerprint: dict[str, str] = {}
    for name in dirs:
        base = root / name
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = str(path.relative_to(root)).replace("\\", "/")
            fingerprint[relative] = sha256(path.read_bytes()).hexdigest()[:12]
    return fingerprint


def diff_fingerprint(baseline: dict, current: dict) -> dict[str, list[str]]:
    """指纹差异：新增 / 修改 / 删除的文件列表。"""
    base_keys, current_keys = set(baseline or {}), set(current or {})
    return {
        "added": sorted(current_keys - base_keys),
        "removed": sorted(base_keys - current_keys),
        "modified": sorted(
            key for key in base_keys & current_keys if baseline[key] != current[key]
        ),
    }


# ---------------- 门禁：逐项对比 ----------------


def compare_metric(spec: MetricSpec, baseline: float | None, current: float | None) -> dict:
    """按方向与容差给出判定：改善 / 持平 / 退化 / 仅记录 / 新增 / 缺失。"""
    row = {
        "key": spec.key,
        "label": spec.label,
        "stage": spec.stage,
        "direction": spec.direction,
        "tolerance": max(spec.tolerance, spec.rel_tolerance * abs(baseline or 0.0)),
        "baseline": baseline,
        "current": current,
        "delta": None,
        "status": "缺失",
    }
    if spec.direction == "info":
        row["status"] = "仅记录" if current is not None else "缺失"
    elif current is None:
        row["status"] = "缺失"
    elif baseline is None:
        row["status"] = "新增"
    else:
        delta = round(current - baseline, 6)
        row["delta"] = delta
        allowed = row["tolerance"]
        better = delta > 0 if spec.direction == "up" else delta < 0
        within = abs(delta) <= allowed
        if better:
            row["status"] = "改善"
        elif within:
            # 容差内的下降记作"持平"：模型与检索都有随机波动，
            # 不给容差的话每天都会因为零点几个点把报告刷红，门禁就没人看了。
            row["status"] = "持平"
        else:
            row["status"] = "退化"
    return row


def compare_all(
    specs: tuple[MetricSpec, ...],
    baseline_metrics: dict[str, float],
    current_metrics: dict[str, float],
) -> list[dict]:
    return [
        compare_metric(spec, baseline_metrics.get(spec.key), current_metrics.get(spec.key))
        for spec in specs
    ]


def merge_baseline(baseline: dict | None, current: dict) -> dict:
    """把本次结果合并进基线：只覆盖本次真正测到的指标，其余沿用旧基线的值。

    为什么不是直接覆盖：日常回归经常只跑一两个阶段（`--only retrieval intent`），
    直接覆盖会把没跑的指标从基线里抹掉，下次跑全都变成"新增"，门禁等于失效。
    代码指纹与 git 信息用**本次**的：那是"从这一刻起"的对比起点。
    """
    if not baseline:
        return current
    current_metrics = dict(current.get("metrics") or {})
    baseline_metrics = dict(baseline.get("metrics") or {})
    reused = sorted(set(baseline_metrics) - set(current_metrics))
    merged = {**current}
    merged["metrics"] = {**baseline_metrics, **current_metrics}
    merged["merged_from"] = {
        "baseline_created_at": baseline.get("created_at"),
        "baseline_label": baseline.get("label", ""),
        "reused_metrics": reused,
    }
    return merged


def run_stage(stage: Stage, *, python: str, timeout: int, dry_run: bool = False) -> dict:
    """跑一个阶段，返回执行记录（含退出码、耗时、输出尾部、抽取到的指标）。"""
    command = [python, str(AGENT_ROOT / "scripts" / stage.script), *stage.args]
    print(f"\n[{stage.name}] {stage.title}")
    print("  $ " + " ".join(command))
    record: dict = {
        "name": stage.name,
        "title": stage.title,
        "command": command,
        "json": str(stage.json_path),
        "duration_s": 0.0,
        "returncode": 0,
        "ok": False,
        "metrics": {},
        "output_tail": [],
        "error": "",
    }
    if dry_run:
        record["ok"] = True
        record["error"] = "dry-run：未执行"
        return record

    started = datetime.now()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    try:
        completed = subprocess.run(
            command,
            cwd=str(AGENT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout or stage.timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        record["error"] = f"超时（>{timeout or stage.timeout}s）"
        record["returncode"] = -1
        record["duration_s"] = round((datetime.now() - started).total_seconds(), 1)
        return record

    record["duration_s"] = round((datetime.now() - started).total_seconds(), 1)
    record["returncode"] = completed.returncode
    tail = (completed.stdout or "").strip().splitlines()[-12:]
    record["output_tail"] = tail
    if completed.stderr and completed.returncode != 0:
        record["output_tail"] = tail + (completed.stderr or "").strip().splitlines()[-8:]

    if completed.returncode != 0:
        record["error"] = f"脚本退出码 {completed.returncode}"
        return record
    if not stage.json_path.exists():
        record["error"] = f"没有产出 {stage.json_path.name}"
        return record

    try:
        payload = json.loads(stage.json_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"读取 {stage.json_path.name} 失败：{exc}"
        return record

    record["metrics"] = EXTRACTORS[stage.extract](payload)
    record["ok"] = True
    print(f"  ✓ {record['duration_s']}s，抽取到 {len(record['metrics'])} 个指标")
    return record


# ---------------- 输出 ----------------


def print_table(rows: list[dict]) -> None:
    print("\n指标".ljust(34) + "上次".rjust(12) + "本次".rjust(12) + "变化".rjust(12) + "  判定")
    print("-" * 88)
    for row in rows:
        baseline = "—" if row["baseline"] is None else f"{row['baseline']:.4f}"
        current = "—" if row["current"] is None else f"{row['current']:.4f}"
        delta = "—" if row["delta"] is None else f"{row['delta']:+.4f}"
        print(f"{row['label'][:32]:34}{baseline:>12}{current:>12}{delta:>12}  {row['status']}")


def write_report(
    path: Path,
    *,
    comparison: list[dict],
    stages: list[dict],
    baseline_payload: dict | None,
    current_payload: dict,
    fingerprint_diff: dict[str, list[str]],
    commit_lines: list[str],
) -> None:
    failed = [row for row in comparison if row["status"] == "退化"]
    broken = [stage for stage in stages if not stage["ok"]]
    lines = [
        "# 自动回归总表",
        "",
        f"- 生成时间：{current_payload['created_at']}",
        f"- 程序：{'、'.join(stage['title'] for stage in stages)}",
        f"- 基线："
        + (
            f"{baseline_payload['created_at']}（{baseline_payload.get('label', '') or '上次运行'}）"
            if baseline_payload
            else "无（本次已记为基线）"
        ),
        "",
    ]

    git = current_payload.get("git") or {}
    if git.get("available"):
        lines.append(
            f"- commit：`{git['sha']}` {git.get('subject', '')}"
            + ("（工作区有未提交改动）" if git.get("dirty") else "")
        )
    else:
        lines.append(f"- commit：无 git 信息（{git.get('note', '-')}）")
    if commit_lines:
        lines += ["", "基线之后的提交：", "", *[f"- {line}" for line in commit_lines]]

    changed_files = [
        *(f"新增 {name}" for name in fingerprint_diff.get("added", [])),
        *(f"修改 {name}" for name in fingerprint_diff.get("modified", [])),
        *(f"删除 {name}" for name in fingerprint_diff.get("removed", [])),
    ]
    lines += ["", "## 一、代码改动（归因用）", ""]
    if not baseline_payload:
        lines.append("本次没有基线，改动清单从下一次开始才有意义。")
    elif not changed_files:
        lines.append("和基线相比没有代码改动（.py 指纹一致）——此时指标若出现波动，只可能来自数据或模型的不确定性。")
    else:
        lines += [f"共 {len(changed_files)} 个文件与基线不同：", "", *[f"- {name}" for name in changed_files]]

    lines += [
        "",
        "## 二、逐项对比",
        "",
        f"结论：**{'有红项' if failed else '全部在容差内'}**"
        + (f"（{len(failed)} 项退化）" if failed else ""),
        "",
        "| 指标 | 上次 | 本次 | 变化 | 容差 | 判定 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in comparison:
        baseline = "—" if row["baseline"] is None else f"{row['baseline']:.4f}"
        current = "—" if row["current"] is None else f"{row['current']:.4f}"
        delta = "—" if row["delta"] is None else f"{row['delta']:+.4f}"
        tolerance = "—" if row["direction"] == "info" else f"±{row['tolerance']:g}"
        lines.append(
            f"| {row['label']} | {baseline} | {current} | {delta} | {tolerance} | "
            f"{'**' + row['status'] + '**' if row['status'] == '退化' else row['status']} |"
        )

    lines += ["", "## 三、红项与阶段状态", ""]
    if failed:
        lines += [
            "以下指标超出容差，先看是不是上表里那批改动文件引起的：",
            "",
            *[
                f"- **{row['label']}**：{row['baseline']:.4f} → {row['current']:.4f}"
                f"（{row['delta']:+.4f}，容差 ±{row['tolerance']:g}）"
                for row in failed
            ],
            "",
        ]
    else:
        lines += ["没有超出容差的指标。", ""]

    lines += ["| 阶段 | 耗时 | 退出码 | 状态 | 说明 |", "| --- | --- | --- | --- | --- |"]
    for stage in stages:
        status = "通过" if stage["ok"] else "**失败**"
        lines.append(
            f"| {stage['title']} | {stage['duration_s']}s | {stage['returncode']} | {status} | "
            f"{stage['error'] or '-'} |"
        )
    if broken:
        for stage in broken:
            lines += ["", f"### {stage['title']} 的输出尾部", "", "```", *stage["output_tail"], "```"]

    lines += [
        "",
        "## 四、怎么用这一条命令",
        "",
        "1. 改完代码先跑 `python scripts/regression.py`，看第二节有没有红项；",
        "2. 红了就看第一节的改动清单——范围越小越好定位；",
        "3. 确认这次没有真退化之后，用 `--update-baseline` 把本次记为新基线；",
        "4. 只是想快跑一遍、不花钱时加 `--no-judge`（judge 层走缓存时其实也是秒级）。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------- 主流程 ----------------


def main() -> int:
    parser = argparse.ArgumentParser(description="一键自动回归：跑完全部评测并与基线逐项对比")
    parser.add_argument("--only", nargs="*", default=None, help="只跑这些阶段")
    parser.add_argument("--skip", nargs="*", default=None, help="跳过这些阶段")
    parser.add_argument("--with-judge", dest="with_judge", action="store_true", default=True)
    parser.add_argument("--no-judge", dest="with_judge", action="store_false")
    parser.add_argument(
        "--review",
        type=Path,
        default=REPORTS / "judge_review.jsonl",
        help="人工复核文件（存在时自动算 judge 一致率）",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--update-baseline", action="store_true", help="把本次结果记为基线")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--timeout", type=int, default=0, help="单阶段超时秒数（0=用阶段默认值）")
    parser.add_argument("--python", default=sys.executable, help="用哪个解释器跑子脚本")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    stages = build_stages(with_judge=args.with_judge, review_file=args.review)
    if args.only:
        wanted = set(args.only)
        stages = [stage for stage in stages if stage.name in wanted]
    if args.skip:
        skipped = set(args.skip)
        stages = [stage for stage in stages if stage.name not in skipped]
    if not stages:
        raise SystemExit("没有要跑的阶段（检查 --only / --skip）")

    print(f"回归阶段：{' → '.join(stage.name for stage in stages)}")
    print(f"解释器：{args.python}")
    if args.dry_run:
        for stage in stages:
            print(f"  $ {args.python} scripts/{stage.script} {' '.join(stage.args)}")

    stage_records = [
        run_stage(stage, python=args.python, timeout=args.timeout, dry_run=args.dry_run)
        for stage in stages
    ]

    metrics: dict[str, float] = {}
    for record in stage_records:
        metrics.update(record["metrics"])

    fingerprint = code_fingerprint(AGENT_ROOT)
    git = git_info(AGENT_ROOT)
    current_payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": "、".join(record["name"] for record in stage_records),
        "git": git,
        "code_fingerprint": fingerprint,
        "stages": stage_records,
        "metrics": metrics,
    }

    baseline_payload: dict | None = None
    if args.baseline.exists():
        try:
            baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 基线文件读不出来（{exc}），本次按「首次运行」处理")
            baseline_payload = None

    baseline_metrics = (baseline_payload or {}).get("metrics") or {}
    comparison = compare_all(METRIC_SPECS, baseline_metrics, metrics)
    # 只跑了一部分阶段时（--only / --skip），没跑的阶段标成"未测"，
    # 而不是"缺失"——否则报告里一排"缺失"看着像全线崩了。
    ran_stages = {record["name"] for record in stage_records}
    for row in comparison:
        if row["stage"] not in ran_stages and row["status"] in {"缺失", "新增"}:
            row["status"] = "未测"
    fingerprint_diff = diff_fingerprint(
        (baseline_payload or {}).get("code_fingerprint") or {}, fingerprint
    )
    commit_lines = []
    if baseline_payload:
        commit_lines = commits_between(
            AGENT_ROOT,
            _get(baseline_payload, "git", "sha", default=""),
            git.get("sha", ""),
        )

    if not args.dry_run:
        print_table(comparison)

    failed = [row for row in comparison if row["status"] == "退化"]
    broken = [stage for stage in stage_records if not stage["ok"]]

    current_payload["comparison"] = comparison
    current_payload["fingerprint_diff"] = fingerprint_diff
    current_payload["passed"] = not failed and not broken
    if not args.dry_run:
        # dry-run 只打印计划，不写任何产物，免得把正式报告覆盖成空表
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(current_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if not args.no_report:
            write_report(
                args.report,
                comparison=comparison,
                stages=stage_records,
                baseline_payload=baseline_payload,
                current_payload=current_payload,
                fingerprint_diff=fingerprint_diff,
                commit_lines=commit_lines,
            )

    if not args.dry_run and (args.update_baseline or baseline_payload is None):
        merged = merge_baseline(baseline_payload, current_payload)
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if baseline_payload is None:
            print(f"\n首次运行：本次结果已记为基线（{args.baseline}），下次跑才有对比")
        else:
            reused = (merged.get("merged_from") or {}).get("reused_metrics") or []
            print(
                f"\n基线已更新：{args.baseline}"
                f"（本次重测 {len(current_payload['metrics'])} 个指标"
                + (f"，其余 {len(reused)} 个沿用旧基线、本次没测" if reused else "")
                + "）"
            )

    if not args.dry_run:
        print(f"\n指标 JSON：{args.json}")
        if not args.no_report:
            print(f"报告已写入：{args.report}")
    else:
        print("\n（dry-run：只打印计划，不执行、不写任何文件）")
        return 0

    if broken:
        print(f"\n有 {len(broken)} 个阶段没跑通：{'、'.join(stage['name'] for stage in broken)}")
        return 1
    if failed:
        print(f"\n有 {len(failed)} 项指标超出容差：")
        for row in failed:
            files = ", ".join(
                fingerprint_diff.get("modified", [])[:5] or fingerprint_diff.get("added", [])[:5]
            )
            print(
                f"  {row['label']}：{row['baseline']:.4f} → {row['current']:.4f}"
                f"（{row['delta']:+.4f}）"
                + (f"；本次改动：{files}" if files else "")
            )
        return 1
    print("\n全部指标在容差内。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
