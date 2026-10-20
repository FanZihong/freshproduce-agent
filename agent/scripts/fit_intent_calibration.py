"""置信度校准 + 拒答阈值拟合（读取本地推理缓存，不需要重新跑模型）。

本地推理结果已经按行缓存在 `eval/reports/local_eval_cache.json`（含每条样本的置信度），
所以拟合温度与阈值只需读文件，几乎不耗 CPU。

用法（在 agent/ 目录下执行）：
    python scripts/fit_intent_calibration.py                          # 用 test 缓存试算（打通流程）
    python scripts/fit_intent_calibration.py --fit-split dev --eval-split test   # 正式口径

口径提醒：温度和阈值**必须在 dev 上拟合**，test 只用于验收。用 test 拟合属于数据泄漏，
脚本会在报告里显式标注。想拿 dev 的缓存先跑一次：
    python scripts/eval_intent.py --backends local --splits dev
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DATA_DIR  # noqa: E402
from agent.intent.decision import (  # noqa: E402
    DecisionConfig,
    apply_temperature,
    choose_thresholds,
    expected_calibration_error,
    fit_temperature,
    reliability_table,
    save_calibration,
)
from scripts.eval_intent import row_cache_key  # noqa: E402

DEFAULT_DATASET_DIRS = (
    DATA_DIR / "intent" / "dataset_v1_balanced",
    DATA_DIR / "intent" / "dataset_v1",
)
DEFAULT_CACHE = AGENT_ROOT / "eval" / "reports" / "local_eval_cache.json"
DEFAULT_JSON = AGENT_ROOT / "eval" / "reports" / "intent_calibration.json"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "intent_calibration.md"
DEFAULT_LOCAL_META = AGENT_ROOT / "eval" / "reports" / "local_eval_meta.json"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_dataset_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in DEFAULT_DATASET_DIRS:
        if (candidate / "test.jsonl").exists():
            return candidate
    raise SystemExit(f"找不到数据集：{DEFAULT_DATASET_DIRS[0]}")


def load_pairs(
    dataset_dir: Path, split: str, cache: dict, namespace: str = ""
) -> tuple[list[float], list[int], list[dict]]:
    """从缓存里取出 (置信度, 主标签是否正确, 明细) —— 主标签口径与 macro-F1 一致。"""
    path = dataset_dir / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"找不到 {path}")
    confidences: list[float] = []
    corrects: list[int] = []
    details: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        entry = cache.get(row_cache_key(row.get("text", ""), row.get("context"), namespace))
        if not entry:
            continue
        gold = (row.get("labels") or ["chitchat"])[0]
        predicted = (entry.get("labels") or [None])[0]
        confidence = float(entry.get("confidence") or 0.0)
        confidences.append(confidence)
        corrects.append(int(predicted == gold))
        details.append(
            {"id": row.get("id"), "text": row.get("text"), "gold": gold,
             "pred": predicted, "confidence": confidence}
        )
    return confidences, corrects, details


def build_report_lines(
    args: argparse.Namespace,
    split_sizes: dict[str, int],
    temperature: float,
    table_raw: list[dict],
    table_cal: list[dict],
    ece_raw: float,
    ece_cal: float,
    thresholds: dict,
    config: DecisionConfig,
    worst: list[dict],
    verdict: list[str],
) -> list[str]:
    lines = [
        "# 意图置信度校准与拒答报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据：{args.dataset_dir.name}"
        + "".join(f"（{name} {size} 条）" for name, size in split_sizes.items()),
        f"- 拟合集 / 验收集：{args.fit_split} / {args.eval_split}",
        f"- 校准前 ECE：{ece_raw:.4f} → 校准后：{ece_cal:.4f}（温度 T = {temperature:.4f}）",
        "",
        "## 可靠性曲线（校准前）",
        "",
        "| 置信度区间 | 样本数 | 平均置信度 | 真实准确率 | 差距 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in table_raw:
        lines.append(
            f"| {item['bin']} | {item['count']} | {item['avg_confidence']:.4f} | "
            f"{item['accuracy']:.4f} | {item['gap']:+.4f} |"
        )
    lines += ["", "## 可靠性曲线（校准后）", "", "| 置信度区间 | 样本数 | 平均置信度 | 真实准确率 | 差距 |",
              "| --- | --- | --- | --- | --- |"]
    for item in table_cal:
        lines.append(
            f"| {item['bin']} | {item['count']} | {item['avg_confidence']:.4f} | "
            f"{item['accuracy']:.4f} | {item['gap']:+.4f} |"
        )
    lines += [
        "",
        "## 拒答阈值",
        "",
        f"- 放行阈值：置信度 ≥ {config.accept_threshold}",
        f"- 转人工阈值：置信度 < {config.handoff_threshold}",
        f"- 拒答率：{thresholds.get('refusal_rate')}（上限 {thresholds.get('max_refusal_rate')}）",
        f"- 被拒样本错误率：{thresholds.get('refused_error_rate')}；"
        f"被接受样本错误率：{thresholds.get('accepted_error_rate')}；"
        f"倍数 ratio：{thresholds.get('ratio')}",
    ]
    if thresholds.get("note"):
        lines.append(f"- 注意：{thresholds['note']}")
    if worst:
        lines += ["", "## 低置信度且判错的样本（Top 10，用于看拒答是否挑得准）", "",
                  "| id | 问句 | 真值 | 预测 | 置信度 |", "| --- | --- | --- | --- | --- |"]
        for item in worst:
            lines.append(
                f"| {item['id']} | {str(item['text'])[:36]} | {item['gold']} | {item['pred']} | "
                f"{item['confidence']:.4f} |"
            )
    lines += ["", "## 验收判定", ""]
    lines += [f"- {item}" for item in verdict]
    lines += [
        "",
        f"复现命令：`python scripts/fit_intent_calibration.py --fit-split {args.fit_split} "
        f"--eval-split {args.eval_split} --max-refusal {args.max_refusal}`",
        "",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="拟合置信度温度与拒答阈值")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fit-split", default="test", help="拟合温度的集合（正式口径应为 dev）")
    parser.add_argument("--eval-split", default=None, help="验收集合，默认与 --fit-split 相同")
    parser.add_argument("--max-refusal", type=float, default=0.10, help="拒答率上限")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    args.dataset_dir = resolve_dataset_dir(args.dataset_dir)
    args.eval_split = args.eval_split or args.fit_split
    if not args.cache.exists():
        raise SystemExit(
            f"没有本地推理缓存 {args.cache}\n"
            "  先跑：python scripts/eval_intent.py --backends local --splits "
            f"{args.fit_split}"
        )
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    entries = sum(1 for key in cache if not str(key).startswith("_"))
    print(f"缓存：{args.cache}（{entries} 条）")
    # 缓存键里带了 adapter 路径，这里必须用同一把盐，否则一条都命中不了。
    # 优先用缓存自带的 _meta（最可靠），其次 sidecar，最后按命中数自动探测。
    namespace = (cache.get("_meta") or {}).get("adapter_dir") or _adapter_dir() or ""
    detected, hits = detect_namespace(
        cache, args.dataset_dir, args.eval_split, candidate_namespaces(namespace)
    )
    if hits and detected != namespace:
        print(f"[提示] sidecar 里的 adapter（{namespace or '空'}）对不上缓存，"
              f"按命中数改用：{detected}（{hits} 条命中）")
        namespace = detected
    print(f"本地推理使用的 adapter：{namespace or '（无法确定，缓存可能对不上）'}")

    split_sizes: dict[str, int] = {}
    fit_conf, fit_correct, _ = load_pairs(args.dataset_dir, args.fit_split, cache, namespace)
    split_sizes[args.fit_split] = len(fit_conf)
    if args.eval_split != args.fit_split:
        eval_conf, eval_correct, eval_details = load_pairs(
            args.dataset_dir, args.eval_split, cache, namespace
        )
        split_sizes[args.eval_split] = len(eval_conf)
    else:
        eval_conf, eval_correct, eval_details = fit_conf, fit_correct, []
        _, _, eval_details = load_pairs(args.dataset_dir, args.fit_split, cache, namespace)

    if not fit_conf:
        raise SystemExit(
            f"{args.fit_split} 里的样本在缓存里一条都没命中；先跑 "
            f"python scripts/eval_intent.py --backends local --splits {args.fit_split}"
        )
    if args.fit_split == "test":
        print("[口径提醒] 温度和阈值在 test 上拟合属于数据泄漏，仅用于打通流程；")
        print("           正式结论请先跑 dev 的本地推理，再用 --fit-split dev --eval-split test")

    temperature = fit_temperature(fit_conf, fit_correct)
    calibrated = [apply_temperature(value, temperature) for value in eval_conf]
    table_raw = reliability_table(eval_conf, eval_correct)
    table_cal = reliability_table(calibrated, eval_correct)
    ece_raw = expected_calibration_error(table_raw)
    ece_cal = expected_calibration_error(table_cal)
    thresholds = choose_thresholds(calibrated, eval_correct, max_refusal_rate=args.max_refusal)
    config = DecisionConfig(
        temperature=temperature,
        accept_threshold=thresholds["accept_threshold"],
        handoff_threshold=round(max(0.05, thresholds["accept_threshold"] * 0.6), 4),
    )

    accuracy = sum(eval_correct) / len(eval_correct)
    verdict: list[str] = []
    verdict.append(
        f"整体准确率 {accuracy:.4f}；平均原始置信度 {sum(eval_conf) / len(eval_conf):.4f}"
        f"（过度自信 {'是' if sum(eval_conf) / len(eval_conf) > accuracy else '否'}）"
    )
    verdict.append(
        f"校准：T={temperature:.4f}，ECE {ece_raw:.4f} → {ece_cal:.4f}，"
        + ("校准有效" if ece_cal < ece_raw else "校准没有改善，检查置信度是否可用")
    )
    ratio = thresholds.get("ratio")
    refusal_ok = (thresholds.get("refusal_rate") or 0) <= args.max_refusal
    ratio_ok = bool(ratio) and ratio >= 2.0
    verdict.append(
        f"拒答率 {thresholds.get('refusal_rate')} ≤ {args.max_refusal}？{'✓' if refusal_ok else '✗'}；"
        f"被拒样本错误率是被接受样本的 {ratio} 倍（≥2 才算阈值有效）：{'✓' if ratio_ok else '✗'}"
    )
    if not ratio_ok:
        verdict.append(
            "阈值区分度不足：优先做置信度源的改进（例如用候选标签 softmax 代替生成序列概率），"
            "而不是继续调阈值"
        )

    worst = sorted(
        (item for item, correct in zip(eval_details, eval_correct) if not correct),
        key=lambda item: item["confidence"],
    )[:10]

    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "adapter_dir": _adapter_dir(),
        # 校准作用在哪个后端上：图里的决策层只在"当前分类器 == 这个值"时才启用
        "backend": "local-lora",
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "samples": len(eval_conf),
        "accuracy": round(accuracy, 4),
        "temperature": temperature,
        "ece_before": ece_raw,
        "ece_after": ece_cal,
        "thresholds": thresholds,
        "config": config.to_dict(),
        "reliability_before": table_raw,
        "reliability_after": table_cal,
        "verdict": verdict,
        "note": "fit_split=test 时属于数据泄漏，正式口径为 --fit-split dev --eval-split test",
    }
    save_calibration(args.json, payload)

    print(f"\n温度 T = {temperature}")
    print(f"ECE：{ece_raw} → {ece_cal}")
    print(f"放行阈值 {config.accept_threshold}（拒答率 {thresholds.get('refusal_rate')}，"
          f"ratio {thresholds.get('ratio')}）")
    print("\n可靠性曲线（校准前 → 后）：")
    for raw_item, cal_item in zip(table_raw, table_cal):
        print(
            f"  {raw_item['bin']}  样本 {raw_item['count']:>4}  "
            f"置信 {raw_item['avg_confidence']:.3f} -> {cal_item['avg_confidence']:.3f}  "
            f"准确率 {raw_item['accuracy']:.3f}  gap {raw_item['gap']:+.3f} -> {cal_item['gap']:+.3f}"
        )
    print("\n验收判定：")
    for item in verdict:
        print(f"  - {item}")
    print(f"\n配置已写入：{args.json}")

    if not args.no_report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            "\n".join(
                build_report_lines(
                    args, split_sizes, temperature, table_raw, table_cal, ece_raw, ece_cal,
                    thresholds, config, worst, verdict,
                )
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"报告已写入：{args.report}")
    return 0


def _adapter_dir() -> str | None:
    """从本地推理元信息里读 adapter 路径，让校准结果能挂回具体版本。"""
    if not DEFAULT_LOCAL_META.exists():
        return None
    try:
        return json.loads(DEFAULT_LOCAL_META.read_text(encoding="utf-8")).get("adapter_dir")
    except json.JSONDecodeError:
        return None


def candidate_namespaces(extra: str | None = None) -> list[str]:
    """缓存键里可能用过的 namespace：sidecar 里的值 + 现存 adapter 的两种路径写法。

    为什么需要列举：namespace 就是 `str(resolve_adapter_dir(--local-dir))`，
    用户可能写成相对路径（data/intent/models/v1），也可能是绝对路径，
    而 sidecar 一旦被写坏（我们踩过一次），只能靠"试哪个命中多"来还原。
    """
    candidates: list[str] = []
    if extra:
        candidates.append(str(extra))
    models_dir = DATA_DIR / "intent" / "models"
    if models_dir.exists():
        for directory in sorted(models_dir.iterdir()):
            if directory.is_dir():
                candidates.append(str(directory))
                candidates.append(f"data/intent/models/{directory.name}")
    return list(dict.fromkeys(candidates))


def detect_namespace(
    cache: dict, dataset_dir: Path, split: str, candidates: list[str]
) -> tuple[str, int]:
    """挑出命中缓存最多的那个 namespace（命中 0 时返回 ('', 0)）。"""
    rows = load_jsonl(dataset_dir / f"{split}.jsonl")
    best, best_hits = "", 0
    for namespace in candidates:
        hits = sum(
            1
            for row in rows
            if row_cache_key(row.get("text", ""), row.get("context"), namespace) in cache
        )
        if hits > best_hits:
            best, best_hits = namespace, hits
    return best, best_hits


if __name__ == "__main__":
    raise SystemExit(main())
