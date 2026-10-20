"""意图识别评测：规则版 / few-shot 基线（后续再加本地小模型）横向对比并出报告。

对齐 `scripts/eval_retrieval.py` 的用法，默认同时给出：
- 模块指标：10 类 macro-F1、主标签准确率、多标签完全一致率、多标签 macro/micro F1；
- 逐类 P/R/F1 与混淆矩阵（面试时要能指出"哪两类最容易混"）；
- 性能与成本：P95 时延、单次成本、千次成本、失败率、token 用量。

用法（在 agent/ 目录下执行）：
    python scripts/eval_intent.py                    # rule + few-shot，默认测 test 集
    python scripts/eval_intent.py --backends rule    # 只跑离线规则版，不花 API 钱
    python scripts/eval_intent.py --splits dev test --limit 100
    python scripts/eval_intent.py --no-cache         # 忽略缓存，全部真实调用
    python scripts/eval_intent.py --fallback-rule    # 调用失败时回落规则版（口径另记）

缓存说明：默认读写 `eval/reports/intent_cache.json`，键是 (模型, 提示词, 输入) 的哈希，
所以调参后重跑不会重复付费；缓存条目里保留 token 数，成本口径不受影响。
P95 时延只统计真实调用（缓存命中的那条不计入，因为它的时延不是模型时延）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DATA_DIR  # noqa: E402
from agent.intent.baseline import FewShotIntentClassifier, load_intent_config  # noqa: E402
from agent.intent.classifier import RuleBasedIntentClassifier  # noqa: E402
from agent.intent.local_model import (  # noqa: E402
    LocalIntentClassifier,
    ModelNotFoundError,
    resolve_adapter_dir,
)
from agent.intent.taxonomy import ALIASES, load_taxonomy  # noqa: E402
from eval.metrics import (  # noqa: E402
    SKLEARN_AVAILABLE,
    accuracy,
    confusion_matrix,
    cost_per_1k,
    exact_match_ratio,
    format_confusion_matrix,
    format_prf_table,
    latency_stats,
    macro_f1,
    multilabel_f1,
    per_class_prf,
    token_stats,
)

DATASET_DIRS: tuple[Path, ...] = (
    DATA_DIR / "intent" / "dataset_v1_balanced",
    DATA_DIR / "intent" / "dataset_v1",
)
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "intent_baseline.md"
DEFAULT_JSON = AGENT_ROOT / "eval" / "reports" / "intent_baseline.json"
DEFAULT_LOCAL_META = AGENT_ROOT / "eval" / "reports" / "local_eval_meta.json"
DEFAULT_CACHE = AGENT_ROOT / "eval" / "reports" / "intent_cache.json"
DEFAULT_LOCAL_CACHE = AGENT_ROOT / "eval" / "reports" / "local_eval_cache.json"
UNRECOGNIZED = "<未识别>"


def row_cache_key(text: str, context, namespace: str = "") -> str:
    """本地推理的行级缓存键：同一条输入（含上下文）永远映射到同一个键。

    namespace 用来隔离"哪个 adapter 算出来的"——不带它的话，换模型重跑会**静默复用
    上一个模型的预测**，指标看着正常但其实是别人的结果（我们踩过一次）。
    """
    payload = json.dumps([namespace, text, context], ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def partition_rows_by_cache(
    rows: list[dict], cache: dict, namespace: str = ""
) -> tuple[list[tuple[int, dict]], dict[int, dict]]:
    """拆成"已有缓存"与"待推理"两部分，返回 (待处理的行+下标, 命中缓存的记录)。

    CPU 上跑 701 条本地推理要几十分钟，终端一断就白跑；有了这个拆分，
    重跑会跳过已算过的行，只补剩下的。
    """
    pending: list[tuple[int, dict]] = []
    hits: dict[int, dict] = {}
    for index, row in enumerate(rows):
        key = row_cache_key(row.get("text", ""), row.get("context"), namespace)
        if key in cache:
            hits[index] = cache[key]
        else:
            pending.append((index, row))
    return pending, hits


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def resolve_dataset_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in DATASET_DIRS:
        if (candidate / "test.jsonl").exists():
            return candidate
    raise SystemExit(f"找不到数据集目录，先跑 scripts/split_intent_dataset.py；预期位置：{DATASET_DIRS[0]}")


def primary_label(row: dict) -> str:
    labels = row.get("labels") or []
    return labels[0] if labels else "unknown"


def stratified_sample(rows: list[dict], limit: int, seed: int = 42) -> list[dict]:
    """按主标签分层抽样：每个类别轮流取，保证抽到的 slice 仍然覆盖所有类别。

    直接切文件头会踩坑——划分脚本是按类别分块写盘的，前 N 行可能只有一两个类，
    指标会变成"8 个类 F1=0"，看起来像模型崩了。
    """
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(primary_label(row), []).append(row)
    for group in grouped.values():
        rng.shuffle(group)

    buckets = [grouped[label] for label in sorted(grouped)]
    picked: list[dict] = []
    index = 0
    while len(picked) < limit:
        progressed = False
        for bucket in buckets:
            if index < len(bucket):
                picked.append(bucket[index])
                progressed = True
                if len(picked) >= limit:
                    break
        if not progressed:
            break
        index += 1
    rng.shuffle(picked)
    return picked


def load_split(dataset_dir: Path, name: str, limit: int = 0) -> list[dict]:
    path = dataset_dir / f"{name}.jsonl"
    if not path.exists():
        print(f"[warn] 找不到 {path}，跳过 {name}")
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit and limit < len(rows):
        return stratified_sample(rows, limit)
    return rows


def normalize_labels(labels, valid: set[str]) -> tuple[str, ...]:
    """把预测标签对齐到体系：去掉别名、去重、剔除体系外的标签。"""
    result: list[str] = []
    for label in labels or ():
        candidate = ALIASES.get(str(label), str(label))
        if candidate in valid and candidate not in result:
            result.append(candidate)
    return tuple(result)


def run_rows(rows: list[dict], predict, concurrency: int) -> list[dict]:
    """按行跑预测；concurrency > 1 时用线程池（只对需要网络的后端有意义）。"""
    if concurrency <= 1 or len(rows) <= 1:
        return [predict(row) for row in rows]

    results: list[dict | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(predict, row): index for index, row in enumerate(rows)}
        done = 0
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            done += 1
            if done % 100 == 0:
                print(f"    已评测 {done}/{len(rows)} 条")
    return [item for item in results if item is not None]


def _base_record(row: dict) -> dict:
    return {
        "id": row.get("id"),
        "text": row.get("text", ""),
        "context": row.get("context"),
        "gold": list(row.get("labels") or []),
    }


def eval_rule(rows: list[dict], valid: set[str]) -> list[dict]:
    classifier = RuleBasedIntentClassifier()

    def predict(row: dict) -> dict:
        call = classifier.classify(row.get("text", ""))
        record = _base_record(row)
        record.update(
            {
                "pred": list(normalize_labels(call.all_labels, valid)),
                "confidence": call.confidence,
                "latency_ms": 0.0,
                "cost": 0.0,
                "cached": False,
                "error": "",
                "strategy": classifier.name,
            }
        )
        return record

    return run_rows(rows, predict, 1)


def eval_fewshot(
    rows: list[dict],
    valid: set[str],
    args: argparse.Namespace,
    cache: dict | None,
) -> list[dict]:
    config = load_intent_config()
    if args.model:
        config.model = args.model
    classifier = FewShotIntentClassifier(
        config=config,
        shots_per_class=args.shots,
        response_cache=cache,
        cache_namespace=f"intent:{config.model}:{args.shots}shot",
    )
    fallback = RuleBasedIntentClassifier()

    def predict(row: dict) -> dict:
        call = classifier.classify_with_stats(row.get("text", ""), row.get("context"))
        labels = list(normalize_labels(call.labels, valid))
        record = _base_record(row)
        if not labels and args.fallback_rule:
            # 失败样本用规则版顶上：指标更好看，但必须单独记口径
            labels = list(normalize_labels(fallback.classify(row.get("text", "")).all_labels, valid))
            record["fallback_used"] = True
        record.update(
            {
                "pred": labels,
                "confidence": call.confidence,
                "latency_ms": call.latency_ms,
                "cost": call.cost,
                "prompt_tokens": call.prompt_tokens,
                "completion_tokens": call.completion_tokens,
                "cached": call.cached,
                "error": call.error,
                "strategy": call.strategy,
                "raw": call.raw,
            }
        )
        return record

    return run_rows(rows, predict, args.concurrency)


def eval_local(
    rows: list[dict],
    valid: set[str],
    args: argparse.Namespace,
    taxonomy,
) -> list[dict] | None:
    """本地 LoRA 模型后端：批量推理，成本记 0，时延按实测；结果按行缓存可续跑。"""
    adapter_dir = resolve_adapter_dir(args.local_dir)
    classifier = LocalIntentClassifier(adapter_dir=adapter_dir, batch_size=args.local_batch_size)
    print(f"  本地模型：{adapter_dir}")
    if not classifier.available:
        return None

    cache: dict = {}
    cache_enabled = not args.no_cache
    if cache_enabled and args.local_cache.exists():
        cache = json.loads(args.local_cache.read_text(encoding="utf-8"))

    namespace = str(adapter_dir)  # 缓存按 adapter 隔离，换模型不会串味
    pending, hits = (
        partition_rows_by_cache(rows, cache, namespace)
        if cache_enabled
        else (list(enumerate(rows)), {})
    )
    print(f"    缓存命中 {len(hits)} 条，待推理 {len(pending)} 条（缓存：{args.local_cache.name if cache_enabled else '已禁用'}）")

    records: list[dict | None] = [None] * len(rows)

    def fill(index: int, row: dict, prediction: dict) -> None:
        record = _base_record(row)
        record.update(
            {
                "pred": list(normalize_labels(prediction.get("labels") or [], valid)),
                "confidence": float(prediction.get("confidence") or 0.0),
                "latency_ms": float(prediction.get("latency_ms") or 0.0),
                "cost": 0.0,  # 本地推理只有电费
                "cached": bool(prediction.get("cached")),
                "error": str(prediction.get("error") or ""),
                "strategy": classifier.name,
            }
        )
        records[index] = record

    for index, prediction in hits.items():
        fill(index, rows[index], {**prediction, "cached": True})

    batch_size = max(1, args.local_batch_size)
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        texts = [row.get("text", "") for _, row in chunk]
        contexts = [row.get("context") for _, row in chunk]
        try:
            predictions, latencies = classifier.classify_batch(texts, contexts)
        except (ValueError, ModelNotFoundError):
            # 整批解码失败时逐条重试，避免一批坏掉整段评测
            predictions, latencies = [], []
            for text, context in zip(texts, contexts):
                try:
                    predictions.append(classifier.classify(text, context))
                except (ValueError, ModelNotFoundError):
                    predictions.append(None)
                latencies.append(0.0)
        for offset, (index, row) in enumerate(chunk):
            prediction = predictions[offset] if offset < len(predictions) else None
            payload = {
                "labels": list(prediction.all_labels) if prediction is not None else [],
                "confidence": prediction.confidence if prediction is not None else 0.0,
                "latency_ms": latencies[offset] if offset < len(latencies) else 0.0,
                "error": "" if prediction is not None else "LocalDecodeError: 输出无法解析",
            }
            fill(index, row, payload)
            if cache_enabled:
                cache[row_cache_key(row.get("text", ""), row.get("context"), namespace)] = payload
        done = min(start + batch_size, len(pending))
        if done % (batch_size * 5) == 0:
            print(f"    已推理 {done}/{len(pending)} 条")
            stamp_cache_meta(cache, namespace)
            flush_cache(args.local_cache, cache, cache_enabled)
    stamp_cache_meta(cache, namespace)
    flush_cache(args.local_cache, cache, cache_enabled)
    return [record for record in records if record is not None]


def flush_cache(path: Path, cache: dict, enabled: bool) -> None:
    """每几批落一次盘：终端断了也能接着跑。"""
    if not enabled:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def stamp_cache_meta(cache: dict, namespace: str) -> None:
    """把"这份缓存是哪个 adapter 算出来的"写进缓存**自身**。

    之前这个信息只存在 sidecar（local_eval_meta.json）里，结果被测试里的占位值写坏过一次，
    校准阶段就一条缓存都命中不了。缓自带元信息以后，sidecar 只是给人看的副本。
    """
    cache["_meta"] = {
        "adapter_dir": namespace,
        "written_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "entries": sum(1 for key in cache if not str(key).startswith("_")),
    }


def summarize(
    records: list[dict], taxonomy, price_in: float, price_out: float, backend: str
) -> dict:
    """把逐条结果聚合成一份指标包。"""
    labels = list(taxonomy.keys)
    gold_primary = [record["gold"][0] if record["gold"] else "chitchat" for record in records]
    pred_primary = [record["pred"][0] if record["pred"] else UNRECOGNIZED for record in records]

    fresh = [record for record in records if not record.get("cached")]
    # 失败样本也算"一次真实调用"：它既产生时延也产生成本
    latencies = [record["latency_ms"] for record in fresh if record["latency_ms"]]
    costs = [record["cost"] for record in records]
    errors = [record for record in records if record.get("error")]

    metrics = {
        "backend": backend,
        "queries": len(records),
        "macro_f1": macro_f1(gold_primary, pred_primary, labels),
        "accuracy": accuracy(gold_primary, pred_primary),
        "exact_match": exact_match_ratio(
            [record["gold"] for record in records], [record["pred"] for record in records]
        ),
        "multilabel": multilabel_f1(
            [record["gold"] for record in records], [record["pred"] for record in records], labels
        ),
        "multi_label_rows": sum(1 for record in records if len(record["gold"]) > 1),
        "errors": len(errors),
        "fallbacks": sum(1 for record in records if record.get("fallback_used")),
        "cached": sum(1 for record in records if record.get("cached")),
        "latency": latency_stats(latencies),
        "cost_total": round(sum(costs), 6),
        "cost_avg": round(sum(costs) / len(costs), 8) if costs else 0.0,
        "cost_per_1k": cost_per_1k(costs),
        "tokens": token_stats(
            [record.get("prompt_tokens", 0) for record in records],
            [record.get("completion_tokens", 0) for record in records],
        ),
        "prices": {"in_per_m": price_in, "out_per_m": price_out},
    }
    metrics["per_class"] = per_class_prf(gold_primary, pred_primary, labels)
    metrics["confusion"] = confusion_matrix(
        gold_primary, pred_primary, labels + [UNRECOGNIZED]
    )
    return metrics


def print_table(results: list[tuple[str, dict]]) -> None:
    header = (
        _pad("后端", 34)
        + _pad("macro-F1", 12)
        + _pad("acc", 10)
        + _pad("多标签一致", 14)
        + _pad("P95(ms)", 11)
        + _pad("单次成本(¥)", 14)
        + _pad("千次成本(¥)", 14)
        + _pad("失败", 8)
        + _pad("缓存命中", 10)
    )
    print(header)
    print("-" * _width(header))
    for name, metrics in results:
        line = (
            _pad(name, 34)
            + _pad(f"{metrics['macro_f1']:.4f}", 12)
            + _pad(f"{metrics['accuracy']:.4f}", 10)
            + _pad(f"{metrics['exact_match']:.4f}", 14)
            + _pad(f"{metrics['latency']['p95_ms']:.1f}", 11)
            + _pad(f"{metrics['cost_avg']:.6f}", 14)
            + _pad(f"{metrics['cost_per_1k']:.4f}", 14)
            + _pad(str(metrics["errors"]), 8)
            + _pad(str(metrics["cached"]), 10)
        )
        print(line)


def top_confusions(matrix: dict, labels: list[str], limit: int = 3) -> list[tuple[str, str, int]]:
    pairs: list[tuple[str, str, int]] = []
    for gold_label in labels:
        for pred_label, count in matrix.get(gold_label, {}).items():
            if pred_label != gold_label and count:
                pairs.append((gold_label, pred_label, count))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:limit]


def write_report(
    path: Path,
    dataset_dir: Path,
    split_sizes: dict[str, int],
    results: list[tuple[str, dict]],
    labels: list[str],
    command: str,
    notes: list[str],
    sampling_note: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first = results[0][1] if results else {}
    prices = first.get("prices", {})
    lines = [
        "# 意图识别基线报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据集：{dataset_dir.name}"
        + "".join(f"（{name} {size} 条）" for name, size in split_sizes.items())
        + sampling_note,
        f"- 评测命令：`{command}`",
        f"- 计价假设：输入 ¥{prices.get('in_per_m', 0)}/百万 token、"
        f"输出 ¥{prices.get('out_per_m', 0)}/百万 token"
        "（改动请设 INTENT_PRICE_IN / INTENT_PRICE_OUT，报告口径随之变化）",
        f"- scikit-learn：{'已启用（F1 走 sklearn 口径）' if SKLEARN_AVAILABLE else '未安装，F1 走等价兜底实现（建议 pip install scikit-learn）'}",
        "- test 集锁死只用于最终评估，dev 集用于调阈值；P95 时延只统计真实调用（缓存命中不计入）。",
        "",
        "## 总览",
        "",
        "| 后端 | 条数 | macro-F1 | accuracy | 多标签完全一致 | 多标签 F1(macro/micro) | P95 时延(ms) | 单次成本(¥) | 千次成本(¥) | 失败 | 缓存命中 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, metrics in results:
        lines.append(
            f"| {name} | {metrics['queries']} | {metrics['macro_f1']:.4f} | {metrics['accuracy']:.4f} | "
            f"{metrics['exact_match']:.4f} | {metrics['multilabel']['macro']:.4f} / "
            f"{metrics['multilabel']['micro']:.4f} | {metrics['latency']['p95_ms']:.1f} | "
            f"{metrics['cost_avg']:.6f} | {metrics['cost_per_1k']:.4f} | {metrics['errors']} | "
            f"{metrics['cached']} |"
        )

    for name, metrics in results:
        lines += ["", f"## 逐类指标（{name}）", "", *format_prf_table(metrics["per_class"], labels)]
        confusions = top_confusions(metrics["confusion"], labels + [UNRECOGNIZED])
        if confusions:
            lines += ["", "主要混淆（真值 → 误判为）："]
            lines += [f"- {gold_label} → {pred_label}：{count} 条" for gold_label, pred_label, count in confusions]
        lines += ["", f"token 用量：平均输入 {metrics['tokens']['avg_prompt_tokens']} / "
                  f"平均输出 {metrics['tokens']['avg_completion_tokens']}，"
                  f"时延 P50 {metrics['latency']['p50_ms']}ms / 均值 {metrics['latency']['mean_ms']}ms"]
        if metrics["multi_label_rows"]:
            lines.append(
                f"多标签样本 {metrics['multi_label_rows']} 条，多标签完全一致率 "
                f"{metrics['exact_match']:.4f}（主标签 macro-F1 {metrics['macro_f1']:.4f} 不含辅助标签）"
            )
        if metrics["fallbacks"]:
            lines.append(f"失败后用规则版顶上的样本：{metrics['fallbacks']} 条（口径已单独记录）")

    # 错例明细：只看主标签判错的样本，便于做失败归因
    lines += ["", "## 错例（主标签判错，每个后端最多 10 条）"]
    for name, metrics in results:
        shown: list[str] = []
        for record in metrics.get("records", []):
            gold = record["gold"][0] if record["gold"] else "chitchat"
            pred = record["pred"][0] if record["pred"] else UNRECOGNIZED
            if gold == pred:
                continue
            shown.append(
                f"| {record['id']} | {record['text'][:40]} | {gold} | {pred} | "
                f"{record.get('confidence', 0):.2f} |"
            )
            if len(shown) >= 10:
                break
        lines += ["", f"### {name}", ""]
        lines += (
            ["| id | 问句 | 真值 | 预测 | 置信度 |", "| --- | --- | --- | --- | --- |", *shown]
            if shown
            else ["主标签全对。"]
        )

    lines += ["", "## 结论与下一步", ""]
    lines += [f"- {note}" for note in notes] if notes else ["- 暂无"]
    lines += [
        "",
        "复现命令：",
        "",
        "```bash",
        command,
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入：{path}")


def build_notes(results: list[tuple[str, dict]], labels: list[str]) -> list[str]:
    """数据驱动的结论：不写空话，只写这次跑出来的事实。"""
    notes: list[str] = []
    for name, metrics in results:
        worst = sorted(
            ((label, item["f1"], item["support"]) for label, item in metrics["per_class"].items()),
            key=lambda item: item[1],
        )[:3]
        notes.append(
            f"{name}：macro-F1 {metrics['macro_f1']:.4f}，最弱三类 "
            + "、".join(f"{label}(F1 {f1:.2f}/真值 {support})" for label, f1, support in worst)
        )
        if metrics["errors"]:
            notes.append(
                f"{name}：{metrics['errors']} 条调用失败（{metrics['errors'] / metrics['queries']:.1%}），"
                "这些样本按判错计入指标；超时/限流居多时可调大 INTENT_TIMEOUT 或降低 --concurrency，"
                "重跑会自动补上（失败不写缓存）"
            )
        if metrics["queries"] and metrics["cached"] == metrics["queries"]:
            notes.append(
                f"{name}：本次全部命中缓存，P95 时延不可用；要拿真实时延请加 --no-cache 重跑"
            )
        if metrics["cost_per_1k"]:
            notes.append(
                f"{name}：千次调用 ¥{metrics['cost_per_1k']:.2f}，P95 时延 "
                f"{metrics['latency']['p95_ms']:.0f}ms —— 这两个数字就是本地小模型要打败的目标"
            )
    confusions = top_confusions(
        (results[-1][1]["confusion"] if results else {}), labels + [UNRECOGNIZED], 3
    )
    for gold_label, pred_label, count in confusions:
        notes.append(f"最需要盯的混淆：{gold_label} 被误判为 {pred_label}（{count} 条），先查 taxonomy 边界定义")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description="意图识别评测与后端对比")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--splits", nargs="*", default=["test"], help="要评测的集合：dev / test")
    parser.add_argument(
        "--backends",
        nargs="*",
        default=["rule", "fewshot"],
        choices=["rule", "fewshot", "local"],
    )
    parser.add_argument("--limit", type=int, default=0, help="每个集合最多评测多少条（0=全部）")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--shots", type=int, default=1, help="每类给几条例句")
    parser.add_argument("--model", default="", help="覆盖 INTENT_MODEL")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="本地 adapter 目录，默认 data/intent/models/v1（也可用 INTENT_LOCAL_DIR）",
    )
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument(
        "--local-cache",
        type=Path,
        default=DEFAULT_LOCAL_CACHE,
        help="本地推理的行级缓存，中断后重跑会自动接着算",
    )
    parser.add_argument("--fallback-rule", action="store_true", help="调用失败时用规则版顶上")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="指标 JSON（后端接口读它）")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    dataset_dir = resolve_dataset_dir(args.data_dir)
    taxonomy = load_taxonomy()
    labels = list(taxonomy.keys)
    valid = set(labels)

    splits = {name: load_split(dataset_dir, name, args.limit) for name in args.splits}
    splits = {name: rows for name, rows in splits.items() if rows}
    if not splits:
        raise SystemExit("没有可评测的数据，先跑 scripts/split_intent_dataset.py")

    config = load_intent_config()
    model_name = args.model or config.model
    print(f"数据集：{dataset_dir}")
    print("集合：" + "，".join(f"{name} {len(rows)} 条" for name, rows in splits.items()))
    if args.limit:
        print("抽样方式：按主标签分层抽样（每类轮流取，保证类别覆盖）")
        for name, rows in splits.items():
            distribution = Counter(primary_label(row) for row in rows)
            print(
                f"  {name} 抽样分布："
                + "，".join(f"{label} {count}" for label, count in sorted(distribution.items()))
            )
    print(f"意图体系：{len(labels)} 类；scikit-learn：{'可用' if SKLEARN_AVAILABLE else '未安装（F1 走兜底实现）'}")
    print(f"计价假设：输入 ¥{config.price_in_per_m}/百万、输出 ¥{config.price_out_per_m}/百万（可用 INTENT_PRICE_* 覆盖）\n")

    cache: dict | None = None
    if "fewshot" in args.backends:
        if args.no_cache:
            print("缓存：已禁用（--no-cache）")
        else:
            cache = {}
            if args.cache.exists():
                cache = json.loads(args.cache.read_text(encoding="utf-8"))
            print(f"缓存：{args.cache}（已有 {len(cache)} 条）")
        if config.configured or args.model:
            print(f"few-shot 模型：{model_name}，每类 {args.shots} 条例句，并发 {args.concurrency}")
        else:
            print("[warn] agent/.env 里没有可用的 LLM_API_KEY，few-shot 后端会被跳过")
        print()

    results: list[tuple[str, dict]] = []
    for backend in args.backends:
        for split_name, rows in splits.items():
            if backend == "rule":
                records = eval_rule(rows, valid)
                display = f"rule-based-v1（{split_name}）"
                metrics = summarize(records, taxonomy, config.price_in_per_m, config.price_out_per_m, backend)
                metrics["strategy"] = "rule-based-v1"
            elif backend == "fewshot":
                classifier_available = config.configured or args.model
                if not classifier_available:
                    print("[skip] few-shot：缺少 LLM_API_KEY（先填 agent/.env，或用 --backends rule）")
                    continue
                records = eval_fewshot(rows, valid, args, cache)
                display = f"few-shot {model_name}（{args.shots}-shot，{split_name}）"
                metrics = summarize(records, taxonomy, config.price_in_per_m, config.price_out_per_m, backend)
                metrics["strategy"] = "fewshot-llm"
                metrics["model"] = model_name
            elif backend == "local":
                records = eval_local(rows, valid, args, taxonomy)
                if records is None:
                    print(
                        f"[skip] local：{resolve_adapter_dir(args.local_dir)} 里没有 adapter"
                        "（先跑 scripts/train_intent.py，或用 --local-dir 指定路径）"
                    )
                    continue
                display = f"local LoRA（{split_name}）"
                metrics = summarize(records, taxonomy, 0.0, 0.0, backend)
                metrics["strategy"] = "local-lora"
                metrics["local_dir"] = str(resolve_adapter_dir(args.local_dir))
            else:  # pragma: no cover - argparse 已限制取值
                continue

            metrics["records"] = records
            metrics["split"] = split_name
            results.append((display, metrics))
            print(
                f"[{display}] macro-F1 {metrics['macro_f1']:.4f}，准确率 {metrics['accuracy']:.4f}，"
                f"P95 {metrics['latency']['p95_ms']:.0f}ms，千次成本 ¥{metrics['cost_per_1k']:.4f}"
            )

    if not results:
        raise SystemExit("没有任何后端跑成功，检查配置后重试")

    print()
    print_table(results)

    if cache is not None and not args.no_cache:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n缓存已更新：{args.cache}（{len(cache)} 条）")

    best_name, best_metrics = max(results, key=lambda item: item[1]["macro_f1"])
    print(f"\n最佳后端：{best_name}（macro-F1 {best_metrics['macro_f1']:.4f}）")
    print("\n逐类指标：")
    for label, item in best_metrics["per_class"].items():
        tag = "" if item["support"] else "（真值里没有该类）"
        precision = f"P {item['precision']:.3f}"
        recall = f"R {item['recall']:.3f}"
        f1 = f"F1 {item['f1']:.3f}"
        print(
            f"  {_pad(label, 18)}{_pad(precision, 10)}{_pad(recall, 10)}"
            f"{_pad(f1, 11)}真值 {item['support']}{tag}"
        )

    if not args.no_report:
        command = "python scripts/eval_intent.py " + " ".join(sys.argv[1:])
        sampling_note = (
            f"，本次为分层抽样（--limit {args.limit}），只能验证链路，不能当作基线数字"
            if args.limit
            else ""
        )
        write_report(
            args.report,
            dataset_dir,
            {name: len(rows) for name, rows in splits.items()},
            results,
            labels,
            command.strip(),
            build_notes(results, labels),
            sampling_note,
        )

    write_json_summary(args, dataset_dir, splits, results, config, model_name, cache)
    return None


def write_json_summary(
    args: argparse.Namespace,
    dataset_dir: Path,
    splits: dict[str, list[dict]],
    results: list[tuple[str, dict]],
    config,
    model_name: str,
    cache: dict | None,
) -> None:
    """写机器可读的指标 JSON（后端 /api/intent/versions 直接读它，不解析 markdown）。

    按 (后端, 集合, adapter) 合并而不是整体覆盖：分几次跑（test 三方 + dev 逐行）
    时，后一次不该把前一次的 rule / few-shot 行冲掉。
    """
    merged: dict[tuple, dict] = {}
    previous_splits: dict = {}
    if args.json.exists():
        try:
            previous = json.loads(args.json.read_text(encoding="utf-8"))
            for row in previous.get("backends") or []:
                merged[_backend_signature(row)] = row
            previous_splits = dict(previous.get("split_sizes") or {})
        except json.JSONDecodeError:
            merged, previous_splits = {}, {}
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_dir.name,
        "dataset_dir": str(dataset_dir),
        "split_sizes": {**previous_splits, **{name: len(rows) for name, rows in splits.items()}},
        "prices": {"in_per_m": config.price_in_per_m, "out_per_m": config.price_out_per_m},
        "shots": args.shots,
        "backends": [],
    }
    for display, metrics in results:
        payload["backends"].append(
            {
                "backend": metrics["backend"],
                "display": display,
                "split": metrics["split"],
                "strategy": metrics.get("strategy"),
                "model": metrics.get("model") or model_name,
                "local_dir": metrics.get("local_dir"),
                "queries": metrics["queries"],
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "exact_match": metrics["exact_match"],
                "multilabel": metrics["multilabel"],
                "latency": metrics["latency"],
                "cost_avg": metrics["cost_avg"],
                "cost_per_1k": metrics["cost_per_1k"],
                "errors": metrics["errors"],
                "cached": metrics["cached"],
                "per_class": metrics["per_class"],
            }
        )
    for row in payload["backends"]:
        merged[_backend_signature(row)] = row
    payload["backends"] = sorted(
        merged.values(), key=lambda item: ((item.get("backend") or ""), (item.get("split") or ""))
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"指标 JSON：{args.json}（累计 {len(payload['backends'])} 行："
          + "、".join(f"{row.get('backend')}/{row.get('split')}" for row in payload["backends"])
          + "）")

    # 记录本地推理用的 adapter：校准脚本靠它把结果挂回具体版本
    local_rows = [metrics for _, metrics in results if metrics["backend"] == "local"]
    if local_rows:
        # sidecar 必须跟着 --json 的目录走：早先写死成 eval/reports/ 下的真实路径，
        # 结果单元测试里传入的临时 --json 也把生产用的那份元信息覆盖成了占位值。
        meta_path = args.json.with_name("local_eval_meta.json")
        meta = {
            "adapter_dir": local_rows[-1].get("local_dir"),
            "split": local_rows[-1].get("split"),
            "written_at": payload["created_at"],
            "cache_entries": sum(1 for key in (cache or {}) if not str(key).startswith("_")),
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"本地推理元信息：{meta_path}")


def _backend_signature(row: dict) -> tuple:
    """同一行指标的标识：换 adapter 或换集合都算不同行。"""
    return (
        row.get("backend"),
        row.get("split"),
        row.get("local_dir"),
        row.get("model"),
    )


if __name__ == "__main__":
    main()
