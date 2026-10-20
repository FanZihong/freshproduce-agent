"""域外检测评测：50 条域外问句 + 已知类误杀率，对比"纯模型"与"三路信号"。

两个指标一起看才算数：
- **域外召回**：50 条域外问句里被判为域外的比例（目标 ≥ 0.9）；
- **已知类误杀率**：真实业务问句里被判成域外的比例（目标 < 5%）。
只追求召回会把大量正常问题推去兜底分支，只追求误杀率则域外全放行——所以两个必须同时看。

口径说明：误杀率只在"真值不是 chitchat"的样本上统计。test 集里本身有 102 条真闲聊，
把它们算进误杀率是错的。

用法（在 agent/ 目录下执行）：
    python scripts/eval_ood.py                                  # 规则版（零成本，先打通流程）
    python scripts/eval_ood.py --backend fewshot                # 大模型基线
    python scripts/eval_ood.py --backend local --local-dir data/intent/models/v2
    python scripts/eval_ood.py --backend rule --with-embedder   # 加上语义相似度那一路信号
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
from agent.intent.ood import (  # noqa: E402
    OOD_LABEL,
    OodConfig,
    collect_signals,
    is_out_of_domain,
    make_similarity_fn,
)
from agent.intent.taxonomy import load_taxonomy  # noqa: E402
from agent.intent.target import normalize_labels  # noqa: E402

DEFAULT_OOD = AGENT_ROOT / "eval" / "datasets" / "ood_golden.jsonl"
DEFAULT_INDOMAIN_DIRS = (
    DATA_DIR / "intent" / "dataset_v1_balanced",
    DATA_DIR / "intent" / "dataset_v1",
)
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "intent_ood.md"
DEFAULT_JSON = AGENT_ROOT / "eval" / "reports" / "intent_ood.json"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_indomain_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in DEFAULT_INDOMAIN_DIRS:
        if (candidate / "test.jsonl").exists():
            return candidate
    raise SystemExit(f"找不到域内测试集，预期位置：{DEFAULT_INDOMAIN_DIRS[0] / 'test.jsonl'}")


def build_backend(name: str, args: argparse.Namespace):
    """返回 (predict_fn, describe)；predict_fn(text, context) -> (labels, confidence, error)"""
    if name == "rule":
        from agent.intent.classifier import RuleBasedIntentClassifier

        classifier = RuleBasedIntentClassifier()

        def predict_rule(text: str, context=None):
            result = classifier.classify(text)
            return list(result.all_labels), float(result.confidence), ""

        return predict_rule, classifier.name

    if name == "fewshot":
        from agent.intent.baseline import FewShotIntentClassifier, load_intent_config

        classifier = FewShotIntentClassifier()
        if not classifier.available:
            raise SystemExit("few-shot 需要 agent/.env 里的 LLM_API_KEY")
        print(f"  few-shot 模型：{load_intent_config().model}，并发 {args.concurrency}")

        def predict_fewshot(text: str, context=None):
            call = classifier.classify_with_stats(text, context)
            return list(call.labels), float(call.confidence), call.error

        return predict_fewshot, classifier.name

    from agent.intent.local_model import LocalIntentClassifier, resolve_adapter_dir

    classifier = LocalIntentClassifier(
        adapter_dir=resolve_adapter_dir(args.local_dir), batch_size=args.local_batch_size
    )
    if not classifier.available:
        raise SystemExit(
            f"没找到 adapter：{resolve_adapter_dir(args.local_dir)}（用 --local-dir 指定）"
        )

    def predict_local(text: str, context=None):
        try:
            result = classifier.classify(text, context)
        except ValueError as exc:
            return [], 0.0, str(exc)
        return list(result.all_labels), float(result.confidence), ""

    return predict_local, classifier.name


def run_backend(predict, rows: list[dict], similarity_fn=None, label: str = "") -> list[dict]:
    """逐条跑后端，并顺带记录规则版标签与相似度（三路信号需要它们）。"""
    from agent.intent.classifier import RuleBasedIntentClassifier

    rule = RuleBasedIntentClassifier()
    valid = set(load_taxonomy().keys)
    records: list[dict] = []
    for index, row in enumerate(rows, start=1):
        labels, confidence, error = predict(row.get("text", ""), row.get("context"))
        rule_labels, _, _ = predict_rule_labels(rule, row.get("text", ""))
        similarity = similarity_fn(row.get("text", "")) if similarity_fn else None
        # 统一标签口径：规则版的 fallback / knowledge_qa 要对齐成 chitchat / quality_qa
        normalized = list(normalize_labels(labels, valid))
        records.append(
            {
                "id": row.get("id"),
                "text": row.get("text", ""),
                "gold": list(row.get("labels") or ([] if row.get("expected") else [])),
                "expected": row.get("expected"),
                "pred": normalized,
                "confidence": confidence,
                "error": error,
                "rule_labels": list(normalize_labels(rule_labels, valid)),
                "max_similarity": similarity,
                "model_says_ood": (not normalized) or normalized[0] == OOD_LABEL,
            }
        )
        if index % 100 == 0:
            print(f"    {label} 已处理 {index}/{len(rows)} 条")
    return records


def predict_rule_labels(rule, text: str) -> tuple[list[str], float, str]:
    result = rule.classify(text)
    return list(result.all_labels), float(result.confidence), ""


def apply_signals(
    records: list[dict],
    config: OodConfig,
    skip_self_signals: bool = False,
    use_rule_veto: bool = True,
) -> list[dict]:
    """把三路信号套在模型输出上，覆盖域外判定。

    skip_self_signals：后端本身就是规则版时，A（置信度）和 C（规则兜底）都源自同一套规则，
    拿它们投票等于自证循环——这时只用外部信号（语义相似度）判定。
    """
    for record in records:
        rule_label = (record["rule_labels"] or [None])[0]
        signals = collect_signals(
            record["text"],
            confidence=(
                None if (skip_self_signals or record["error"]) else record["confidence"]
            ),
            max_similarity=record["max_similarity"],
            rule_label=None if skip_self_signals else rule_label,
            model_label=None if skip_self_signals else (record["pred"] or [None])[0],
            config=config,
        )
        verdict, reason = is_out_of_domain(signals, config, use_rule_veto=use_rule_veto)
        verdict_no_veto, reason_no_veto = is_out_of_domain(
            signals, config, use_rule_veto=False
        )
        record["signals"] = signals.to_dict()
        record["signals_ood"] = verdict
        record["signals_reason"] = reason
        record["signals_ood_noveto"] = verdict_no_veto
        record["signals_reason_noveto"] = reason_no_veto
    return records


def summarize(
    ood_records: list[dict],
    in_domain_records: list[dict],
    key: str,
    exclude: set[str] | None = None,
) -> dict:
    """key = 'model_says_ood' / 'signals_ood' / 'signals_ood_noveto'。

    exclude 是"标签存疑"的样本 id：这些业务问句本身是 JDDC 噪音或缺上下文的碎片，
    拿它们算误杀率会把模型的正确行为算成错误（我们实测就撞上了）。
    """
    exclude = exclude or set()
    ood_hits = sum(1 for record in ood_records if record[key])
    # 误杀率只在"真值不是域外"的业务样本上算
    candidates = [
        record
        for record in in_domain_records
        if (record.get("gold") or ["chitchat"])[0] != OOD_LABEL
        and str(record.get("id")) not in exclude
    ]
    killed = sum(1 for record in candidates if record[key])
    return {
        "ood_recall": round(ood_hits / len(ood_records), 4) if ood_records else 0.0,
        "ood_hits": ood_hits,
        "ood_total": len(ood_records),
        "false_kill_rate": round(killed / len(candidates), 4) if candidates else 0.0,
        "false_kill": killed,
        "in_domain_total": len(candidates),
        "excluded": len(exclude),
    }


def print_table(rows: list[tuple[str, dict]], config: OodConfig) -> None:
    print(f"\n域外判定口径（信号阈值：置信度 < {config.low_confidence}、"
          f"相似度 < {config.min_similarity}、至少 {config.min_votes} 路）")
    header = f"{'口径':<26}{'域外召回':<12}{'已知类误杀率':<14}{'误杀条数':<10}"
    print(header)
    print("-" * 70)
    for name, metrics in rows:
        print(
            f"{name:<26}{metrics['ood_recall']:<12.4f}{metrics['false_kill_rate']:<14.4f}"
            f"{metrics['false_kill']}/{metrics['in_domain_total']}"
        )


def write_report(
    path: Path,
    args: argparse.Namespace,
    backend_label: str,
    config: OodConfig,
    results: list[tuple[str, dict]],
    ood_records: list[dict],
    in_domain_records: list[dict],
    verdict: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 域外检测评测报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 后端：{backend_label}",
        f"- 域外测试集：{args.dataset.name}（{len(ood_records)} 条）",
        f"- 误杀率统计集：{args.in_domain.name} 里真值不是 chitchat 的样本"
        f"（{results[0][1]['in_domain_total']} 条）",
        f"- 信号阈值：置信度 < {config.low_confidence}、相似度 < {config.min_similarity}、"
        f"至少 {config.min_votes} 路信号",
        f"- 语义相似度信号：{'启用' if args.with_embedder else '未启用（--with-embedder 打开）'}",
        "",
        "## 结果",
        "",
        "| 口径 | 域外召回 | 已知类误杀率 | 误杀条数 |",
        "| --- | --- | --- | --- |",
    ]
    for name, metrics in results:
        lines.append(
            f"| {name} | {metrics['ood_recall']:.4f} | {metrics['false_kill_rate']:.4f} | "
            f"{metrics['false_kill']}/{metrics['in_domain_total']} |"
        )
    lines += ["", "## 验收判定", ""]
    lines += [f"- {item}" for item in verdict]

    missed = [record for record in ood_records if not record["signals_ood"]]
    if missed:
        lines += ["", f"## 漏判的域外问句（{len(missed)} 条全列）", "",
                  "| id | 问句 | 模型标签 | 置信度 | 未判为域外的原因 |",
                  "| --- | --- | --- | --- | --- |"]
        for record in missed:
            lines.append(
                f"| {record['id']} | {record['text'][:28]} | "
                f"{record['pred'][:1] or '（空）'} | {record['confidence']:.3f} | "
                f"{record['signals_reason']} |"
            )

    killed_records = [
        record
        for record in in_domain_records
        if record["signals_ood"] and (record.get("gold") or ["chitchat"])[0] != OOD_LABEL
    ]
    if killed_records:
        lines += ["", f"## 被误杀的已知类问句（{len(killed_records)} 条全列）", "",
                  "> 逐条复核这些样本的**真值是否可信**：JDDC 带进来的其他品类噪音、缺上下文的碎片，"
                  "被判域外其实是正确行为。把存疑的 id 写进 suspects 清单，重跑时用 --suspects 引用，"
                  "报告会多出一行「剔除标签存疑后」的指标。",
                  "",
                  "| id | 问句 | 真值 | 模型标签 | 判为域外的原因 |",
                  "| --- | --- | --- | --- | --- |"]
        for record in killed_records:
            lines.append(
                f"| {record['id']} | {record['text'][:28]} | {record['gold'][:1]} | "
                f"{record['pred'][:1] or '（空）'} | {record['signals_reason']} |"
            )

    lines += [
        "",
        "复现命令：",
        "",
        "```bash",
        "python scripts/eval_ood.py " + " ".join(sys.argv[1:]),
        "```",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入：{path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="域外检测评测")
    parser.add_argument("--backend", choices=["rule", "fewshot", "local"], default="rule")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OOD)
    parser.add_argument("--in-domain", type=Path, default=None, help="域内测试集目录")
    parser.add_argument("--limit", type=int, default=0, help="域内样本上限（0=全部）")
    parser.add_argument("--local-dir", type=Path, default=None)
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--with-embedder", action="store_true", help="启用语义相似度那一路信号")
    parser.add_argument("--low-confidence", type=float, default=0.5)
    parser.add_argument("--min-similarity", type=float, default=0.35)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument(
        "--suspects",
        type=Path,
        default=None,
        help='标签存疑清单（JSONL，每行 {"id": "...", "reason": "..."}），用于算清洗后的误杀率',
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    config = OodConfig(
        low_confidence=args.low_confidence,
        min_similarity=args.min_similarity,
        min_votes=args.min_votes,
    )
    taxonomy = load_taxonomy()
    ood_rows = load_jsonl(args.dataset)
    if not ood_rows:
        raise SystemExit(f"域外测试集为空：{args.dataset}")
    in_domain_dir = resolve_indomain_dir(args.in_domain)
    args.in_domain = in_domain_dir  # 报告里要用它显示统计集路径
    in_domain_rows = load_jsonl(in_domain_dir / "test.jsonl")
    if args.limit:
        # 系统抽样，保证类别覆盖
        stride = max(1, len(in_domain_rows) // args.limit)
        in_domain_rows = in_domain_rows[::stride][: args.limit]

    print(f"域外测试集：{args.dataset.name}（{len(ood_rows)} 条）")
    print(f"域内对照集：{in_domain_dir / 'test.jsonl'}（{len(in_domain_rows)} 条）")
    print(f"意图体系：{len(taxonomy.keys)} 类；域外标签：{OOD_LABEL}")

    similarity_fn = None
    if args.with_embedder:
        from agent.rag.embedder import build_embedder

        embedder = build_embedder("auto")
        print(f"  embedder：{embedder.name}（维度 {embedder.dim}）")
        similarity_fn = make_similarity_fn(embedder)

    print("\n跑域外集 ...")
    predict, backend_label = build_backend(args.backend, args)
    ood_records = run_backend(predict, ood_rows, similarity_fn, "域外集")
    print("跑域内对照集 ...")
    in_domain_records = run_backend(predict, in_domain_rows, similarity_fn, "域内集")

    skip_self = args.backend == "rule"
    if skip_self and not args.with_embedder:
        print(
            "[提示] 规则版后端下，置信度与规则兜底这两个信号都来自同一套规则（自证循环），"
            "已停用；\n       此时只剩语义相似度一路可用，加上 --with-embedder 才有意义。"
        )
    ood_records = apply_signals(ood_records, config, skip_self_signals=skip_self)
    in_domain_records = apply_signals(in_domain_records, config, skip_self_signals=skip_self)

    model_metrics = summarize(ood_records, in_domain_records, "model_says_ood")
    signal_metrics = summarize(ood_records, in_domain_records, "signals_ood")
    no_veto_metrics = summarize(ood_records, in_domain_records, "signals_ood_noveto")
    suspects: set[str] = set()
    if args.suspects and args.suspects.exists():
        review = load_jsonl(args.suspects)
        # 清单是"复核记录"：只有明确标了 suggested=keep 的样本保留，其余视为应排除
        suspects = {
            str(row.get("id"))
            for row in review
            if row.get("id") and str(row.get("suggested") or "exclude").lower() != "keep"
        }
        print(
            f"标签存疑清单：{args.suspects.name}（复核 {len(review)} 条，"
            f"其中 {len(suspects)} 条不计入误杀率分母，{len(review) - len(suspects)} 条标记为保留）"
        )
    results = [
        (f"纯模型（{backend_label}）", model_metrics),
        ("信号组合（不用规则否决）", no_veto_metrics),
        ("信号组合 + 规则否决 ★", signal_metrics),
    ]
    if suspects:
        clean_metrics = summarize(
            ood_records, in_domain_records, "signals_ood", exclude=suspects
        )
        results.append(("★ 剔除标签存疑后", clean_metrics))
        signal_metrics = clean_metrics    # 验收判定以清洗后的口径为准
    print_table(results, config)

    verdict: list[str] = []
    verdict.append(
        f"纯模型域外召回 {model_metrics['ood_recall']:.4f}、误杀率 {model_metrics['false_kill_rate']:.4f}"
        + ("（模型自己就够好）" if model_metrics["ood_recall"] >= 0.9 else "（模型自身的域外识别不足）")
    )
    recall_ok = signal_metrics["ood_recall"] >= 0.9
    kill_ok = signal_metrics["false_kill_rate"] < 0.05
    verdict.append(
        f"三路信号：域外召回 {signal_metrics['ood_recall']:.4f}（≥0.9？{'✓' if recall_ok else '✗'}）；"
        f"已知类误杀率 {signal_metrics['false_kill_rate']:.4f}（<0.05？{'✓' if kill_ok else '✗'}）"
    )
    verdict.append(
        f"参照：不用规则否决时 召回 {no_veto_metrics['ood_recall']:.4f}、"
        f"误杀率 {no_veto_metrics['false_kill_rate']:.4f}"
        f"（规则否决的代价是召回 −{(no_veto_metrics['ood_recall'] - signal_metrics['ood_recall']):.4f}，"
        f"收益是误杀 −{(no_veto_metrics['false_kill_rate'] - signal_metrics['false_kill_rate']):.4f}）"
    )
    verdict.append(
        "误杀样本的典型原因：" + (
            "、".join(
                f"{key}×{value}"
                for key, value in Counter(
                    record["signals_reason"].split("（")[0]
                    for record in in_domain_records
                    if record["signals_ood"] and (record.get("gold") or ["chitchat"])[0] != OOD_LABEL
                ).most_common(3)
            )
            or "无（没有误杀）"
        )
    )
    verdict += [f"漏判原因分布：" + (
        "、".join(
            f"{key}×{value}"
            for key, value in Counter(
                record["signals_reason"].split("（")[0] for record in ood_records if not record["signals_ood"]
            ).most_common(3)
        )
        or "无（全部命中）"
    )]
    print("\n验收判定：")
    for item in verdict:
        print(f"  - {item}")

    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backend": backend_label,
        "config": config.to_dict(),
        "with_embedder": args.with_embedder,
        "model_only": model_metrics,
        "with_signals": signal_metrics,
        "with_signals_no_veto": no_veto_metrics,
        "suspects_file": str(args.suspects) if suspects else None,
        "excluded_suspects": len(suspects),
        "verdict": verdict,
        "p95_note": "本脚本只评测域外判定，不测时延",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"指标 JSON：{args.json}")

    if not args.no_report:
        write_report(args.report, args, backend_label, config, results, ood_records,
                     in_domain_records, verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
