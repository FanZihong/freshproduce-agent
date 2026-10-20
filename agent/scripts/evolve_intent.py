"""意图层自进化闭环（最小实现）：badcase 导出 → 追加训练数据 → 评测门禁。

刻意不做自动上线：闭环的定义是"数据飞轮 + 评测门禁"，任何改动都要过门禁才允许生效，
而"跑几小时训练"这种事交回给人（脚本只打印命令）。

数据有两个来源，`--source` 选：
- **dev 集**（默认）：错例来自本地模型的评测缓存，防泄漏口径见下；
- **trace 池**：人工标好的线上 badcase（`eval/datasets/trace_badcase_pool.jsonl`），
  线上对话是新数据，不涉及验收集泄漏。人工在池子里填的是自然语言（"转人工"），
  脚本先按体系的名字/别名精确匹配，匹配不上才关键词兜底，兜底的行会在报告里点名让人复核。

三种用法（在 agent/ 目录下执行）：

1) 导出 badcase 并生成 v3 训练数据（默认从 **dev** 导出，见下）：
   python scripts/evolve_intent.py --export --adapter-dir data/intent/models/v1
   python scripts/evolve_intent.py --export --source both          # dev 错例 + trace 池一起用
   python scripts/evolve_intent.py --export --source pool          # 只用人工标好的 trace 池

2) 训练（脚本会打印完整命令，不替你跑）：
   python scripts/train_intent.py --train-file data/intent/dataset_v1_balanced/train_sft_v3.jsonl ...

3) 门禁：比较基线与候选版本的指标，判定是否允许替换：
   python scripts/evolve_intent.py --gate --baseline-dir data/intent/models/v1 \
       --candidate-dir data/intent/models/v3

为什么默认只从 dev 导出：**test 是锁死的验收集**，拿它的错例去训练再拿它评测就是数据泄漏。
想用 test 必须显式加 `--allow-test-leakage`，脚本会在报告里写明这属于自欺欺人。
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
from agent.intent.baseline import build_training_system_prompt  # noqa: E402
from agent.intent.target import normalize_labels  # noqa: E402
from agent.intent.taxonomy import ALIASES, load_taxonomy  # noqa: E402
from scripts.eval_intent import row_cache_key  # noqa: E402
from scripts.prepare_sft_data import build_two_part  # noqa: E402

DATASET_DIR = DATA_DIR / "intent" / "dataset_v1_balanced"
REPORTS_DIR = AGENT_ROOT / "eval" / "reports"
DEFAULT_BASELINE_JSON = REPORTS_DIR / "intent_baseline.json"
DEFAULT_CACHE = REPORTS_DIR / "local_eval_cache.json"
DEFAULT_META = REPORTS_DIR / "local_eval_meta.json"
DEFAULT_REPORT = REPORTS_DIR / "intent_gate.md"
DEFAULT_MINING_REPORT = REPORTS_DIR / "intent_badcase_mining.md"
DEFAULT_POOL = AGENT_ROOT / "eval" / "datasets" / "trace_badcase_pool.jsonl"
# 门禁盯住的关键类：v1 的短板就这几个（quality_qa 0.52 / chitchat 0.60 / human_handoff 0.62）
KEY_CLASSES: tuple[str, ...] = ("quality_qa", "chitchat", "human_handoff")

# 人工在池子里填的是自然语言（"转人工"、"域外回答拒绝"），先按体系的名字/别名精确匹配，
# 匹配不上再按关键词兜底。**兜底命中的行会在报告里点名**——猜出来的标签不能当真值悄悄用。
LABEL_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("human_handoff", ("转人工", "人工客服", "handoff")),
    ("complaint", ("投诉", "曝光", "处罚", "差评")),
    ("bad_fruit_claim", ("坏果", "烂", "理赔", "赔付", "赔偿")),
    ("short_weight", ("缺斤少两", "少称", "分量", "重量", "称重")),
    ("urge_logistics", ("催单", "催物流", "催一下", "还没发货", "没发货")),
    ("query_order", ("查单", "查订单", "物流", "到哪了")),
    ("refund_consult", ("退款", "退货", "退钱")),
    ("invoice", ("发票", "开票")),
    ("quality_qa", ("品质", "知识问答", "保存", "储存", "怎么吃")),
    ("chitchat", ("域外", "闲聊", "无关", "兜底", "不予回答", "拒绝回答")),
)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_badcases(
    rows: list[dict], cache: dict, namespace: str, valid: set[str]
) -> list[dict]:
    """从逐行缓存里挑出"主标签判错"的样本，返回带真值的 badcase 列表。"""
    badcases: list[dict] = []
    for row in rows:
        entry = cache.get(row_cache_key(row.get("text", ""), row.get("context"), namespace))
        if not entry:
            continue
        gold = list(normalize_labels(row.get("labels") or [], valid))
        predicted = list(normalize_labels(entry.get("labels") or [], valid))
        if not gold or gold == predicted:
            continue
        badcases.append(
            {
                "id": row.get("id"),
                "text": row.get("text", ""),
                "context": row.get("context"),
                "labels": gold,
                "predicted": predicted,
                "confidence": entry.get("confidence"),
                "source": f"badcase:{row.get('source') or 'unknown'}",
            }
        )
    return badcases


def build_extra_training_rows(badcases: list[dict], taxonomy) -> list[dict]:
    """把 badcase 转成与训练数据同格式的 SFT 样本（标签用**真值**，不是模型的错判）。"""
    system_prompt = build_training_system_prompt(taxonomy, target_style="key", detail="catalog")
    return build_two_part(badcases, system_prompt, taxonomy, "key", 2)


# ---------------- trace 池：人工标好的线上 badcase ----------------


def build_label_index(taxonomy) -> dict[str, str]:
    """把体系里的 key、中文名、历史别名都映射到 key。"""
    index: dict[str, str] = {}
    for key in taxonomy.keys:
        index[key] = key
        name = taxonomy.name_of(key)
        if name:
            index[name] = key
    for alias, key in ALIASES.items():
        index.setdefault(alias, key)
    return index


def resolve_expected_label(raw: str, taxonomy) -> tuple[str | None, str]:
    """把人工填的期望标签翻译成体系 key；返回 (key, 匹配方式)。

    匹配不了就返回 (None, "")——**不猜**。猜错的标签会当成真值进训练集，比不收更糟。
    """
    text = str(raw or "").strip()
    if not text:
        return None, ""
    index = build_label_index(taxonomy)
    if text in index:
        return index[text], "名字/别名精确匹配"
    compact = text.replace(" ", "")
    for label, key in index.items():
        if label and label in compact:
            return key, f"包含体系名「{label}」"
    for key, keywords in LABEL_KEYWORDS:
        hits = [word for word in keywords if word in compact]
        if hits:
            return key, f"关键词兜底（{'/'.join(hits)}）"
    return None, ""


def pool_rows(path: Path) -> list[dict]:
    return load_jsonl(path)


def is_pool_row_labelled(row: dict) -> bool:
    """池子里这行算不算"标完了"：状态是 labeled，或三列里填了至少一列。"""
    return bool(row.get("status") == "labeled") or any(
        row.get(field) for field in ("expected_intent", "expected_action", "note")
    )


def pool_badcases(
    rows: list[dict], taxonomy, *, only_labelled: bool = True
) -> tuple[list[dict], list[dict]]:
    """把 trace 池转成训练样本可用的 badcase；返回 (可用的, 映射不了的)。

    `predicted` 用 trace 里记录的实际意图，`agrees_with_model` 标记"人工确认判对了"的行——
    这类行是正样本，不是错例，训练集里照样有用（等于给了同分布的正反馈），
    但报告里要说清，免得被误读成"模型又错了 N 条"。
    """
    valid = set(taxonomy.keys)
    usable: list[dict] = []
    unmapped: list[dict] = []
    for row in rows:
        if only_labelled and not is_pool_row_labelled(row):
            continue
        expected = str(row.get("expected_intent") or "").strip()
        if not expected:
            # 没填期望意图的行等于还没标，不算"映射失败"——别让它污染待修清单
            continue
        key, how = resolve_expected_label(expected, taxonomy)
        observed = row.get("observed") or {}
        predicted = list(normalize_labels([observed.get("intent")], valid))
        if not key:
            unmapped.append(
                {
                    "id": row.get("id"),
                    "text": row.get("text"),
                    "expected_intent": row.get("expected_intent"),
                    "observed_intent": observed.get("intent"),
                    "note": row.get("note"),
                }
            )
            continue
        usable.append(
            {
                "id": row.get("id"),
                "text": row.get("text"),
                "context": None,
                "labels": [key],
                "predicted": predicted,
                "confidence": observed.get("confidence"),
                "source": "trace",
                "match": how,
                "agrees_with_model": predicted == [key],
            }
        )
    return usable, unmapped


def merge_training_rows(base_rows: list[dict], extra_rows: list[dict]) -> tuple[list[dict], int]:
    """按文本去重后合并基础训练集与 badcase 样本，返回 (合并结果, 新增条数)。"""
    seen = {row.get("text") for row in base_rows}
    merged = list(base_rows)
    added = 0
    for row in extra_rows:
        if row.get("text") in seen:
            continue
        seen.add(row.get("text"))
        merged.append(row)
        added += 1
    return merged, added


def format_confusion_counts(counter: Counter, top: int = 5) -> str:
    """把 (真值, 错判) 的计数渲染成 "真值→错判×N" 的可读串。

    单独抽出来是因为这里踩过坑：Counter 的键是"元组套元组"，直接 join 会 TypeError。
    """
    parts: list[str] = []
    for (gold, predicted), count in counter.most_common(top):
        gold_text = "/".join(gold) or "（空）"
        predicted_text = "/".join(predicted) or "（空）"
        parts.append(f"{gold_text}→{predicted_text}×{count}")
    return "，".join(parts)


def evaluate_gate(
    baseline: dict,
    candidate: dict,
    key_classes: tuple[str, ...] = KEY_CLASSES,
    tolerance: float = 0.0,
) -> dict:
    """评测门禁：候选版本必须"整体不劣化 + 关键类不劣化"才算通过。

    只看 macro-F1 是不够的：一个版本可能靠高频类涨分，把 quality_qa / chitchat 这类
    短板悄悄换掉。所以关键类要单独设闸。
    """
    if not baseline or not candidate:
        return {
            "passed": False,
            "reason": "缺少可比对的评测行（先跑 eval_intent.py 生成 intent_baseline.json）",
            "rows": [],
        }

    baseline_f1 = float(baseline.get("macro_f1") or 0.0)
    candidate_f1 = float(candidate.get("macro_f1") or 0.0)
    rows: list[dict] = [
        {
            "metric": "macro-F1",
            "baseline": baseline_f1,
            "candidate": candidate_f1,
            "delta": round(candidate_f1 - baseline_f1, 4),
            "ok": candidate_f1 >= baseline_f1 - tolerance,
        }
    ]
    baseline_per_class = baseline.get("per_class") or {}
    candidate_per_class = candidate.get("per_class") or {}
    for label in key_classes:
        base_value = float((baseline_per_class.get(label) or {}).get("f1") or 0.0)
        cand_value = float((candidate_per_class.get(label) or {}).get("f1") or 0.0)
        rows.append(
            {
                "metric": f"{label} F1",
                "baseline": base_value,
                "candidate": cand_value,
                "delta": round(cand_value - base_value, 4),
                "ok": cand_value >= base_value - tolerance,
            }
        )
    failed = [row for row in rows if not row["ok"]]
    return {
        "passed": not failed,
        "reason": "全部门禁指标不劣化" if not failed else "以下指标劣化：" + "、".join(
            row["metric"] for row in failed
        ),
        "rows": rows,
        "baseline_row": baseline.get("display"),
        "candidate_row": candidate.get("display"),
    }


def find_row(payload: dict, local_dir: str, split: str = "test") -> dict | None:
    for row in payload.get("backends") or []:
        if row.get("backend") != "local" or row.get("split") != split:
            continue
        if local_dir and str(row.get("local_dir") or "").rstrip("/") != local_dir.rstrip("/"):
            continue
        return row
    return None


def default_split(explicit: str | None, *, for_gate: bool) -> str:
    """两个命令的默认集合不一样，别用一个 --split 糊在一起：

    - **导出** badcase 默认 dev：test 是锁死的验收集，拿它的错例去训练就是数据泄漏；
    - **门禁** 默认 test：那才是两个版本都评过的验收口径（dev 只是拟合集）。
    混用会造成"门禁找不到可比对的行"这种看起来像 bug 的现象。
    """
    if explicit:
        return explicit
    return "test" if for_gate else "dev"


def write_gate_report(path: Path, gate: dict, baseline_dir: str, candidate_dir: str) -> None:
    lines = [
        "# 意图版本门禁报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 基线：{baseline_dir}",
        f"- 候选：{candidate_dir}",
        f"- 判定：{'✅ 通过（允许替换）' if gate['passed'] else '❌ 不通过（保留基线）'}",
        f"- 理由：{gate['reason']}",
        "",
        "| 指标 | 基线 | 候选 | Δ | 是否达标 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in gate.get("rows") or []:
        lines.append(
            f"| {row['metric']} | {row['baseline']:.4f} | {row['candidate']:.4f} | "
            f"{row['delta']:+.4f} | {'✓' if row['ok'] else '✗'} |"
        )
    lines += [
        "",
        "门禁口径：macro-F1 **与**关键类（quality_qa / chitchat / human_handoff）都不劣化才放行；",
        "只看 macro-F1 会让「高频类涨分、短板悄悄退化」的版本蒙混过关。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_export(args: argparse.Namespace) -> int:
    taxonomy = load_taxonomy()
    valid = set(taxonomy.keys)
    sources = {"dev": ("dev",), "pool": ("pool",), "both": ("dev", "pool")}[args.source]

    dev_badcases: list[dict] = []
    adapter_dir = ""
    if "dev" in sources:
        args.split = default_split(args.split, for_gate=False)
        if args.split == "test" and not args.allow_test_leakage:
            raise SystemExit(
                "拒绝从 test 集导出 badcase：它是锁死的验收集，用它训练再评测等于自欺欺人。\n"
                "  用 --split dev，或确认无误后加 --allow-test-leakage"
            )
        cache = load_json(args.cache)
        meta = load_json(args.meta)
        adapter_dir = str(args.adapter_dir or meta.get("adapter_dir") or "")
        if not cache or not adapter_dir:
            raise SystemExit(
                f"缺少本地推理缓存或 adapter 信息（{args.cache} / {args.meta}）。\n"
                f"  先跑：python scripts/eval_intent.py --backends local --local-dir {adapter_dir or 'data/intent/models/v1'} --splits dev"
            )
        rows = load_jsonl(DATASET_DIR / f"{args.split}.jsonl")
        dev_badcases = select_badcases(rows, cache, adapter_dir, valid)
        print(
            f"从 {args.split} 集导出 badcase：{len(dev_badcases)}/{len(rows)} 条"
            f"（adapter：{adapter_dir}）"
        )
        print(
            "误判分布（真值→错判）："
            + format_confusion_counts(
                Counter(
                    (tuple(case["labels"]), tuple(case["predicted"])) for case in dev_badcases
                )
            )
        )

    pool_cases: list[dict] = []
    unmapped: list[dict] = []
    if "pool" in sources:
        if not args.pool.exists():
            raise SystemExit(
                f"找不到 trace 池：{args.pool}\n"
                "  先采几轮对话（run_cli.py）再跑 scripts/mine_traces.py，然后人工标注池子里的行"
            )
        raw_pool = pool_rows(args.pool)
        pool_cases, unmapped = pool_badcases(raw_pool, taxonomy)
        print(
            f"trace 池：{len(raw_pool)} 条 → 可用 {len(pool_cases)} 条、"
            f"映射不了 {len(unmapped)} 条"
        )
        for case in pool_cases:
            verdict = "一致（正样本）" if case["agrees_with_model"] else "不一致（错例）"
            print(
                f"  {case['id']}：人工期望 {case['labels'][0]}（{case['match']}），"
                f"模型当时判 {'/'.join(case['predicted']) or '（空）'} → {verdict}"
            )
        for row in unmapped:
            print(
                f"  [!] 映射不了：{row['id']} 的 expected_intent=「{row['expected_intent']}」"
                "，请改成体系里的类别名或 key"
            )

    if not dev_badcases and not pool_cases:
        raise SystemExit("两个来源都没拿到可用样本，检查 --source 与池子文件")

    extra_rows = build_extra_training_rows(dev_badcases + pool_cases, taxonomy)
    base_rows = load_jsonl(DATASET_DIR / "train_sft.jsonl")
    if not base_rows:
        raise SystemExit(f"找不到基础训练集：{DATASET_DIR / 'train_sft.jsonl'}")
    merged, added = merge_training_rows(base_rows, extra_rows)
    out_path = DATASET_DIR / f"train_sft_{args.tag}.jsonl"
    out_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in merged) + "\n", encoding="utf-8"
    )

    # 报告：把这次挖掘写下来，作为"数据飞轮"的一环留档
    lines = [
        f"# badcase 挖掘与 v3 训练数据（{args.tag}）",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 来源：{args.source}"
        + (
            f"（dev 用 {args.split} 集，"
            + ("⚠️ 含 test，已确认数据泄漏" if args.split == "test" else "未污染验收集")
            + "；trace 池是线上新数据，不涉及泄漏）"
            if "dev" in sources
            else "（trace 池：线上新数据，不涉及验收集泄漏）"
        ),
        f"- adapter：{adapter_dir or '（本次没跑 dev，未用到）'}",
        f"- dev 错例：{len(dev_badcases)} 条；trace 池可用：{len(pool_cases)} 条"
        f"（映射不了 {len(unmapped)} 条）",
        f"- 生成的训练样本 {len(extra_rows)} 条；去重后实际新增 {added} 条"
        + ("（新增少于生成数说明与基础训练集重文）" if added < len(extra_rows) else ""),
        f"- 基础训练集 {len(base_rows)} 条 → 合并后 {len(merged)} 条：`{out_path.name}`",
        "",
    ]

    if dev_badcases:
        lines += [
            "## 一、dev 集误判样本（真值 → 模型错判）",
            "",
            "| id | 问句 | 真值 | 模型错判 | 置信度 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for case in dev_badcases[:30]:
            lines.append(
                f"| {case['id']} | {str(case['text'])[:30]} | {'/'.join(case['labels'])} | "
                f"{'/'.join(case['predicted']) or '（空）'} | {case['confidence']} |"
            )

    if "pool" in sources:
        lines += [
            "",
            "## 二、trace 池（人工标注的线上样本）",
            "",
            "> 人工填的是自然语言，脚本翻译成体系 key 的方式记在「匹配方式」列。"
            "凡是走「关键词兜底」的行都要人再过一眼。",
            "",
            "| id | 用户原话 | 人工期望 → key | 匹配方式 | 模型当时判定 | 结果 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for case in pool_cases:
            lines.append(
                f"| {case['id']} | {str(case['text'])[:30]} | {case['labels'][0]} | "
                f"{case['match']} | {'/'.join(case['predicted']) or '（空）'} | "
                f"{'一致（正样本）' if case['agrees_with_model'] else '不一致（错例）'} |"
            )
        if unmapped:
            lines += [
                "",
                "### 映射不了、本次没收进训练集的行",
                "",
                "| id | 用户原话 | expected_intent 填的是 | 模型当时判定 |",
                "| --- | --- | --- | --- |",
            ]
            for row in unmapped:
                lines.append(
                    f"| {row['id']} | {str(row['text'])[:30]} | {row['expected_intent']} | "
                    f"{row['observed_intent'] or '（空）'} |"
                )
            lines += [
                "",
                "把这些行改成体系里的类别名（例如 `转人工` / `闲聊与域外`）或 key（例如 `human_handoff`）后重跑。",
            ]
    lines += [
        "",
        "## 下一步（训练完成后跑门禁）",
        "",
        "```bash",
        f"python scripts/train_intent.py --train-file {out_path} \\",
        f"  --prompt-detail catalog --max-len 0 --epochs 1 --save-steps 100 --eval-limit 0 \\",
        f"  --out-dir data/intent/models/{args.tag}",
        f"python scripts/eval_intent.py --backends local --local-dir data/intent/models/{args.tag} --splits test",
        f"python scripts/evolve_intent.py --gate --baseline-dir {adapter_dir} --candidate-dir data/intent/models/{args.tag}",
        "```",
        "",
    ]
    DEFAULT_MINING_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"\n训练数据：{out_path}（生成 {len(extra_rows)} 条，去重后新增 {added} 条）"
    )
    print(f"挖掘报告：{DEFAULT_MINING_REPORT}")
    if "pool" in sources and pool_cases:
        agree = sum(1 for case in pool_cases if case["agrees_with_model"])
        print(
            f"trace 池贡献：{len(pool_cases) - agree} 条错例 + {agree} 条"
            "人工确认判对的正样本（正样本也有用，但别当成模型改进了）"
        )
    print("\n下一步训练命令：")
    print(
        f"  python scripts/train_intent.py --train-file {out_path} --prompt-detail catalog "
        f"--max-len 0 --epochs 1 --save-steps 100 --eval-limit 0 --out-dir data/intent/models/{args.tag}"
    )
    return 0


def run_gate(args: argparse.Namespace) -> int:
    payload = load_json(args.compare_file)
    args.split = default_split(args.split, for_gate=True)
    baseline = find_row(payload, str(args.baseline_dir), args.split)
    candidate = find_row(payload, str(args.candidate_dir), args.split)
    gate = evaluate_gate(baseline or {}, candidate or {}, tolerance=args.tolerance)

    print(f"门禁比较（{args.split} 集）")
    if not gate["rows"]:
        print(f"  ✗ {gate['reason']}")
        return 1
    for row in gate["rows"]:
        print(
            f"  {'✓' if row['ok'] else '✗'} {row['metric']:<20} 基线 {row['baseline']:.4f} → "
            f"候选 {row['candidate']:.4f}（{row['delta']:+.4f}）"
        )
    print(f"\n判定：{'✅ 通过，允许替换' if gate['passed'] else '❌ 不通过，保留基线'}")
    print(f"理由：{gate['reason']}")
    write_gate_report(args.report, gate, str(args.baseline_dir), str(args.candidate_dir))
    print(f"门禁报告：{args.report}")
    return 0 if gate["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="意图层自进化闭环：badcase 导出 + 评测门禁")
    parser.add_argument("--export", action="store_true", help="导出 badcase 并生成追加训练数据")
    parser.add_argument("--gate", action="store_true", help="比较两个版本，判定是否允许替换")
    parser.add_argument(
        "--split",
        default=None,
        help="数据集：导出 badcase 默认 dev（防泄漏），门禁默认 test（验收口径）",
    )
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--tag", default="v3", help="候选版本名（用于输出文件与目录名）")
    parser.add_argument(
        "--source",
        choices=["dev", "pool", "both"],
        default="dev",
        help="数据来源：dev 集错例 / trace 池（人工标注）/ 两者都要",
    )
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL, help="trace 候选池（jsonl）")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--allow-test-leakage", action="store_true")
    parser.add_argument("--compare-file", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--baseline-dir", type=Path, default=DATA_DIR / "intent" / "models" / "v1")
    parser.add_argument("--candidate-dir", type=Path, default=DATA_DIR / "intent" / "models" / "v3")
    parser.add_argument("--tolerance", type=float, default=0.0, help="允许的劣化幅度（默认 0，即不允许）")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    if args.gate:
        return run_gate(args)
    return run_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
