"""用检索错例驱动 reranker / 检索配置的选择：拿"救回了几条错例"当判断依据。

为什么要单独一个脚本：光看 recall@5 会误导人。检索错例是有限的几条（比如 q014、q018），
能救回它们的配置才是真有用；总指标涨 0.03 但错例一条没救回来，对用户体验等于没变。
所以对比表除了总指标，还专门写"把当前配置救不回来的错例救回了几条"。

同时如实区分两类结论（这是最容易含糊过去的地方）：
- **可直接切换**：生产链路是"混合检索 + 规则改写 + FAQ_RERANK 重排器"，
  所以能靠改环境变量生效的只有 `FAQ_RERANK=score|bge` 两档；
- **需要改代码**：纯向量路、融合权重、RRF k、分块策略这些目前没有环境开关，
  报告里会标出来，但不会假装改一行 .env 就能上线。

用法（在 agent/ 目录下执行）：
    python scripts/iterate_rerank.py                      # 用已有 retrieval_baseline.json 对比
    python scripts/iterate_rerank.py --refresh            # 先重跑一遍检索再对比
    python scripts/iterate_rerank.py --with-cross-encoder # 把 BGE 精排也拉进对比（要 1.1GB 权重）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

REPORTS = AGENT_ROOT / "eval" / "reports"
DEFAULT_BASELINE = REPORTS / "retrieval_baseline.json"
DEFAULT_JSON = REPORTS / "retrieval_rerank.json"
DEFAULT_REPORT = REPORTS / "rerank_iteration.md"

# 生产链路实际用的那一档（FAQ_RERANK=score -> 混合 + 规则改写 + 融合分数重排）
DEFAULT_CURRENT = "hybrid+rw+score"
# 不下载 CrossEncoder 时的对比范围：混合/向量两路 + 分数重排
DEFAULT_CONFIGS: tuple[str, ...] = ("hybrid+rw+score", "hybrid+rw", "vector+rw+score", "vector+rw")
CROSS_ENCODER_CONFIGS: tuple[str, ...] = ("hybrid+rw+bge", "vector+rw+bge")

# 环境变量能切换的配置 -> FAQ_RERANK 取值。
# **只有 hybrid+* 在这张表里**：生产链路（tools/faq.py -> get_retriever）固定走混合检索，
# 能靠环境变量换的只有重排器那一档；纯向量那些配置要改代码才能上线，
# 把它们也算成"可切换"会给出一个根本生效不了的结论。
RERANKER_BY_CONFIG: dict[str, str] = {
    "hybrid+rw": "score",
    "hybrid+rw+score": "score",
    "hybrid+rw+bge": "bge",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def config_row(payload: dict, config: str) -> dict | None:
    for item in payload.get("configs") or []:
        if item.get("config") == config:
            return item
    return None


def config_metrics(payload: dict, config: str) -> dict:
    """取某个配置的指标与未命中清单；缺字段一律给安全默认值。"""
    row = config_row(payload, config) or {}
    metrics = row.get("metrics") or {}
    misses = row.get("misses") or []
    return {
        "config": config,
        "label": row.get("label") or config,
        "recall@5": float(metrics.get("recall@5") or 0.0),
        "recall@1": float(metrics.get("recall@1") or 0.0),
        "mrr": float(metrics.get("mrr") or 0.0),
        "miss_ids": sorted(str(item.get("id")) for item in misses),
        "misses": misses,
        "available": bool(row),
    }


def rescue_count(base_miss_ids: list[str], candidate_miss_ids: list[str]) -> int:
    """候选配置把当前配置救不回来的错例救回了几条。"""
    still_missing = set(candidate_miss_ids)
    return sum(1 for item in base_miss_ids if item not in still_missing)


def rank_configs(rows: list[dict]) -> list[dict]:
    """按 (recall@5, recall@1, MRR) 从好到差排序；没数据的排最后。"""
    return sorted(
        rows,
        key=lambda row: (
            0 if row["available"] else 1,
            -row["recall@5"],
            -row["recall@1"],
            -row["mrr"],
        ),
    )


def build_advice(current: dict, candidates: list[dict], *, with_cross_encoder: bool) -> dict:
    """给出"要不要换、换什么、怎么换"的结论，并区分能不能靠环境变量生效。"""
    ranked = rank_configs([row for row in candidates if row["available"]])
    if not ranked or not current["available"]:
        return {"ok": False, "reason": "缺少可对比的配置数据"}

    best = ranked[0]
    switchable = best["config"] in RERANKER_BY_CONFIG and best["config"] != current["config"]
    no_worse = best["recall@5"] >= current["recall@5"] and best["recall@1"] >= current["recall@1"]
    rescued = rescue_count(current["miss_ids"], best["miss_ids"])

    if best["config"] == current["config"]:
        verdict = f"保持现状：{current['config']} 在候选里已经是最好的"
    elif switchable and no_worse:
        verdict = (
            f"建议切到 {best['config']}（改环境变量即可生效）："
            f"FAQ_RERANK={RERANKER_BY_CONFIG[best['config']]}"
            + (
                "；注意 bge 精排要下载约 1.1GB 权重，每条 query 也更慢"
                if RERANKER_BY_CONFIG[best["config"]] == "bge"
                else ""
            )
        )
    elif no_worse:
        verdict = (
            f"{best['config']} 指标更好，但它不是环境变量能切换的档位"
            "（涉及检索路径/权重/分块，需要改代码），先记进候选清单"
        )
    else:
        verdict = (
            f"保持现状：{best['config']} 虽然指标最高，但 recall@5 或 recall@1 比当前配置低，"
            "别为了 MRR 换一个召回更差的配置"
        )
    return {
        "ok": True,
        "current": current,
        "best": best,
        "ranked": ranked,
        "rescued": rescued,
        "verdict": verdict,
        "switchable": switchable,
        "with_cross_encoder": with_cross_encoder,
    }


def run_comparison(configs: tuple[str, ...], json_path: Path, report_path: Path) -> int:
    """调 eval_retrieval.py 重新跑一遍对比，产出带 misses 的 JSON。"""
    command = [
        sys.executable,
        str(AGENT_ROOT / "scripts" / "eval_retrieval.py"),
        "--configs",
        *configs,
        "--report",
        str(report_path),
        "--json",
        str(json_path),
    ]
    print("$ " + " ".join(command))
    completed = subprocess.run(
        command,
        cwd=str(AGENT_ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in (completed.stdout or "").strip().splitlines()[-12:]:
        print("  " + line)
    if completed.returncode != 0:
        print("  [!] 检索对比失败：")
        print("\n".join("  " + line for line in (completed.stderr or "").splitlines()[-6:]))
    return completed.returncode


def write_report(path: Path, advice: dict, current_config: str) -> None:
    current = advice["current"]
    lines = [
        "# reranker / 检索配置迭代记录",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 当前生产口径：{current_config}（FAQ_RERANK=score）",
        f"- 对比范围：{'、'.join(row['config'] for row in advice['ranked'])}",
        "",
        "## 一、对比表（按 recall@5 / recall@1 / MRR 排序）",
        "",
        "| 配置 | recall@5 | recall@1 | MRR | 未命中 | 救回当前错例 | 可改环境变量 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in advice["ranked"]:
        rescue = rescue_count(current["miss_ids"], row["miss_ids"])
        switchable = "✓" if row["config"] in RERANKER_BY_CONFIG else "✗（要改代码）"
        lines.append(
            f"| {row['config']} | {row['recall@5']:.4f} | {row['recall@1']:.4f} | "
            f"{row['mrr']:.4f} | {len(row['miss_ids'])} | {rescue} | "
            f"{'—（就是它）' if row['config'] == current_config else switchable} |"
        )

    lines += [
        "",
        "## 二、当前配置救不回来的错例（这就是靶子）",
        "",
        f"共 {len(current['miss_ids'])} 条：{'、'.join(current['miss_ids']) or '无'}",
        "",
    ]
    for item in current["misses"][:10]:
        lines.append(
            f"- `{item.get('id')}` {item.get('question')}｜期望 "
            f"{', '.join(item.get('relevant') or [])}｜实际 top-k "
            f"{', '.join((item.get('retrieved') or [])[:5])}"
        )

    lines += [
        "",
        "## 三、结论",
        "",
        f"{advice['verdict']}",
        "",
        f"最好的候选配置救回了 {advice['rescued']} 条当前错例。",
        "",
        "## 四、下一步",
        "",
        "```bash",
        "# 切了配置之后必须过门禁",
        "python scripts/regression.py --only retrieval",
        "```",
        "",
        "> 提醒：纯向量、融合权重、RRF k、分块策略目前没有环境变量开关，"
        "表里这些配置就算指标更好，也只能先记进候选清单，改代码才能上线。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="用检索错例选 reranker/检索配置")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help="已有检索结果 JSON")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="本次对比结果写到哪")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--current", default=DEFAULT_CURRENT, help="当前生产口径的配置名")
    parser.add_argument("--refresh", action="store_true", help="先重跑一遍检索再对比")
    parser.add_argument(
        "--with-cross-encoder",
        action="store_true",
        help="把 BGE 精排拉进对比（需下载约 1.1GB 权重）",
    )
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    configs = list(DEFAULT_CONFIGS)
    if args.with_cross_encoder:
        configs += [item for item in CROSS_ENCODER_CONFIGS if item not in configs]

    source = args.json if args.refresh else args.baseline
    if args.refresh:
        print(f"重跑检索对比（{len(configs)} 个配置）…")
        if run_comparison(tuple(configs), args.json, args.report) != 0:
            return 1
    payload = load_json(source)
    if not payload:
        raise SystemExit(
            f"找不到检索结果：{source}\n"
            "  先跑：python scripts/eval_retrieval.py --configs "
            + " ".join(configs)
            + "（或给本脚本加 --refresh）"
        )

    rows = [config_metrics(payload, name) for name in configs]
    missing = [row["config"] for row in rows if not row["available"]]
    rows = [row for row in rows if row["available"]]
    if missing:
        print(f"（这些配置在 {source.name} 里没有数据，跳过：{'、'.join(missing)}）")

    current = config_metrics(payload, args.current)
    if not current["available"]:
        raise SystemExit(
            f"{source.name} 里没有当前口径 {args.current} 的数据，"
            "把 --current 换成已有的配置名，或加 --refresh"
        )

    print(
        f"\n当前口径 {current['config']}：recall@5 {current['recall@5']:.4f}、"
        f"recall@1 {current['recall@1']:.4f}、MRR {current['mrr']:.4f}、"
        f"未命中 {len(current['miss_ids'])} 条（{'、'.join(current['miss_ids']) or '无'}）"
    )

    advice = build_advice(current, rows, with_cross_encoder=args.with_cross_encoder)
    if not advice["ok"]:
        raise SystemExit(f"无法给出结论：{advice['reason']}")

    print("\n配置对比（按 recall@5/recall@1/MRR 排序）：")
    for row in advice["ranked"]:
        rescue = rescue_count(current["miss_ids"], row["miss_ids"])
        switchable = "✓ 可改 env" if row["config"] in RERANKER_BY_CONFIG else "✗ 要改代码"
        mark = "（当前）" if row["config"] == args.current else ""
        print(
            f"  {row['config'] + mark:<22} recall@5 {row['recall@5']:.4f}  "
            f"recall@1 {row['recall@1']:.4f}  MRR {row['mrr']:.4f}  "
            f"未命中 {len(row['miss_ids'])}  救回 {rescue}  {switchable}"
        )
    print(f"\n结论：{advice['verdict']}")

    if not args.no_report:
        write_report(args.report, advice, args.current)
        print(f"\n报告已写入：{args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
