"""抽样体检外部语料（如 JDDC）：看它能覆盖我们 10 类意图中的哪些、各能提供多少。

用法（在 agent/ 目录下执行）：
    python scripts/sample_jddc.py --limit 200 --label-mode dry-run   # 不花钱，先看流程
    python scripts/sample_jddc.py --limit 200 --label-mode llm       # 真实预标注，得到可用配额
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.config import get_env  # noqa: E402
from agent.db import DATA_DIR  # noqa: E402
from agent.intent.taxonomy import load_taxonomy  # noqa: E402
from scripts.build_intent_corpus import EXTERNAL_DIR, load_external  # noqa: E402
from scripts.label_intent import (  # noqa: E402
    _call_llm,
    build_system_prompt,
    dry_run_label,
    parse_label_response,
)

REVIEW_DIR = DATA_DIR / "intent" / "review"
DEFAULT_SAMPLE_OUT = REVIEW_DIR / "external_coverage_sample.jsonl"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "external_coverage.md"
TARGET_PER_CLASS = 400  # 每类希望拿到的训练样本量（用于估算补充量）


def main() -> None:
    parser = argparse.ArgumentParser(description="外部语料覆盖度抽样")
    parser.add_argument("--external-dir", type=Path, default=EXTERNAL_DIR)
    parser.add_argument("--limit", type=int, default=200, help="抽样条数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    parser.add_argument("--label-mode", choices=["dry-run", "llm"], default="dry-run")
    parser.add_argument("--concurrency", type=int, default=int(get_env("LABEL_CONCURRENCY", "6")))
    parser.add_argument("--model", default=get_env("LABEL_MODEL", "deepseek-chat"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sample-out", type=Path, default=DEFAULT_SAMPLE_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    rows = load_external(args.external_dir)
    if not rows:
        print(f"外部语料为空：{args.external_dir}")
        print("请先把 JDDC 等语料文件放进该目录，再运行 build_intent_corpus.py --inspect-external 检查解析。")
        return

    random.seed(args.seed)
    sample = random.sample(rows, min(args.limit, len(rows)))

    taxonomy = load_taxonomy()
    valid_labels = set(taxonomy.keys)
    system_prompt = build_system_prompt()

    print(f"外部语料共 {len(rows)} 条，抽样 {len(sample)} 条（seed={args.seed}）")
    print(f"标注模式：{'dry-run（规则版兜底，仅供参考）' if args.label_mode == 'dry-run' else f'LLM {args.model}'}\n")

    def label_row(row: dict) -> list[str]:
        if args.label_mode == "dry-run":
            labels, _, _ = dry_run_label(row, valid_labels)
            return labels
        raw = _call_llm(system_prompt, row, args.model, args.timeout)
        labels, _, _ = parse_label_response(raw, valid_labels)
        return labels

    if args.label_mode == "llm":
        api_key = get_env("LLM_API_KEY", "")
        if not api_key or "替换" in api_key:
            print("[错误] agent/.env 里的 LLM_API_KEY 还没配置（当前是占位符）。")
            print("       要么填真实 key，要么先用 --label-mode dry-run 验证流程。")
            return

    labeled: list[dict] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {pool.submit(label_row, row): row for row in sample}
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                labels = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            labeled.append({**row, "labels": labels})
            if index % 50 == 0:
                print(f"  已处理 {index}/{len(sample)}")

    if not labeled:
        print(f"\n[错误] {len(sample)} 条全部标注失败，无法统计覆盖度。")
        print("前几条失败原因：")
        for message in failures[:3]:
            print(f"  - {message}")
        print("\n常见原因：key 无效 / 余额不足 / 模型名不存在 / 网络不通。")
        return

    primary_counts = Counter(row["labels"][0] for row in labeled if row["labels"])
    domain_out = primary_counts.get("chitchat", 0)
    usable = len(labeled) - domain_out
    sample_ratio = len(labeled) / len(rows) if rows else 0

    print(f"\n可识别为业务意图：{usable}/{len(labeled)}（{usable / len(labeled):.1%}）")
    print(f"域外/闲聊：{domain_out} 条（{domain_out / len(labeled):.1%}）")
    if failures:
        print(f"标注失败：{len(failures)} 条（前 3 条原因）")
        for message in failures[:3]:
            print(f"  - {message}")

    print(f"\n{'类别':<18}{'抽样命中':<10}{'按比例推算可得':<16}建议补充")
    suggestions: list[tuple[str, int, int, int]] = []
    for spec in taxonomy.labels:
        hit = primary_counts.get(spec.key, 0)
        estimated = int(hit / sample_ratio) if sample_ratio else 0
        suggest = max(0, TARGET_PER_CLASS - estimated)
        suggestions.append((spec.key, hit, estimated, suggest))
        print(f"{spec.key:<18}{hit:<10}{estimated:<16}{suggest}")

    args.sample_out.parent.mkdir(parents=True, exist_ok=True)
    args.sample_out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in labeled) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# 外部语料覆盖度体检",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 外部语料总量：{len(rows)} 条；抽样：{len(labeled)} 条（seed={args.seed}）",
        f"- 标注模式：{args.label_mode}",
        f"- 可识别为业务意图：{usable}/{len(labeled)}（{usable / len(labeled):.1%}）",
        f"- 域外/闲聊：{domain_out} 条（{domain_out / len(labeled):.1%}）",
        "",
        "| 类别 | 抽样命中 | 按比例推算可得 | 建议补充到 400 条 |",
        "| --- | --- | --- | --- |",
    ]
    for key, hit, estimated, suggest in suggestions:
        lines.append(f"| {key} | {hit} | {estimated} | {suggest} |")
    lines += [
        "",
        "> 说明：dry-run 用规则版兜底，数字只用于验证流程；"
        "要拿到可信配额，请用 `--label-mode llm` 跑一次。",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n抽样明细：{args.sample_out}")
    print(f"报告：{args.report}")


if __name__ == "__main__":
    main()
