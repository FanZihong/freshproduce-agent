"""校验意图体系，并用规则版分类器做一次边界体检。

用法（在 agent/ 目录下执行）：
    python scripts/check_taxonomy.py
    python scripts/check_taxonomy.py --show-mismatch 20
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.intent.classifier import classify  # noqa: E402
from agent.intent.taxonomy import load_taxonomy, validate  # noqa: E402


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def main() -> None:
    parser = argparse.ArgumentParser(description="意图体系校验与规则版体检")
    parser.add_argument("--show-mismatch", type=int, default=10, help="最多展示多少条不一致样例")
    args = parser.parse_args()

    taxonomy = load_taxonomy()
    problems = validate(taxonomy)

    print(f"体系版本：v{taxonomy.version}（{taxonomy.updated}）")
    print(f"类别数：{len(taxonomy.labels)}；域外配置：{taxonomy.out_of_domain.get('action')}")
    print(f"校验结果：{'通过' if not problems else '发现问题'}")
    for problem in problems:
        print(f"  - {problem}")

    print(f"\n{'类别':<16}{'路由':<10}{'正例':<6}{'反例':<6}定义")
    print("-" * 96)
    for spec in taxonomy.labels:
        definition = spec.definition if len(spec.definition) <= 34 else spec.definition[:34] + "..."
        print(
            f"{_pad(f'{spec.key}/{spec.name}', 16)}{_pad(spec.route, 10)}"
            f"{len(spec.positives):<6}{len(spec.negatives):<6}{definition}"
        )

    # 规则版体检：正例能否落到正确分支，反例会不会误落到本类分支
    mismatches: list[tuple[str, str, str, str]] = []
    total_positive = 0
    hit_positive = 0
    negative_leak = 0
    total_negative = 0

    for spec in taxonomy.labels:
        class_hit = 0
        for text in spec.positives:
            total_positive += 1
            result = classify(text)
            predicted_route = taxonomy.route_of(result.label)
            if predicted_route == spec.route:
                hit_positive += 1
                class_hit += 1
            else:
                mismatches.append(
                    (spec.key, text, result.label, f"期望分支 {spec.route}，实际 {predicted_route}")
                )
        for text in spec.negatives:
            total_negative += 1
            result = classify(text)
            if taxonomy.route_of(result.label) == spec.route:
                negative_leak += 1
                mismatches.append(
                    (spec.key, text, result.label, "反例被分到本类分支（边界风险）")
                )
        rate = class_hit / len(spec.positives) if spec.positives else 0.0
        print(f"  {_pad(spec.name, 12)}正例分支命中率 {rate:.0%}（{class_hit}/{len(spec.positives)}）")

    print(
        f"\n规则版整体：正例分支准确率 {hit_positive}/{total_positive}"
        f"（{hit_positive / total_positive:.0%}），反例误落本类 {negative_leak}/{total_negative}"
    )
    if mismatches:
        print(f"\n不一致样例（最多 {args.show_mismatch} 条）：")
        for class_key, text, predicted, reason in mismatches[: args.show_mismatch]:
            print(f"  [{class_key}] {text}")
            print(f"      规则版判为 {predicted} —— {reason}")

    print(
        "\n说明：规则版只是零成本兜底，正例命中率不高是预期的；"
        "这一步的作用是暴露边界与规则覆盖的缺口，为 few-shot 基线和微调提供对照。"
    )


if __name__ == "__main__":
    main()
