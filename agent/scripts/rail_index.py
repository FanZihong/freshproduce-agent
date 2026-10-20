"""打印护栏规则：检查 YAML 是否都能被加载、五类护栏各有多少条。

用法（在 agent/ 目录下执行）：
    python scripts/rail_index.py
    python scripts/rail_index.py --scope input
    python scripts/rail_index.py --json
    python scripts/rail_index.py --strict     # 有坏规则直接失败
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.rails import SCOPES, load_registry  # noqa: E402


def _match_text(item: dict) -> str:
    parts: list[str] = []
    if item["keywords"]:
        parts.append("词：" + "/".join(item["keywords"][:4]))
    if item["regex"]:
        parts.append(f"正则：{item['regex']}")
    if item.get("not_regex"):
        parts.append(f"非：{item['not_regex']}")
    if item["conditions"]:
        parts.append(f"条件：{len(item['conditions'])} 条")
    return "；".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="护栏规则目录")
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--scope", choices=SCOPES, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    registry = load_registry(args.dir, strict=args.strict)
    items = [item for item in registry.index() if not args.scope or item["scope"] == args.scope]

    if args.json:
        print(
            json.dumps(
                {
                    "directory": str(registry.directory),
                    "count": len(items),
                    "rules": items,
                    "errors": registry.errors,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if not registry.errors else 1

    print(f"护栏目录：{registry.directory}")
    print(f"规则：{len(items)}（非法：{len(registry.errors)}）")
    for scope in SCOPES:
        group = [item for item in items if item["scope"] == scope]
        if not group:
            continue
        print()
        print(f"[{scope}] {len(group)} 条")
        print(f"  {'id':<28}{'event':<14}{'action':<9}{'prio':<6}match")
        for item in group:
            print(
                f"  {item['id']:<28}{item['event']:<14}{item['action']:<9}"
                f"{item['priority']:<6}{_match_text(item)}"
            )

    if registry.errors:
        print()
        print("非法规则：")
        for error in registry.errors:
            print(f"  - {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
