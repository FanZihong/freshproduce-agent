"""打印技能目录：验证"只加载元数据、不读正文"的注册表能不能跑起来。

用法（在 agent/ 目录下执行）：
    python scripts/skill_index.py
    python scripts/skill_index.py --json
    python scripts/skill_index.py --dir /tmp/skills --strict

有非法包时退出码为 1（可以直接挂到 CI 上）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.skills import load_registry  # noqa: E402


def _width(rows: list[list[str]], index: int) -> int:
    return max(len(row[index]) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="技能目录")
    parser.add_argument("--dir", type=Path, default=None, help="覆盖技能根目录")
    parser.add_argument("--json", action="store_true", help="输出 JSON（给接口/脚本用）")
    parser.add_argument("--strict", action="store_true", help="有非法包直接失败")
    args = parser.parse_args()

    registry = load_registry(args.dir, strict=args.strict)
    index = registry.index()

    if args.json:
        print(
            json.dumps(
                {
                    "directory": str(registry.directory),
                    "count": len(index),
                    "skills": index,
                    "errors": registry.errors,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if not registry.errors else 1

    print(f"技能目录：{registry.directory}")
    print(f"有效技能：{len(index)}　非法包：{len(registry.errors)}")
    print()
    if index:
        rows = [
            [
                item["name"],
                item["version"],
                str(item["chars"]),
                str(len(item["references"])),
                item["when_to_use"],
            ]
            for item in index
        ]
        widths = [_width(rows, column) for column in range(5)]
        header = ["name", "version", "chars", "refs", "when_to_use"]
        print("  ".join(header[i].ljust(widths[i]) for i in range(5)))
        print("  ".join("-" * widths[i] for i in range(5)))
        for row in rows:
            print("  ".join(row[i].ljust(widths[i]) for i in range(5)))
    else:
        print("（没有可用技能）")

    if registry.errors:
        print()
        print("非法包：")
        for error in registry.errors:
            print(f"  - {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
