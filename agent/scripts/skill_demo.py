"""技能路由 + 注入预览：不跑图、不调模型，一眼看清"选了谁、为什么、注入了什么"。

用法（在 agent/ 目录下执行）：
    python scripts/skill_demo.py "车厘子怎么保存"
    python scripts/skill_demo.py "退款一般几天到账" --max-tokens 120
    python scripts/skill_demo.py "你好" --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.skills.loader import default_max_tokens, load_skill  # noqa: E402
from agent.skills.prompt import build_skill_prompt  # noqa: E402
from agent.skills.references import (  # noqa: E402
    build_reference_prompt,
    load_references,
    select_references,
    wants_details,
)
from agent.skills.router import select_skill  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="技能路由与注入预览")
    parser.add_argument("query", nargs="?", default="车厘子怎么保存")
    parser.add_argument("--max-tokens", type=int, default=None, help="覆盖 SKILL_MAX_TOKENS")
    parser.add_argument("--preview", type=int, default=800, help="预览字符数")
    parser.add_argument("--no-details", action="store_true", help="即使追问细节也不加载 references")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    match = select_skill(args.query)
    if match is None:
        if args.json:
            print(json.dumps({"query": args.query, "skill": None}, ensure_ascii=False, indent=2))
        else:
            print(f"查询：{args.query}")
            print("未命中技能：分数不够或与描述重合度太低（宁可不注入）")
        return 0

    loaded = load_skill(match.skill, max_tokens=args.max_tokens)
    prompt = build_skill_prompt(loaded)
    budget = args.max_tokens or default_max_tokens()
    references = []
    if loaded.skill.references and not args.no_details and wants_details(args.query):
        names = select_references(loaded.skill, args.query)
        references = load_references(loaded.skill, names)
        block = build_reference_prompt(loaded.skill, references)
        if block:
            prompt = f"{prompt}\n\n{block}"

    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "skill": {
                        "name": loaded.skill.name,
                        "version": loaded.skill.version,
                        "reason": match.reason,
                        "score": match.score,
                        "tokens": loaded.tokens,
                        "truncated": loaded.truncated,
                        "body_chars": len(loaded.body),
                        "references": [doc.to_dict() for doc in references],
                    },
                    "prompt": prompt,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(f"查询：{args.query}")
    print(f"选中技能：{loaded.skill.name} v{loaded.skill.version}")
    print(f"选择理由：{match.reason}（匹配分 {match.score}）")
    print(
        f"注入预算：{budget} tokens；实际 {loaded.tokens} tokens；"
        f"截断：{'是' if loaded.truncated else '否'}"
    )
    if references:
        print("加载细则：" + "、".join(
            f"{doc.name}（{doc.tokens} tokens）" for doc in references
        ))
    else:
        print("加载细则：无（没有追问细节，或该技能没有 references）")
    print("---- 注入预览 ----")
    print(prompt[: args.preview] + ("…" if len(prompt) > args.preview else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
