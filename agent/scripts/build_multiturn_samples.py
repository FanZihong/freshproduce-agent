"""从 JDDC 会话里构造"带上下文的多轮意图样本"。

单句碎片（"那它呢""这个能退吗""大概一半吧"）单独看没有意图，
但放回会话上下文就是有效数据。这个脚本：

1. 按 session_id 还原完整会话（保留发言顺序与角色）；
2. 找出"依赖上文"的用户问句（含指代词 / 追问式开头 / 缺话题的短问句）；
3. 把前 1–2 轮的用户问题与客服回答拼成 context；
4. 输出 data/intent/raw/multiturn_v1.jsonl，可并进主语料一起预标注。

用法（在 agent/ 目录下执行）：
    python scripts/build_multiturn_samples.py
    python scripts/build_multiturn_samples.py --limit 800 --show 5
    python scripts/build_multiturn_samples.py --external-dir data/intent/external
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DATA_DIR  # noqa: E402
from agent.rag.rewrite import KNOWN_PRODUCTS  # noqa: E402
from scripts.build_intent_corpus import (  # noqa: E402
    EXTERNAL_DIR,
    _split_row,
    is_noise,
    normalize,
)

BRANCH_DIR = DATA_DIR / "intent" / "raw"
OUTPUT_PATH = BRANCH_DIR / "multiturn_v1.jsonl"

# 指代词：出现即判定依赖上文
ANAPHORA = (
    "这个", "那个", "它", "这款", "这件", "这种", "该商品", "此款", "上述",
    "刚才", "之前", "上面说的", "你刚说的",
)
# 追问式开头
FOLLOWUP_STARTS = ("那", "还有", "然后", "所以", "那么", "另外", "并且", "顺便")
# 疑问标记
QUESTION_MARKERS = (
    "怎么办", "多久", "多少", "能不能", "可以吗", "怎么弄", "如何", "为什么", "呢", "吗",
)
# 自带话题的名词：出现就说明这句能独立成立
TOPIC_NOUNS = (
    *KNOWN_PRODUCTS,
    "订单", "物流", "快递", "发票", "退款", "理赔", "赔付", "重量", "保存", "时效",
    "工单", "优惠券", "地址", "签收", "发货", "退货",
)


def is_context_dependent(text: str) -> bool:
    """判断这句话是否依赖上文才能确定意图。"""
    if any(word in text for word in ANAPHORA):
        return True
    if len(text) <= 15 and text.startswith(FOLLOWUP_STARTS):
        return True
    has_question = any(marker in text for marker in QUESTION_MARKERS)
    has_topic = any(noun in text for noun in TOPIC_NOUNS)
    return has_question and not has_topic and len(text) <= 12


def detect_product(history: list[str]) -> str:
    for text in reversed(history):
        for product in KNOWN_PRODUCTS:
            if product and product in text:
                return product
    return ""


def load_sessions(directory: Path) -> dict[str, list[tuple[str, str]]]:
    """按 session_id 还原会话：[(role, text), ...]，role 为 user / waiter。"""
    sessions: dict[str, list[tuple[str, str]]] = {}
    for path in sorted(directory.rglob("*.txt")):
        try:
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except UnicodeDecodeError:
            continue
        if not lines:
            continue

        header = _split_row(lines[0])
        if "content" not in header or "session_id" not in header:
            continue
        index = {column: header.index(column) for column in header}
        content_index = index["content"]
        session_index = index["session_id"]
        sender_index = index.get("waiter_send")
        separator = "\t" if "\t" in lines[0] else " "

        for line in lines[1:]:
            parts = _split_row(line)
            if len(parts) <= content_index:
                continue
            content = normalize(separator.join(parts[content_index:]))
            if is_noise(content):
                continue
            role = "waiter" if sender_index is not None and parts[sender_index] not in {"0", ""} else "user"
            sessions.setdefault(parts[session_index], []).append((role, content))
    return sessions


def build_samples(
    sessions: dict[str, list[tuple[str, str]]],
    *,
    limit: int = 800,
    seed: int = 42,
) -> list[dict]:
    samples: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for session_id, turns in sessions.items():
        user_history: list[str] = []
        for position, (role, text) in enumerate(turns):
            if role != "user":
                continue
            if user_history and is_context_dependent(text):
                previous_users = [t for r, t in turns[:position] if r == "user"]
                previous_waiters = [t for r, t in turns[:position] if r == "waiter"]
                context: dict = {"last_question": previous_users[-1]}
                if previous_waiters:
                    context["last_answer"] = previous_waiters[-1][:60]
                product = detect_product(previous_users)
                if product:
                    context["product"] = product

                key = (text, json.dumps(context, ensure_ascii=False, sort_keys=True))
                if key not in seen:
                    seen.add(key)
                    samples.append(
                        {
                            "text": text,
                            "context": context,
                            "hint_labels": [],
                            "source": "jddc_multiturn",
                            "session_id": session_id,
                        }
                    )
            user_history.append(text)

    if limit and len(samples) > limit:
        random.seed(seed)
        samples = random.sample(samples, limit)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="从会话构造多轮意图样本")
    parser.add_argument("--external-dir", type=Path, default=EXTERNAL_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=800, help="最多输出多少条（0 表示不限）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", type=int, default=3, help="抽样展示条数")
    args = parser.parse_args()

    sessions = load_sessions(args.external_dir)
    if not sessions:
        print(f"没有在 {args.external_dir} 找到会话文件（需要带表头的 chat.txt 风格 TSV）")
        return

    turn_count = sum(len(turns) for turns in sessions.values())
    print(f"会话数：{len(sessions)}，去噪后轮次：{turn_count}")

    samples = build_samples(sessions, limit=args.limit, seed=args.seed)
    if not samples:
        print("没有找到依赖上文的问句，检查 is_context_dependent 的判定规则。")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in samples) + "\n",
        encoding="utf-8",
    )

    with_product = sum(1 for row in samples if row["context"].get("product"))
    with_answer = sum(1 for row in samples if row["context"].get("last_answer"))
    print(f"多轮样本：{len(samples)} 条（带商品 {with_product}，带客服回答 {with_answer}）")
    print(f"已写入：{args.output}")

    if args.show:
        print(f"\n抽样 {args.show} 条：")
        for row in samples[: args.show]:
            print(f"  上一轮：{row['context'].get('last_question', '')}")
            print(f"  当前问：{row['text']}")
            if row["context"].get("product"):
                print(f"  商品  ：{row['context']['product']}")
            print()


if __name__ == "__main__":
    main()
