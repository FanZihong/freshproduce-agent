"""事实记忆写入演示：召回旧事实 → 只抽增量 → 两层去重 → 三处落盘。

不需要起图、不需要真模型（`--fake-llm` 默认开）：用脚本化的"模型输出"把三条路径各演示一遍。

    python scripts/memory_demo.py            # 演示：新增 -> 重复跳过 -> 换说法更新
    python scripts/memory_demo.py --real     # 用真实 LLM（需要 LLM_API_KEY）
    python scripts/memory_demo.py --session demo-1 --db data/memory_demo.db

演示用的会话和库都是临时的（默认写到 /tmp），不会污染正式数据。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

import numpy as np  # noqa: E402

from agent.memory import facts as facts_store  # noqa: E402
from agent.memory.extractor import set_extractor_completer  # noqa: E402
from agent.memory.writer import remember  # noqa: E402

OLD_FACT = "用户草莓订单 2026093000123 坏果一半，按 50% 赔付 12.90 元"
PARAPHRASE = "草莓订单 2026093000123 坏果约一半，已赔 12.9 元"
NEW_FACT = "用户要求开具电子发票，抬头为公司"
NOTE = "（演示用假嵌入：把这三句话映射到固定向量，真实运行用 BGE）"


class DemoEmbedder:
    name = "demo"
    dim = 4
    mapping = {
        OLD_FACT: [1, 0, 0, 0],
        PARAPHRASE: [1, 0, 0, 0],
        NEW_FACT: [0, 1, 0, 0],
    }

    def encode(self, texts, *, is_query: bool = False):
        return np.asarray(
            [self.mapping.get(str(text), [0, 0, 0, 1]) for text in texts], dtype="float32"
        )


def scripted_llm(call: dict):
    """按顺序模拟模型的四种输出：新增 → 重复（没听话）→ 换说法 → 全新事实。"""

    def complete(messages):
        call["n"] += 1
        if call["n"] == 1:
            payload = [
                {"fact": OLD_FACT, "category": "claim", "entities": ["2026093000123"], "action": "add"}
            ]
        elif call["n"] == 2:
            # 模型没按要求只抽增量，把已经记住的又抽了一遍 -> 靠哈希去重挡住
            payload = [{"fact": OLD_FACT, "category": "claim", "action": "add"}]
        elif call["n"] == 3:
            payload = [{"fact": PARAPHRASE, "category": "claim", "action": "add"}]
        else:
            payload = [{"fact": NEW_FACT, "category": "policy", "entities": ["发票"], "action": "add"}]
        return json.dumps(payload, ensure_ascii=False), {"prompt_tokens": 220, "completion_tokens": 50}

    return complete


def show_facts(session_id: str, db: Path) -> None:
    rows = facts_store.list_facts(session_id, db_path=db)
    print(f"  库里现在 {len(rows)} 条事实：")
    for row in rows:
        print(f"    [{row['category']}] {row['fact']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="事实记忆写入演示")
    parser.add_argument("--session", default="memory-demo")
    parser.add_argument("--db", type=Path, default=None, help="事实库路径（默认临时文件）")
    parser.add_argument("--index", type=Path, default=None, help="向量索引路径（默认临时文件）")
    parser.add_argument("--real", action="store_true", help="用真实 LLM 抽取（需要 key）")
    parser.add_argument("--keep", action="store_true", help="保留临时库，方便自己翻")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="memory-demo-"))
    db = args.db or tmp / "facts.db"
    index_path = args.index or tmp / "memory_index.json"
    embedder = DemoEmbedder()

    call = {"n": 0}
    if not args.real:
        set_extractor_completer(scripted_llm(call))
    print(f"模式：{'真实 LLM' if args.real else '脚本化假模型'}" + ("" if args.real else NOTE))
    print(f"事实库：{db}\n向量索引：{index_path}\n")

    turns_a = [{"user": "草莓坏了一半，怎么赔", "reply": "请提供订单号"},
               {"user": "订单 2026093000123", "reply": "按 50% 赔付 12.90 元"}]
    print("① 第一次写入（旧记忆为空）")
    stats = remember(args.session, turns_a, mode="llm", db_path=db, index_path=index_path, embedder=embedder)
    print(f"  新增 {stats.added}、更新 {stats.updated}、跳过 {stats.skipped}；"
          f"调用 {stats.calls} 次、成本 ¥{stats.cost:.6f}")
    show_facts(args.session, db)

    print("\n② 同一段对话再来一次（模型又抽了一遍旧事实 -> 哈希去重应该跳过）")
    stats = remember(args.session, turns_a, mode="llm", db_path=db, index_path=index_path, embedder=embedder)
    print(f"  新增 {stats.added}、更新 {stats.updated}、跳过 {stats.skipped}")
    show_facts(args.session, db)

    print("\n③ 换说法的补充（相似度去重应该走更新）")
    turns_b = [{"user": "那个赔付到底多少来着", "reply": "12.9 元，已经到账了"}]
    stats = remember(args.session, turns_b, mode="llm", db_path=db, index_path=index_path, embedder=embedder)
    print(f"  新增 {stats.added}、更新 {stats.updated}、跳过 {stats.skipped}")
    show_facts(args.session, db)

    print("\n④ 真正的新信息（发票需求 -> 应该新增）")
    turns_c = [{"user": "顺便帮我开个发票", "reply": "好的，请提供抬头"}]
    stats = remember(args.session, turns_c, mode="llm", db_path=db, index_path=index_path, embedder=embedder)
    print(f"  新增 {stats.added}、更新 {stats.updated}、跳过 {stats.skipped}")
    show_facts(args.session, db)

    print("\n⑤ 混合召回：语义查询 vs 字面查询")
    from agent.memory.search import search_facts

    # 让"赔付"这句走语义路（和事实同向量），"发票"走 BM25 字面路
    embedder.mapping["那笔赔付后来怎么样了"] = [1, 0, 0, 0]
    for query in ("那笔赔付后来怎么样了", "发票抬头怎么填"):
        rows = search_facts(
            query,
            args.session,
            db_path=db,
            index_path=index_path,
            embedder=embedder,
            top_k=2,
        )
        print(f"  查询「{query}」")
        for row in rows:
            print(
                f"    {row['fact'][:28]}｜score={row['score']:.4f} "
                f"向量={row['vector_score']:.3f} BM25={row['bm25_score']:.2f} "
                f"实体命中={row['entity_hits']}"
            )

    if not args.real:
        set_extractor_completer(None)
    if not args.keep:
        print(f"\n临时文件：{tmp}（--keep 可保留）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
