"""Memory v2 报告：老化降权 + TTL 清理的可复现演示。

不调模型、不读正式库：事实库与索引都指向临时目录，嵌入用固定向量，
跑一次就能看到真实代码算出的分值与排序（适合写进 README / 简历）。

产出：
    eval/reports/memory_v2.md

用法（在 agent/ 目录下执行）：
    python scripts/memory_v2_report.py
    python scripts/memory_v2_report.py --report /tmp/memory_v2.md
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

# 先定好临时库与参数，再导入记忆模块（模块里读的是调用时的环境变量）
_TMP = Path(tempfile.mkdtemp(prefix="memory-v2-report-"))
os.environ["MEMORY_FACTS_DB"] = str(_TMP / "facts.db")
os.environ["MEMORY_INDEX_PATH"] = str(_TMP / "memory_index.json")
os.environ["MEMORY_FACT_TTL_DAYS"] = "30"
os.environ["MEMORY_DECAY"] = "on"
os.environ["MEMORY_DECAY_HALF_LIFE_DAYS"] = "14"
os.environ["MEMORY_DECAY_FLOOR"] = "0.5"

from agent.memory import facts as facts_store  # noqa: E402
from agent.memory.index import MemoryIndex  # noqa: E402
from agent.memory.search import search_facts  # noqa: E402

DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "memory_v2.md"

SESSION = "memory-v2-demo"
OLD_FACT = "草莓坏果一半已赔付"
NEW_FACT = "草莓坏果一半待赔付"
STALE_FACT = "用户问过发票抬头怎么开"
QUERY = "草莓坏果赔付"


class DemoEmbedder:
    """固定向量：OLD/NEW/查询同一方向，STALE 正交——让差异只来自老化与排序。"""

    name = "demo-fixed"

    def __init__(self) -> None:
        self.mapping = {
            OLD_FACT: [1, 0, 0, 0],
            NEW_FACT: [1, 0, 0, 0],
            QUERY: [1, 0, 0, 0],
            STALE_FACT: [0, 1, 0, 0],
        }

    def encode(self, texts, *, is_query: bool = False):
        return np.asarray(
            [self.mapping.get(str(text), [0, 0, 0, 1]) for text in texts],
            dtype="float32",
        )


def _row(results: list[dict], fact_id: str) -> dict | None:
    for position, item in enumerate(results, start=1):
        if item["id"] == fact_id:
            return {**item, "rank": position}
    return None


def _fmt_rank(row: dict | None) -> str:
    if row is None:
        return "未召回"
    return f"#{row['rank']}（{row['score']:.4f}）"


def build_report(path: Path) -> None:
    now = datetime.now()
    db = _TMP / "facts.db"
    index_path = _TMP / "memory_index.json"
    embedder = DemoEmbedder()

    old, _ = facts_store.insert_fact(SESSION, OLD_FACT, category="claim", db_path=db)
    new, _ = facts_store.insert_fact(SESSION, NEW_FACT, category="claim", db_path=db)
    stale, _ = facts_store.insert_fact(SESSION, STALE_FACT, category="policy", db_path=db)
    facts_store.touch_facts([old["id"]], now=now - timedelta(days=25), db_path=db)
    facts_store.touch_facts([new["id"]], now=now - timedelta(days=1), db_path=db)
    facts_store.touch_facts([stale["id"]], now=now - timedelta(days=40), db_path=db)

    # 先把三条都写进索引，模拟"过期前被检索过"，后面才能验证 purge 会清向量
    index = MemoryIndex(index_path)
    for item, vector in (
        (old, [1, 0, 0, 0]),
        (new, [1, 0, 0, 0]),
        (stale, [0, 1, 0, 0]),
    ):
        index.upsert(
            item["id"],
            fact=item["fact"],
            session_id=SESSION,
            entities=[],
            vector=np.asarray(vector, dtype="float32"),
        )
    index.save()
    indexed_before = len(index.records)

    common = dict(
        db_path=db, index_path=index_path, embedder=embedder, top_k=5, now=now, touch=False
    )
    without = search_facts(QUERY, SESSION, decay=False, **common)
    with_decay = search_facts(QUERY, SESSION, decay=True, **common)

    expired_ids = facts_store.list_expired_ids(now=now, db_path=db)
    purged = facts_store.purge_expired_facts(now=now, db_path=db, index_path=index_path)
    index_after = len(MemoryIndex(index_path).load().records)
    after_purge = search_facts(QUERY, SESSION, decay=True, **common)

    ttl = facts_store.resolve_ttl_days()
    half_life = float(os.environ["MEMORY_DECAY_HALF_LIFE_DAYS"])
    floor = float(os.environ["MEMORY_DECAY_FLOOR"])

    curve = []
    for days in (0, 7, 14, 30, 60, 180):
        item = {"last_accessed_at": (now - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")}
        curve.append((days, facts_store.decay_factor(item, now=now)))

    old_off = _row(without, old["id"])
    new_off = _row(without, new["id"])
    old_on = _row(with_decay, old["id"])
    new_on = _row(with_decay, new["id"])
    old_score_before = (old_off or {}).get("score", 0.0)
    old_score_after = (old_on or {}).get("score", 0.0)
    if old_off and old_off["rank"] == 1 and new_on and (
        old_on is None or new_on["rank"] < old_on["rank"]
    ):
        verdict = (
            "降权前旧事实排第一，降权后新事实反超——"
            f"旧事实分数从 {old_score_before:.4f} 降到 {old_score_after:.4f}，"
            "但没有被清零：高度相关的旧记忆仍有机会靠相关性赢回来。"
        )
    elif new_on and (old_on is None or new_on["rank"] < old_on["rank"]):
        verdict = (
            "两者相关性几乎打平，降权后新事实排在第一——"
            f"旧事实分数从 {old_score_before:.4f} 降到 {old_score_after:.4f}，"
            "衰减拉开了新旧差距，但没有把旧事实清零。"
        )
    else:
        verdict = (
            "降权后旧事实仍然第一：相关性优势足够大，衰减只是缩小差距——"
            "这正是 floor 的作用：不误杀高相关的旧记忆。"
        )

    lines = [
        "# Memory v2：事实老化与过期清理",
        "",
        f"- 生成时间：{now:%Y-%m-%d %H:%M:%S}",
        f"- 演示会话：`{SESSION}`（临时库，不碰正式数据）",
        f"- 参数：TTL **{ttl:g} 天**、半衰期 **{half_life:g} 天**、最低权重 **{floor:g}**",
        "",
        "## 一、老化系数曲线",
        "",
        "`decay = floor + (1 - floor) × 0.5^(age / half_life)`",
        "",
        "| 距上次使用 | 老化系数 |",
        "| --- | --- |",
    ]
    lines += [f"| {days} 天 | {factor:.4f} |" for days, factor in curve]

    lines += [
        "",
        "## 二、同样的相关性，新旧事实的排序变化",
        "",
        "两条事实的可召回性被打平（同一向量、同样命中查询词），"
        "唯一差别是**上次被用到的时间**：",
        "",
        "| 事实 | 距上次使用 | 老化系数 | 不降权排名 | 降权后排名 |",
        "| --- | --- | --- | --- | --- |",
        f"| 旧：{OLD_FACT} | 25 天 | {(old_on or {}).get('decay', 0):.4f} | "
        f"{_fmt_rank(old_off)} | {_fmt_rank(old_on)} |",
        f"| 新：{NEW_FACT} | 1 天 | {(new_on or {}).get('decay', 0):.4f} | "
        f"{_fmt_rank(new_off)} | {_fmt_rank(new_on)} |",
        "",
        f"结论：{verdict}",
        "",
        "## 三、TTL 清理",
        "",
        f"- 过期阈值：{ttl:g} 天（`MEMORY_FACT_TTL_DAYS`）",
        f"- 过期事实：{len(expired_ids)} 条（{STALE_FACT}，40 天未用）",
        f"- 清理结果：删除 {len(purged)} 条；索引 {indexed_before} → {index_after} 条",
        f"- 清理后再召回：{_fmt_rank(_row(after_purge, stale['id']))}",
        "",
        "读时过滤保证「即使没跑到清理，过期事实也不会被召回」；"
        "`purge_expired_facts` 负责真正删库，并同步删向量，避免幽灵向量。",
        "",
        "## 四、接口与运维入口",
        "",
        "```bash",
        "# 只读查看某个会话的记忆（含 ageDays / decay / expiresAt）",
        "curl 'http://localhost:8000/api/memory?sessionId=s-123&includeExpired=false'",
        "",
        "# 每天清理一次（dry-run 先看会删什么）",
        "python scripts/memory_maintenance.py --dry-run",
        "python scripts/memory_maintenance.py",
        "```",
        "",
        "## 五、决策",
        "",
        "- 老化只改**排序**，不改 `vector_score`：注入门槛看相关性，"
        "避免「新但无关」的记忆挤掉「旧但命中」的记忆；",
        "- 半衰期 14 天、floor 0.5：指数部分每两周减半（叠加 floor 后 1.0 → 0.75），"
        "最低保留一半，防止长期未用的事实被永久埋没；",
        "- 事实 TTL 30 天（摘要 7 天）：售后跟进常跨月，事实比摘要更长寿；被召回一次就会续期；",
        "- 默认 `MEMORY_RECALL=off`：机制与门禁都就绪，等真实 trace 消融再决定是否打开。",
        "",
        "> 完整决策记录见 `eval/reports/memory_decision_log.md`。",
        "",
        "## 六、怎么复现",
        "",
        "```bash",
        "python scripts/memory_v2_report.py",
        "pytest -q tests/test_memory_decay.py",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 Memory v2 报告")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    build_report(args.report)
    print(f"报告已写入：{args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
