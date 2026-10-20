"""同会话并发压测：N 个请求同时打到同一个 session，验证不会丢轮次。

为什么需要它：`session_lock` 的单测证明"锁能串行"，但线上真正的问题是
"并发时会不会丢状态/乱轮次"。这个脚本用真实图跑一遍，直接数最终状态里的用户消息：
- 加锁（默认）：每个请求都应该出现在最终状态里，一条不少；
- `--no-lock`：用来演示不加锁时的表现（可能丢更新，也可能侥幸没丢，取决于时序）。

用法（在 agent/ 目录下执行）：
    python scripts/stress_session.py                 # 4 并发 × 4 轮，加锁
    python scripts/stress_session.py --workers 8 --turns 4
    python scripts/stress_session.py --no-lock       # 对照：不加锁

退出码：0 = 加锁版本一条没丢；1 = 有丢失或异常。
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.graph.workflow import get_graph  # noqa: E402
from agent.memory import short_term  # noqa: E402
from agent.session_lock import SessionBusy, session_lock  # noqa: E402

TURNS = ("你好", "在吗", "我想问下售后", "我的订单到哪了", "稍等", "还在吗", "麻烦了", "谢谢")


def main() -> int:
    parser = argparse.ArgumentParser(description="同会话并发压测")
    parser.add_argument("--workers", type=int, default=4, help="并发请求数")
    parser.add_argument("--turns", type=int, default=4, help="发几条消息（默认 4）")
    parser.add_argument("--session", default=None, help="会话 ID（默认按时间生成）")
    parser.add_argument("--no-lock", action="store_true", help="对照：不加锁")
    args = parser.parse_args()

    session_id = args.session or f"stress-{datetime.now():%H%M%S}"
    config = {"configurable": {"thread_id": session_id}}

    # 压测关注"轮次是否丢"，把压缩阈值调高，避免旧消息被折叠进 summary 影响计数
    short_term.COMPRESS_THRESHOLD = 10_000
    graph = get_graph(persistent=False)
    texts = [f"并发消息 {index}（{TURNS[index % len(TURNS)]}）" for index in range(args.turns)]

    def run_one(text: str) -> dict:
        started = time.perf_counter()
        try:
            if args.no_lock:
                graph.invoke({"user_input": text, "session_id": session_id}, config=config)
            else:
                with session_lock(session_id, timeout=60):
                    graph.invoke({"user_input": text, "session_id": session_id}, config=config)
            return {"text": text, "ok": True, "ms": (time.perf_counter() - started) * 1000}
        except SessionBusy as exc:
            return {"text": text, "ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - 压测要把异常收集下来而不是中断
            return {"text": text, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    print(f"会话 {session_id}：{args.workers} 并发 × {args.turns} 条"
          + ("（不加锁，对照组）" if args.no_lock else "（加锁）"))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_one, texts))

    failures = [item for item in results if not item["ok"]]
    state = graph.get_state(config).values
    users = [
        str(getattr(message, "content", ""))
        for message in (state.get("messages") or [])
        if getattr(message, "type", "") == "human"
    ]
    missing = [text for text in texts if text not in users]
    duplicate = len(users) - len(set(users))

    print(f"  请求成功 {len(results) - len(failures)}/{len(results)}")
    if failures:
        for item in failures[:3]:
            print(f"  ✗ {item['text']}：{item.get('error')}")
    print(f"  最终状态里的用户消息：{len(users)} 条（期望 {len(texts)}），丢失 {len(missing)} 条，重复 {duplicate} 条")
    latencies = sorted(item["ms"] for item in results if item["ok"])
    if latencies:
        print(
            f"  单轮耗时：中位 {latencies[len(latencies) // 2]:.0f} ms，"
            f"最大 {latencies[-1]:.0f} ms（串行时会排队）"
        )

    if args.no_lock:
        print("\n对照组（不加锁）只作演示：丢了说明竞态真实存在，不丢也不代表安全。")
        return 0
    if failures or missing or duplicate:
        print("\n结论：有异常或丢轮次，检查 session_lock 是否被绕过。")
        return 1
    print("\n结论：加锁后所有请求都进了最终状态，轮次没有丢。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
