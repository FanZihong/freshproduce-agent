"""对话 trace 落盘：自进化飞轮的数据入口。

每轮对话结束时把"这一轮到底发生了什么"写成一行 JSONL（默认 `data/trace/<日期>.jsonl`），
同时把会话与消息写进 SQLite 的 `sessions` / `messages` 表——这两张表建库时就有了，
在接 trace 之前一直没人写。

三条设计原则：
1. **绝不影响对话**：trace 是观测设施，不是业务链路。写失败只记警告，不抛异常；
2. **一轮一行**：按信号筛选、按会话聚合、直接 grep 都好用，不需要先解析整个文件；
3. **只存决策与结果**：意图/置信度/分支/决策/工具/是否转人工/回复。存的是"系统做了什么"，
   而不是"它为什么这么想"——挖 badcase 要的是可复现的事实，不是推理过程的叙事。

这份数据后面有三个消费方：`scripts/mine_traces.py`（挖候选 badcase）、
意图层的训练数据反哺、以及人工复盘时的原样回放。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from agent.db import DATA_DIR, DB_PATH, connect

# 格式版本：字段有增删时递增，读侧可以据此兼容老文件
TRACE_VERSION = 1

DEFAULT_TRACE_DIR = DATA_DIR / "trace"


def build_turn_record(
    state: dict,
    *,
    session_id: str,
    user_input: str,
    turn: int,
    backend: str = "",
    transport: str = "",
    tool_offset: int = 0,
) -> dict:
    """把一轮对话的最终状态压成一条可落盘的记录。

    `state` 就是 LangGraph 跑完这一轮后的状态字典；这里只挑与"这轮做得对不对"相关的字段，
    不做加工、不做推断——原始事实留给下游的挖掘脚本判断。

    `tool_offset` 是上一轮结束时 `tool_results` 的长度：各分支节点是**追加**工具记录的
    （`[*(state["tool_results"]), 新纪录]`），不切片的话第二轮会把第一轮的失败再报一遍，
    挖出来的 badcase 会重复。记录里的 `tools_total` 是累计长度，调用方拿它推进 offset。
    """
    all_tool_results = list(state.get("tool_results") or [])
    tool_results = all_tool_results[max(0, int(tool_offset)) :]
    labels = state.get("intent_labels") or []
    return {
        "trace_version": TRACE_VERSION,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id,
        "turn": int(turn),
        "user_input": user_input,
        "backend": backend,
        "transport": transport,
        # 意图与路由
        "intent": state.get("intent") or "",
        "confidence": _as_float(state.get("confidence")),
        "intent_labels": [str(label) for label in labels],
        "intent_secondary": state.get("intent_secondary") or "",
        "branches": [str(branch) for branch in (state.get("branches") or [])],
        "multi_strategy": state.get("multi_strategy") or "",
        "multi_branch_inputs": dict(state.get("multi_branch_inputs") or {}),
        # 决策层与域外
        "decision": state.get("decision") or "accept",
        "clarify_rounds": int(state.get("clarify_rounds") or 0),
        "out_of_domain": bool(state.get("out_of_domain")),
        "ood_reason": state.get("ood_reason") or "",
        # 流程与护栏
        "active_flow": state.get("active_flow") or "",
        "pending_slot": state.get("pending_slot") or "",
        "loop_count": int(state.get("loop_count") or 0),
        "slots": dict(state.get("slots") or {}),
        # 工具与结果
        "tools": [
            {
                "name": item.get("name") or "",
                "ok": bool(item.get("ok")),
                "action": item.get("action") or "",
                "attempts": int(item.get("attempts") or 1),
                "error": (item.get("error") or {}).get("kind") if item.get("error") else "",
            }
            for item in tool_results
        ],
        "tools_total": len(all_tool_results),
        "handoff": bool(state.get("handoff")),
        "handoff_reason": state.get("handoff_reason") or "",
        "handoff_summary": state.get("handoff_summary") or "",
        "citations": len(state.get("citations") or []),
        # 本轮注入了哪几条长期事实（只留 id/摘要/分数，trace 不宜太长）
        "memory_hits": [
            {
                "id": item.get("id"),
                "fact": str(item.get("fact") or "")[:60],
                "score": item.get("score"),
                "decay": item.get("decay"),
                "age_days": item.get("age_days"),
            }
            for item in (state.get("memory_hits") or [])[:3]
        ],
        # 本轮用了哪个技能：没命中就是 None，便于统计"技能覆盖率"
        "skill": (
            {
                "name": state.get("skill_name") or "",
                "version": state.get("skill_version") or "",
                "reason": state.get("skill_reason") or "",
                "score": _as_float(state.get("skill_score")),
                "tokens": int(state.get("skill_tokens") or 0),
                "truncated": bool(state.get("skill_truncated")),
                "prompt_chars": len(state.get("skill_prompt") or ""),
                "references": list(state.get("skill_references") or []),
            }
            if state.get("skill_name")
            else None
        ),
        # 本轮命中的护栏规则（输入/对话/执行/检索/输出）
        "rails": list(state.get("rails") or []),
        "reply": state.get("reply") or "",
    }


def _as_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def trace_path(directory: Path | str | None = None, when: str | None = None) -> Path:
    """按日期分文件的 trace 路径：`data/trace/2026-10-16.jsonl`。"""
    base = Path(directory) if directory else DEFAULT_TRACE_DIR
    stamp = (when or datetime.now().strftime("%Y-%m-%d"))[:10]
    return base / f"{stamp}.jsonl"


def append_turn(record: dict, *, directory: Path | str | None = None) -> Path | None:
    """把一条 trace 追加到当天文件；失败返回 None（不抛异常，绝不影响对话）。"""
    try:
        path = trace_path(directory, when=str(record.get("created_at") or "")[:10])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path
    except Exception as exc:  # noqa: BLE001 - 观测设施不该把主流程带崩
        print(f"[warn] trace 写入失败：{type(exc).__name__}: {exc}")
        return None


def save_turn_to_db(record: dict, *, db_path: Path | str | None = None) -> bool:
    """把这一轮写进 SQLite：一条 user 消息 + 一条 assistant 消息，并 upsert 会话行。

    与 JSONL 是同一份事实的两个视图：JSONL 供挖掘脚本批量读，数据库供前端按会话翻。
    """
    try:
        conn = connect(db_path or DB_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] trace 写库失败（连不上库）：{type(exc).__name__}: {exc}")
        return False

    session_id = str(record.get("session_id") or "")
    text = str(record.get("user_input") or "")
    now = str(record.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    slots = json.dumps(record.get("slots") or {}, ensure_ascii=False)
    tools = json.dumps(record.get("tools") or [], ensure_ascii=False)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, title, status, last_message, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_message = excluded.last_message,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    text[:40] or "（空）",
                    "handoff" if record.get("handoff") else "active",
                    (record.get("reply") or "")[:200],
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO messages
                    (message_id, session_id, role, content, intent, confidence, slots, tool_calls, trace_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    session_id,
                    "user",
                    text,
                    record.get("intent") or "",
                    _as_float(record.get("confidence")),
                    slots,
                    tools,
                    f"{session_id}#{record.get('turn')}",
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO messages
                    (message_id, session_id, role, content, intent, confidence, slots, tool_calls, trace_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    session_id,
                    "assistant",
                    record.get("reply") or "",
                    record.get("intent") or "",
                    _as_float(record.get("confidence")),
                    slots,
                    tools,
                    f"{session_id}#{record.get('turn')}",
                    now,
                ),
            )
        return True
    except sqlite3.Error as exc:
        print(f"[warn] trace 写库失败：{exc}")
        return False
    finally:
        conn.close()


def load_traces(path: Path | str | None = None) -> list[dict]:
    """读 trace：传文件读文件，传目录读目录下所有 `*.jsonl`（按名字排序），默认读当期目录。

    坏行跳过并计数，不让一行脏数据毁掉整个挖掘流程。
    """
    target = Path(path) if path else DEFAULT_TRACE_DIR
    files: list[Path]
    if target.is_dir():
        files = sorted(target.glob("*.jsonl"))
    elif target.exists():
        files = [target]
    else:
        return []

    records: list[dict] = []
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    records.sort(key=lambda item: (str(item.get("created_at") or ""), str(item.get("session_id") or ""), int(item.get("turn") or 0)))
    return records


def session_transcript(records: list[dict], session_id: str) -> list[dict]:
    """取出某个会话的逐轮记录，按轮次排序——用于人工复盘的完整回放。"""
    turns = [item for item in records if item.get("session_id") == session_id]
    return sorted(turns, key=lambda item: int(item.get("turn") or 0))


def sessions_of(records: list[dict]) -> list[str]:
    """trace 里出现过的会话 ID，按首次出现顺序去重。"""
    seen: list[str] = []
    for item in records:
        session_id = str(item.get("session_id") or "")
        if session_id and session_id not in seen:
            seen.append(session_id)
    return seen
