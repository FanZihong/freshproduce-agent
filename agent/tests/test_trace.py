"""trace 落盘模块的测试：记录构造、JSONL 往返、会话回放、写库。

写库用临时 sqlite 文件，不碰项目里的正式库。
"""

from __future__ import annotations

import json

from agent.db import DB_PATH, connect, init_db
from agent.trace import (
    TRACE_VERSION,
    append_turn,
    build_turn_record,
    load_traces,
    save_turn_to_db,
    session_transcript,
    sessions_of,
    trace_path,
)


def make_state(**overrides) -> dict:
    state = {
        "intent": "bad_fruit_claim",
        "confidence": 0.93,
        "intent_labels": ("bad_fruit_claim",),
        "intent_secondary": "",
        "branches": ["claim"],
        "multi_strategy": "priority",
        "multi_branch_inputs": {},
        "decision": "accept",
        "clarify_rounds": 0,
        "out_of_domain": False,
        "ood_reason": "",
        "active_flow": "bad_fruit_claim",
        "pending_slot": "damage_ratio",
        "loop_count": 1,
        "slots": {"order_id": "2026093000123"},
        "tool_results": [
            {"name": "query_order", "ok": True, "action": "", "attempts": 1, "error": None}
        ],
        "handoff": False,
        "handoff_reason": "",
        "handoff_summary": "",
        "citations": [],
        "reply": "请告诉我坏果的大致比例。",
    }
    state.update(overrides)
    return state


# ---------------- 记录构造 ----------------


def test_build_turn_record_keeps_decision_facts():
    record = build_turn_record(
        make_state(),
        session_id="cli-101530",
        user_input="草莓坏了一半",
        turn=2,
        backend="local-lora",
        transport="direct",
    )
    assert record["trace_version"] == TRACE_VERSION
    assert record["session_id"] == "cli-101530" and record["turn"] == 2
    assert record["user_input"] == "草莓坏了一半"
    assert record["intent"] == "bad_fruit_claim" and record["confidence"] == 0.93
    assert record["intent_labels"] == ["bad_fruit_claim"]
    assert record["branches"] == ["claim"] and record["pending_slot"] == "damage_ratio"
    assert record["backend"] == "local-lora" and record["transport"] == "direct"
    assert record["tools"][0]["name"] == "query_order" and record["tools"][0]["ok"] is True
    assert record["handoff"] is False and record["reply"].startswith("请告诉我")


def test_build_turn_record_extracts_tool_error_kind():
    state = make_state(
        tool_results=[
            {
                "name": "query_order",
                "ok": False,
                "action": "ask_again",
                "attempts": 2,
                "error": {"kind": "not_found", "message": "查无此单"},
            }
        ],
        confidence=None,
    )
    record = build_turn_record(state, session_id="s", user_input="查一下", turn=1)
    assert record["tools"][0] == {
        "name": "query_order",
        "ok": False,
        "action": "ask_again",
        "attempts": 2,
        "error": "not_found",
    }
    assert record["confidence"] == 0.0        # 拿不到置信度时给 0，不写成 null


def test_build_turn_record_slices_tools_by_offset():
    """工具记录在状态里是累加的：第二轮只应该记第二轮新增的那条。"""
    state = make_state(
        tool_results=[
            {"name": "query_order", "ok": True, "action": "", "attempts": 1, "error": None},
            {"name": "create_ticket", "ok": False, "action": "handoff", "attempts": 1,
             "error": {"kind": "policy", "message": "超过赔付上限"}},
        ]
    )
    record = build_turn_record(
        state, session_id="s", user_input="就这些", turn=2, tool_offset=1
    )
    assert [item["name"] for item in record["tools"]] == ["create_ticket"]
    assert record["tools"][0]["ok"] is False
    assert record["tools_total"] == 2          # 累计长度交给调用方推进 offset

    first = build_turn_record(state, session_id="s", user_input="查订单", turn=1, tool_offset=0)
    assert [item["name"] for item in first["tools"]] == ["query_order", "create_ticket"]


# ---------------- JSONL 往返 ----------------


def test_append_and_load_roundtrip(tmp_path):
    first = build_turn_record(make_state(), session_id="s-1", user_input="第一句", turn=1)
    second = build_turn_record(make_state(), session_id="s-1", user_input="第二句", turn=2)
    second["created_at"] = "2026-10-17 09:00:00"

    path = append_turn(first, directory=tmp_path)
    append_turn(second, directory=tmp_path)
    assert path is not None and path.name.endswith(".jsonl")

    records = load_traces(tmp_path)
    assert len(records) == 2
    # 按时间排序：第二句的 created_at 更晚
    assert [item["user_input"] for item in records] == ["第一句", "第二句"]


def test_load_traces_skips_broken_lines(tmp_path):
    path = tmp_path / "2026-10-16.jsonl"
    good = build_turn_record(make_state(), session_id="s", user_input="正常", turn=1)
    path.write_text(
        "{不是 json}\n\n" + json.dumps(good, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    records = load_traces(path)
    assert len(records) == 1 and records[0]["user_input"] == "正常"
    assert load_traces(tmp_path / "不存在") == []


def test_trace_path_uses_record_date(tmp_path):
    assert trace_path(tmp_path, when="2026-10-16 09:00:00").name == "2026-10-16.jsonl"


# ---------------- 回放 ----------------


def test_session_transcript_and_sessions_of(tmp_path):
    rows = [
        build_turn_record(make_state(), session_id="s-1", user_input="a2", turn=2),
        build_turn_record(make_state(), session_id="s-2", user_input="b1", turn=1),
        build_turn_record(make_state(), session_id="s-1", user_input="a1", turn=1),
    ]
    assert sessions_of(rows) == ["s-1", "s-2"]
    transcript = session_transcript(rows, "s-1")
    assert [item["user_input"] for item in transcript] == ["a1", "a2"]
    assert session_transcript(rows, "s-9") == []


# ---------------- 写库 ----------------


def test_save_turn_to_db_writes_session_and_messages(tmp_path):
    db_path = tmp_path / "trace_test.db"
    init_db(db_path)
    record = build_turn_record(
        make_state(handoff=True, handoff_reason="用户要求转人工"),
        session_id="s-db",
        user_input="我要转人工",
        turn=3,
    )
    assert save_turn_to_db(record, db_path=db_path) is True

    conn = connect(db_path)
    try:
        session = conn.execute(
            "SELECT title, status, last_message FROM sessions WHERE session_id = ?", ("s-db",)
        ).fetchone()
        messages = conn.execute(
            "SELECT role, content, intent FROM messages WHERE session_id = ? ORDER BY rowid",
            ("s-db",),
        ).fetchall()
    finally:
        conn.close()

    assert session["title"] == "我要转人工"
    assert session["status"] == "handoff"                      # 转人工的会话在库里可筛
    assert session["last_message"] == record["reply"][:200]
    assert [row["role"] for row in messages] == ["user", "assistant"]
    assert messages[0]["intent"] == "bad_fruit_claim"


def test_save_turn_to_db_failure_does_not_raise(tmp_path):
    """写库失败只告警：trace 不该把对话带崩。"""
    record = build_turn_record(make_state(), session_id="s", user_input="x", turn=1)
    assert save_turn_to_db(record, db_path=tmp_path) is False   # 目录当库路径 -> 连不上
    assert DB_PATH.name == "after_sales.db"                     # 正式库路径没被改动
