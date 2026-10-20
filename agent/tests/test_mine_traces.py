"""trace 挖掘的测试：信号判定、去重排序、池子输出、报告与回放。

全部用内存里的 trace 记录，不依赖真实对话。
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from collections import Counter

from scripts.mine_traces import (
    SIGNALS,
    build_candidate,
    is_slot_continuation,
    load_pool,
    main,
    merge_pool,
    mine,
    print_transcript,
    signal_counts,
    signals_of,
    slot_continuation_count,
    write_pool,
    write_report,
)


def make_trace(**overrides) -> dict:
    record = {
        "trace_version": 1,
        "created_at": "2026-10-16 10:00:00",
        "session_id": "cli-1",
        "turn": 1,
        "user_input": "草莓坏了一半",
        "backend": "local-lora",
        "transport": "direct",
        "intent": "bad_fruit_claim",
        "confidence": 0.93,
        "intent_labels": ["bad_fruit_claim"],
        "branches": ["claim"],
        "multi_strategy": "priority",
        "decision": "accept",
        "clarify_rounds": 0,
        "out_of_domain": False,
        "ood_reason": "",
        "active_flow": "bad_fruit_claim",
        "pending_slot": "order_id",
        "loop_count": 0,
        "slots": {},
        "tools": [],
        "handoff": False,
        "handoff_reason": "",
        "handoff_summary": "",
        "citations": 0,
        "reply": "请提供订单号，我来核对订单信息。",
    }
    record.update(overrides)
    return record


# ---------------- 信号判定 ----------------


def test_signals_of_clean_turn_has_no_hit():
    assert signals_of(make_trace()) == []


def test_signals_of_covers_each_rule():
    assert [s.key for s in signals_of(make_trace(reply=""))] == ["empty_reply"]
    failed = make_trace(
        tools=[{"name": "query_order", "ok": False, "action": "handoff", "attempts": 1}]
    )
    assert [s.key for s in signals_of(failed)] == ["tool_failed"]
    assert [s.key for s in signals_of(make_trace(loop_count=3))] == ["loop_guard"]
    assert [s.key for s in signals_of(make_trace(clarify_rounds=2))] == ["loop_guard"]
    assert [s.key for s in signals_of(make_trace(out_of_domain=True))] == ["ood"]
    assert [s.key for s in signals_of(make_trace(handoff=True))] == ["handoff"]
    assert [s.key for s in signals_of(make_trace(decision="clarify"))] == ["clarify"]
    assert [s.key for s in signals_of(make_trace(branches=["claim", "faq"]))] == ["multi_intent"]


def test_low_confidence_only_counts_when_let_through():
    low = make_trace(confidence=0.4)
    assert [s.key for s in signals_of(low)] == ["low_confidence"]
    # 走到澄清的已经在 clarify 里，不再重复算低置信
    assert "low_confidence" not in [
        s.key for s in signals_of(make_trace(confidence=0.4, decision="clarify"))
    ]
    # 域外不看置信度
    assert "low_confidence" not in [
        s.key for s in signals_of(make_trace(confidence=0.4, out_of_domain=True))
    ]
    # 阈值可调
    assert signals_of(make_trace(confidence=0.55), min_confidence=0.5) == []
    assert [s.key for s in signals_of(make_trace(confidence=0.55), min_confidence=0.6)] == [
        "low_confidence"
    ]


def test_slot_filling_turns_are_not_low_confidence():
    """补槽位轮：用户回的是订单号/比例这种碎片，标签不影响路由，低置信属正常。

    首次跑 trace 就踩到：4 条候选全是补槽位轮，把池子灌成了噪声。
    """
    slot_turn = make_trace(
        confidence=0.35,
        intent="chitchat",
        slots={"order_id": "2026093000123"},
        active_flow="bad_fruit_claim",
    )
    assert is_slot_continuation(slot_turn) is True
    assert signals_of(slot_turn) == []
    # 显式要求时才收进来
    assert [s.key for s in signals_of(slot_turn, include_slot_turns=True)] == ["low_confidence"]

    # 没有槽位（会话刚开始就说了一句低置信的话）仍然算候选
    cold = make_trace(confidence=0.35, intent="chitchat")
    assert is_slot_continuation(cold) is False
    assert [s.key for s in signals_of(cold)] == ["low_confidence"]
    # 转人工的轮次不按"填槽"处理
    assert is_slot_continuation(make_trace(confidence=0.35, intent="chitchat", slots={"a": 1}, handoff=True)) is False


def test_slot_continuation_count_reports_skipped_turns():
    records = [
        make_trace(turn=1, confidence=0.35, intent="chitchat", slots={"order_id": "x"}),
        make_trace(turn=2, confidence=0.35, intent="chitchat"),
        make_trace(turn=3, confidence=0.9, intent="chitchat", slots={"order_id": "x"}),
    ]
    assert slot_continuation_count(records, min_confidence=0.5) == 1
    assert "low_confidence" in [
        s.key for s in signals_of(records[1])
    ]


def test_signals_can_stack_and_sort_by_severity():
    record = make_trace(
        reply="",
        tools=[{"name": "search_faq", "ok": False, "action": "handoff", "attempts": 1}],
        handoff=True,
        branches=["claim", "faq"],
    )
    keys = [s.key for s in signals_of(record)]
    assert keys[:2] == ["empty_reply", "tool_failed"]      # 严重度 5 > 4
    assert set(keys) == {"empty_reply", "tool_failed", "handoff", "multi_intent"}


# ---------------- 挖掘与去重 ----------------


def test_mine_builds_candidate_with_context():
    records = [
        make_trace(turn=1, user_input="东西坏了"),
        make_trace(turn=2, user_input="订单 2026093000123", intent="query_order"),
        make_trace(turn=3, user_input="我要转人工", handoff=True, handoff_reason="用户要求转人工"),
    ]
    candidates = mine(records)
    assert len(candidates) == 1
    item = candidates[0]
    assert item["id"] == "trace-cli-1-t3"
    assert item["signals"] == ["转人工"]
    assert item["observed"]["handoff"] is True
    assert item["expected_intent"] == "" and item["status"] == "pending"
    # 上下文里要有前面两轮，人工才能判断"是不是真该转人工"
    assert [row["user"] for row in item["transcript"]] == [
        "东西坏了",
        "订单 2026093000123",
        "我要转人工",
    ]


def test_mine_dedupes_same_text_and_counts_seen():
    records = [
        make_trace(session_id="s-1", turn=1, user_input="没有了", out_of_domain=True),
        make_trace(session_id="s-2", turn=1, user_input="  没有了 ", out_of_domain=True),
        make_trace(session_id="s-3", turn=1, user_input="算了", out_of_domain=True),
    ]
    candidates = mine(records)
    texts = [item["text"] for item in candidates]
    assert texts == ["没有了", "算了"]
    assert candidates[0]["seen"] == 2
    assert candidates[0]["sessions"] == ["s-1", "s-2"]


def test_mine_sorts_by_severity_and_honours_limit_and_session_filter():
    records = [
        make_trace(session_id="s-1", turn=1, user_input="a", out_of_domain=True),   # 3
        make_trace(session_id="s-1", turn=2, user_input="b", reply=""),             # 5
        make_trace(session_id="s-2", turn=1, user_input="c", handoff=True),         # 3
    ]
    candidates = mine(records)
    assert [item["text"] for item in candidates] == ["b", "a", "c"]
    assert [item["text"] for item in mine(records, limit=2)] == ["b", "a"]
    assert [item["text"] for item in mine(records, session="s-2")] == ["c"]


def test_mine_can_filter_by_backend():
    records = [
        make_trace(session_id="s-1", user_input="a", handoff=True, backend="local-lora"),
        make_trace(session_id="s-2", user_input="b", handoff=True, backend="rule-based-v1"),
    ]
    assert [item["text"] for item in mine(records, backend="local-lora")] == ["a"]
    assert [item["text"] for item in mine(records, backend="rule-based-v1")] == ["b"]
    assert len(mine(records)) == 2


def test_signal_counts_counts_turns_not_candidates():
    records = [
        make_trace(turn=1, out_of_domain=True),
        make_trace(turn=2, out_of_domain=True),
        make_trace(turn=3, handoff=True),
    ]
    counts = signal_counts(records)
    assert counts["ood"] == 2 and counts["handoff"] == 1


def test_build_candidate_reason_joins_signal_explanations():
    record = make_trace(handoff=True, out_of_domain=True)
    item = build_candidate(record, signals_of(record), [record])
    assert item["reason"].count("；") == 1
    assert "误转" in item["reason"] and "误杀" in item["reason"]


# ---------------- 输出 ----------------


def test_write_pool_roundtrip(tmp_path):
    path = tmp_path / "pool.jsonl"
    write_pool(path, mine([make_trace(handoff=True)]))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["signals"] == ["转人工"]
    write_pool(path, [])
    assert path.read_text(encoding="utf-8") == ""
    assert load_pool(path) == []


def test_merge_pool_keeps_human_labels_and_drops_stale_rows():
    """池子是合并写入的：人工标注不能因为重跑丢失，陈数据也不能越滚越多。"""
    existing = [
        {
            "id": "trace-s-1-t1",
            "text": "旧文本",
            "expected_intent": "query_order",
            "expected_action": "该查单",
            "note": "人工写的",
            "status": "labeled",
        },
        {"id": "trace-s-9-t9", "text": "陈数据", "status": "pending",
         "expected_intent": "", "expected_action": "", "note": ""},
        {
            "id": "trace-s-8-t8",
            "text": "别的会话里已经标过的行",
            "expected_intent": "chitchat",
            "expected_action": "",
            "note": "留着我",
            "status": "labeled",
        },
    ]
    candidates = [{"id": "trace-s-1-t1", "text": "新文本", "signals": ["转人工"], "status": "pending"}]
    merged, stats = merge_pool(existing, candidates)
    by_id = {row["id"]: row for row in merged}

    assert by_id["trace-s-1-t1"]["text"] == "新文本"            # 事实以本次为准
    assert by_id["trace-s-1-t1"]["expected_intent"] == "query_order"   # 人工列搬过来
    assert by_id["trace-s-1-t1"]["note"] == "人工写的"
    assert by_id["trace-s-1-t1"]["status"] == "labeled"
    assert "trace-s-9-t9" not in by_id                          # 没标注又没挖到 -> 丢掉
    assert by_id["trace-s-8-t8"]["note"] == "留着我"            # 已标注 -> 永久保留
    assert stats == {"added": 0, "carried": 1, "kept_labelled": 1}


def test_merge_pool_counts_new_rows():
    merged, stats = merge_pool([], [{"id": "a", "status": "pending"}])
    assert stats == {"added": 1, "carried": 0, "kept_labelled": 0}
    assert [row["id"] for row in merged] == ["a"]


def test_print_transcript_prints_each_turn_and_handles_unknown_session():
    records = [
        make_trace(turn=1, user_input="东西坏了"),
        make_trace(turn=2, user_input="我要转人工", handoff=True, handoff_reason="用户要求转人工"),
    ]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert print_transcript(records, "cli-1") == 0
    text = buffer.getvalue()
    assert "会话 cli-1（2 轮）" in text
    assert "用户：我要转人工" in text and "转人工：用户要求转人工" in text

    with contextlib.redirect_stdout(io.StringIO()):
        assert print_transcript(records, "不存在") == 1


def test_write_report_renders_signals_and_next_steps(tmp_path):
    records = [make_trace(handoff=True), make_trace(turn=2, reply="")]
    candidates = mine(records)
    path = tmp_path / "mining.md"
    write_report(
        path,
        trace_path=tmp_path / "trace",
        records=records,
        candidates=candidates,
        counts=signal_counts(records),
        backends=Counter({"rule-based-v1": 2}),
        slot_skipped=1,
        backend="rule-based-v1",
    )
    text = path.read_text(encoding="utf-8")
    assert "## 一、信号分布" in text and "## 三、怎么用这一池候选" in text
    assert "| 转人工 | 1 |" in text and "| 空回复 | 1 |" in text
    assert "evolve_intent.py" in text
    # 报告要写清这批数据是哪个后端产的、以及跳过了多少补槽位轮
    assert "rule-based-v1 2 轮" in text and "已跳过 1 轮" in text
    assert "--backend rule-based-v1" in text
    # 信号表要覆盖全部注册信号，方便一眼看出哪类信号一条都没挖到
    for signal in SIGNALS:
        assert signal.label in text


def test_main_end_to_end_and_empty_trace_hint(tmp_path, monkeypatch):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "2026-10-16.jsonl").write_text(
        json.dumps(make_trace(handoff=True), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    out = tmp_path / "pool.jsonl"
    report = tmp_path / "mining.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["mine_traces.py", "--trace", str(trace_dir), "--out", str(out), "--report", str(report)],
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert main() == 0
    assert out.exists() and report.exists()
    assert "候选 badcase：1 条" in buffer.getvalue()

    empty = tmp_path / "empty"
    monkeypatch.setattr(
        sys, "argv", ["mine_traces.py", "--trace", str(empty), "--out", str(out)]
    )
    with contextlib.redirect_stdout(io.StringIO()):
        assert main() == 1                        # 没有 trace 要明确报错，不是静默成功


def test_main_print_transcript_exits_after_replay(tmp_path, monkeypatch):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "2026-10-16.jsonl").write_text(
        json.dumps(make_trace(user_input="回放我"), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["mine_traces.py", "--trace", str(trace_dir), "--print-transcript", "cli-1"],
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert main() == 0
    assert "回放我" in buffer.getvalue()
    assert "候选 badcase" not in buffer.getvalue()   # 回放模式不应该顺手挖一遍


def test_main_does_not_truncate_pool_when_filter_finds_nothing(tmp_path, monkeypatch):
    """筛错后端时池子会被写空？不会：没候选就不动原文件，并提示全量里有多少。"""
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    rows = [
        make_trace(session_id="s-1", user_input="有候选的一句", handoff=True, backend="rule-based-v1"),
        make_trace(session_id="s-2", user_input="2026093000123", backend="local-lora",
                   intent="chitchat", confidence=0.9, slots={"order_id": "x"}),
    ]
    (trace_dir / "2026-10-16.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    out = tmp_path / "pool.jsonl"
    out.write_text(
        json.dumps(
            {
                "id": "trace-s-9-t9",
                "text": "以前标过的一条",
                "expected_intent": "invoice",
                "expected_action": "",
                "note": "别丢",
                "status": "labeled",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["mine_traces.py", "--trace", str(trace_dir), "--out", str(out),
         "--report", str(tmp_path / "m.md"), "--backend", "local-lora"],
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert main() == 0
    text = buffer.getvalue()
    assert "本次范围：1 轮" in text
    assert "去掉筛选后全量数据里有 1 条候选" in text
    pool = load_pool(out)
    assert [row["id"] for row in pool] == ["trace-s-9-t9"]     # 老的标注还在
    assert pool[0]["note"] == "别丢"


def test_main_leaves_no_file_when_nothing_mined(tmp_path, monkeypatch):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "2026-10-16.jsonl").write_text(
        json.dumps(
            make_trace(backend="local-lora", intent="quality_qa", confidence=0.9, slots={"a": 1}),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "pool.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["mine_traces.py", "--trace", str(trace_dir), "--out", str(out), "--no-report"]
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert main() == 0
    assert "未创建池子文件" in buffer.getvalue()
    assert not out.exists()
