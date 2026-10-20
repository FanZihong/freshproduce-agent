"""judge 脚本的纯逻辑测试：渲染、解析、缓存、指标、抽样与校准。

调用大模型的部分用假的 complete_fn 注入，所以这些用例不联网、不需要 API key。
"""

from __future__ import annotations

import json

import pytest

from scripts.judge_dialogues import (
    PROMPT_VERSION,
    calibrate,
    cache_key,
    dry_run_verdict,
    group_judge,
    judge_record,
    parse_verdict,
    refresh_review,
    render_dialogue,
    run_judge,
    sample_for_review,
    summarize_judge,
    verdict_to_binary,
    write_report,
)


def make_record(**overrides) -> dict:
    record = {
        "id": "dg-99",
        "scenario": "查单-单轮命中订单号",
        "chain": "order",
        "category": "core",
        "resolution_criteria": "给出订单状态且带订单号",
        "note": "",
        "turn_log": [
            {
                "index": 1,
                "user": "查一下物流",
                "reply": "订单 2026093000123 已发货。",
                "intent": "query_order",
                "branches": ["order"],
                "handoff": False,
            }
        ],
        "facts": {"handoff": False, "handoff_summary": "", "tools": ["query_order"]},
        "tools": {
            "calls": 1,
            "success": 1,
            "retried_success": 0,
            "degraded": 0,
            "failed": 0,
            "by_tool": {"query_order": 1},
        },
        "checks": {"tools": True},
        "resolved_predicted": True,
        "resolved_expected": True,
        "resolved_match": True,
        "handoff_expected": False,
        "handoff_actual": False,
        "handoff_ok": True,
    }
    record.update(overrides)
    return record


def fake_complete(payload: str, counter: list | None = None):
    """假模型：原样返回 payload，并记录调用次数。"""

    def complete(messages, model, timeout):
        if counter is not None:
            counter.append(model)
        return payload, 100, 20

    return complete


# ---------------- 渲染 ----------------


def test_render_dialogue_includes_criteria_turns_tools_and_handoff():
    record = make_record(
        facts={"handoff": True, "handoff_summary": "订单 2026093000123；诉求：坏果理赔"},
        turn_log=[
            {"index": 1, "user": "东西坏了", "reply": "请提供订单号", "intent": "bad_fruit_claim"},
            {"index": 2, "user": "订单 2026093000123", "reply": "已转人工", "intent": "bad_fruit_claim"},
        ],
    )
    text = render_dialogue(record)
    assert "给出订单状态且带订单号" in text
    assert "东西坏了" in text and "订单 2026093000123" in text
    assert "query_order×1" in text
    assert "【是否转人工】是" in text and "订单 2026093000123；诉求：坏果理赔" in text


def test_render_dialogue_does_not_leak_rule_verdict():
    """提示里不能出现我们的判据结论，否则 judge 只是在抄规则版的答案。"""
    text = render_dialogue(make_record())
    assert "bad_fruit_claim" not in text
    assert "checks" not in text
    assert "resolved_predicted" not in text


# ---------------- 解析 ----------------


def test_parse_verdict_reads_json_and_fenced_json():
    assert parse_verdict('{"verdict": "达成", "reason": "给了状态"}') == ("达成", "给了状态")
    fenced = "```json\n{\"verdict\": \"未达成\", \"reason\": \"没给结论\"}\n```"
    assert parse_verdict(fenced) == ("未达成", "没给结论")


def test_parse_verdict_prefers_partial_over_resolved():
    """「部分达成」里含「达成」两字，匹配顺序错了就会判成完全达成。"""
    assert parse_verdict('{"verdict": "部分达成", "reason": "只办了一半"}')[0] == "部分达成"
    assert parse_verdict('{"verdict": "部分达成但不错", "reason": "x"}')[0] == "部分达成"


def test_parse_verdict_rejects_garbage():
    with pytest.raises(ValueError):
        parse_verdict("我觉得还行")
    with pytest.raises(ValueError):
        parse_verdict('{"verdict": "不知道", "reason": "x"}')


def test_verdict_to_binary_and_dry_run():
    assert verdict_to_binary("达成") == "达成"
    assert verdict_to_binary("部分达成") == "未达成"
    assert dry_run_verdict(make_record())[0] == "达成"
    partial = make_record(checks={"tools": True, "reply_contains": False})
    assert dry_run_verdict(partial)[0] == "部分达成"
    failed = make_record(checks={"tools": False})
    assert dry_run_verdict(failed)[0] == "未达成"


# ---------------- 调用与缓存 ----------------


def test_cache_key_changes_with_model_and_prompt_version(monkeypatch):
    record = make_record()
    base = cache_key(record, "deepseek-chat")
    assert base == cache_key(make_record(), "deepseek-chat")          # 内容相同 -> 键相同
    assert base != cache_key(record, "gpt-4o-mini")
    assert base != cache_key(make_record(scenario="别的场景"), "deepseek-chat")
    monkeypatch.setattr("scripts.judge_dialogues.PROMPT_VERSION", PROMPT_VERSION + "-test")
    assert base != cache_key(record, "deepseek-chat")                  # rubric 换版 -> 缓存失效


def test_judge_record_caches_and_counts_cost():
    calls: list[str] = []
    cache: dict = {}
    complete = fake_complete('{"verdict": "达成", "reason": "给了状态"}', calls)
    first = judge_record(make_record(), model="m", timeout=5, cache=cache, complete_fn=complete)
    assert first["verdict"] == "达成" and first["cached"] is False
    assert first["prompt_tokens"] == 100 and first["cost"] > 0
    second = judge_record(make_record(), model="m", timeout=5, cache=cache, complete_fn=complete)
    assert second["verdict"] == "达成" and second["cached"] is True
    assert len(calls) == 1                                             # 第二次走缓存，没有再调模型


def test_judge_record_records_errors_instead_of_raising():
    def broken(messages, model, timeout):
        raise RuntimeError("网络抖了")

    failed = judge_record(make_record(), model="m", timeout=5, cache={}, complete_fn=broken)
    assert failed["verdict"] == "" and "网络抖了" in failed["error"]

    bad_json = judge_record(
        make_record(), model="m", timeout=5, cache={}, complete_fn=fake_complete("抱歉，我不确定")
    )
    assert bad_json["verdict"] == "" and "解析失败" in bad_json["error"]


def test_run_judge_keeps_input_order(monkeypatch):
    records = [make_record(id=f"dg-{index:02d}") for index in range(1, 6)]
    results = run_judge(
        records,
        model="m",
        timeout=5,
        concurrency=3,
        cache={},
        complete_fn=fake_complete('{"verdict": "达成", "reason": "ok"}'),
        progress_every=0,
    )
    assert [item["id"] for item in results] == [record["id"] for record in records]
    assert all(item["verdict"] == "达成" for item in results)


# ---------------- 指标 ----------------


def test_summarize_judge_counts_and_rule_disagreement():
    records = [
        make_record(id="a", resolved_predicted=True),
        make_record(id="b", resolved_predicted=True),
        make_record(id="c", resolved_predicted=False),
        make_record(id="d", resolved_predicted=False),
    ]
    results = [
        {"id": "a", "verdict": "达成", "reason": "", "cached": False, "latency_ms": 10.0, "cost": 0.001},
        {"id": "b", "verdict": "未达成", "reason": "没给结论", "cached": False, "latency_ms": 20.0, "cost": 0.001},
        {"id": "c", "verdict": "达成", "reason": "", "cached": False, "latency_ms": 30.0, "cost": 0.001},
        {"id": "d", "verdict": "部分达成", "reason": "只办一半", "cached": False, "latency_ms": 40.0, "cost": 0.001},
    ]
    metrics = summarize_judge(records, results)
    assert metrics["judged"] == 4 and metrics["failed"] == 0
    assert metrics["verdict_counts"] == {"达成": 2, "部分达成": 1, "未达成": 1}
    assert metrics["resolved_rate"] == 0.5
    assert metrics["judge_vs_rule_agreement"] == 0.5
    assert metrics["rule_ok_judge_not"] == ["b"]
    assert metrics["judge_ok_rule_not"] == ["c"]
    assert metrics["latency_p95_ms"] == 40.0
    assert metrics["cost_per_1k"] == 1.0                               # 每次 0.001 × 1000


def test_summarize_judge_handles_all_failed():
    records = [make_record(id="a")]
    results = [{"id": "a", "verdict": "", "error": "超时", "cached": False, "latency_ms": 0.0, "cost": 0.0}]
    metrics = summarize_judge(records, results)
    assert metrics["judged"] == 0 and metrics["failed"] == 1
    assert metrics["resolved_rate"] == 0.0 and metrics["judge_vs_rule_agreement"] == 0.0


def test_group_judge_by_chain():
    records = [make_record(id="a", chain="order"), make_record(id="b", chain="faq")]
    results = [
        {"id": "a", "verdict": "达成", "cached": False, "latency_ms": 1.0, "cost": 0.0},
        {"id": "b", "verdict": "未达成", "cached": False, "latency_ms": 1.0, "cost": 0.0},
    ]
    rows = dict(group_judge(records, results, "chain"))
    assert rows["order"]["达成"] == 1 and rows["order"]["judge_vs_rule_agreement"] == 1.0
    assert rows["faq"]["未达成"] == 1 and rows["faq"]["judge_vs_rule_agreement"] == 0.0


# ---------------- 人工校准 ----------------


def test_sample_for_review_prioritizes_failures_and_disagreements():
    records = [
        make_record(id="dg-01", checks={"tools": True}),
        make_record(id="dg-11", checks={"tools": False}, resolved_predicted=False),   # 判据未过
        make_record(
            id="dg-02",
            checks={"reply_contains": True},
            resolved_expected=False,
            resolved_match=False,                                                     # 标注待复核
        ),
    ]
    results = [{"id": record["id"], "verdict": "达成", "reason": "x"} for record in records]
    rows = sample_for_review(records, results, sample=2, seed=1)
    picked = [row["id"] for row in rows]
    assert picked == ["dg-11", "dg-02"]                                  # 先失败、再待复核
    assert rows[0]["human_verdict"] == "" and rows[0]["human_note"] == ""
    assert rows[0]["judge_verdict"] == "达成" and rows[0]["rule_note"] == "判据未过"


def test_sample_for_review_is_deterministic_and_fills_to_size():
    records = [make_record(id=f"dg-{index:02d}", chain="order" if index % 2 else "faq")
               for index in range(1, 11)]
    results = [{"id": record["id"], "verdict": "达成", "reason": "x"} for record in records]
    first = [row["id"] for row in sample_for_review(records, results, sample=6, seed=7)]
    second = [row["id"] for row in sample_for_review(records, results, sample=6, seed=7)]
    assert first == second and len(first) == 6


def test_calibrate_computes_agreement_kappa_and_disagreements():
    rows = [
        {"id": "a", "scenario": "s", "human_verdict": "达成", "judge_verdict": "达成",
         "judge_reason": "", "human_note": ""},
        {"id": "b", "scenario": "s", "human_verdict": "未达成", "judge_verdict": "部分达成",
         "judge_reason": "只办一半", "human_note": "其实没办"},
        {"id": "c", "scenario": "s", "human_verdict": "", "judge_verdict": "达成",
         "judge_reason": "", "human_note": ""},
    ]
    report = calibrate(rows)
    assert report["compared"] == 2 and report["pending"] == 1
    assert report["agreement"] == 0.5
    assert report["passed"] is False
    assert [row["id"] for row in report["disagreements"]] == ["b"]


def test_calibrate_passes_at_target():
    rows = [
        {"id": str(index), "scenario": "s", "human_verdict": "达成", "judge_verdict": "达成",
         "judge_reason": "", "human_note": ""}
        for index in range(10)
    ] + [
        {"id": "x", "scenario": "s", "human_verdict": "未达成", "judge_verdict": "达成",
         "judge_reason": "", "human_note": ""}
    ]
    report = calibrate(rows)
    assert report["compared"] == 11 and report["agreement"] == round(10 / 11, 4)
    assert report["passed"] is True                                     # 0.909 > 0.8


def test_refresh_review_keeps_human_labels_and_updates_judge():
    """改完 rubric 重跑：judge 列要刷成本次结果，人工填的那两列必须原样保留。"""
    rows = [
        {"id": "a", "human_verdict": "部分达成", "human_note": "答得不清楚",
         "judge_verdict": "达成", "judge_reason": "旧理由"},
        {"id": "b", "human_verdict": "达成", "human_note": "",
         "judge_verdict": "达成", "judge_reason": "没变"},
        {"id": "c", "human_verdict": "达成", "human_note": "",
         "judge_verdict": "达成", "judge_reason": "本次没结果"},
    ]
    results = [
        {"id": "a", "verdict": "部分达成", "reason": "新理由"},
        {"id": "b", "verdict": "达成", "reason": "没变"},
    ]
    refreshed_rows, refreshed = refresh_review(rows, results)
    assert refreshed == 1
    assert refreshed_rows[0]["judge_verdict"] == "部分达成"
    assert refreshed_rows[0]["judge_reason"] == "新理由"
    assert refreshed_rows[0]["human_verdict"] == "部分达成"
    assert refreshed_rows[0]["human_note"] == "答得不清楚"
    assert refreshed_rows[1]["judge_reason"] == "没变"
    assert refreshed_rows[2]["judge_reason"] == "本次没结果"


# ---------------- 报告 ----------------


def test_write_report_with_and_without_calibration(tmp_path):
    records = [make_record(id="dg-01"), make_record(id="dg-02", chain="faq")]
    results = [
        {"id": "dg-01", "verdict": "达成", "reason": "给了状态", "cached": False,
         "latency_ms": 10.0, "cost": 0.001},
        {"id": "dg-02", "verdict": "未达成", "reason": "没答上来", "cached": False,
         "latency_ms": 12.0, "cost": 0.001},
    ]
    metrics = summarize_judge(records, results)
    path = tmp_path / "judge.md"
    write_report(
        path,
        input_path=tmp_path / "dialogues.json",
        model="deepseek-chat",
        dry_run=False,
        records=records,
        results=results,
        metrics=metrics,
        calibration=None,
    )
    text = path.read_text(encoding="utf-8")
    assert "## 一、judge 判定分布" in text
    assert "## 五、人工校准" in text and "--export-review" in text
    assert "没答上来" in text

    calibration = calibrate(
        [
            {"id": "dg-01", "scenario": "s", "human_verdict": "达成", "judge_verdict": "达成",
             "judge_reason": "", "human_note": ""},
            {"id": "dg-02", "scenario": "s", "human_verdict": "未达成", "judge_verdict": "达成",
             "judge_reason": "给了状态", "human_note": "其实用户还是没拿到结果"},
        ]
    )
    write_report(
        path,
        input_path=tmp_path / "dialogues.json",
        model="deepseek-chat",
        dry_run=False,
        records=records,
        results=results,
        metrics=metrics,
        calibration=calibration,
    )
    text = path.read_text(encoding="utf-8")
    assert "一致率 **0.5000**" in text
    assert "其实用户还是没拿到结果" in text
