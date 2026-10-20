"""自动回归脚本的纯逻辑测试：抽取、门禁判定、代码指纹、阶段编排与报告。

这些用例都不执行子进程，只用内存里的 JSON 走一遍"抽取 → 对比 → 出报告"。
"""

from __future__ import annotations

from pathlib import Path

from scripts.regression import (
    METRIC_SPECS,
    MetricSpec,
    build_stages,
    code_fingerprint,
    compare_all,
    compare_metric,
    diff_fingerprint,
    extract_dialogues,
    extract_intent,
    extract_judge,
    extract_multi,
    extract_ood,
    extract_retrieval,
    merge_baseline,
    run_stage,
    write_report,
)

# ---------------- 抽取：用的是各评测脚本真实产出的 JSON 结构 ----------------


def test_extract_retrieval_reads_each_config():
    payload = {
        "configs": [
            {"config": "bm25+rw", "metrics": {"recall@1": 0.5, "recall@5": 0.9, "mrr": 0.6}},
            {"config": "hybrid+rw", "metrics": {"recall@1": 0.7, "recall@5": 1.0, "mrr": 0.8}},
        ]
    }
    metrics = extract_retrieval(payload)
    assert metrics["retrieval.hybrid+rw.recall@5"] == 1.0
    assert metrics["retrieval.bm25+rw.mrr"] == 0.6
    assert "retrieval.hybrid+rw.recall@3" not in metrics          # 没跑 k=3 就不编数


def test_extract_intent_picks_backend_and_survives_missing_fields():
    payload = {
        "backends": [
            {"backend": "fewshot", "split": "test", "macro_f1": 0.85, "queries": 701},
            {"backend": "local", "split": "dev", "macro_f1": 0.7178, "queries": 704},
            {"backend": "local", "split": "test", "macro_f1": 0.7085, "queries": 701,
             "local_dir": "data/intent/models/v2"},
            {"backend": "local", "split": "test", "macro_f1": 0.7366, "accuracy": 0.7375,
             "queries": 701, "local_dir": "data/intent/models/v1",
             "latency": {"p95_ms": 1470.0}, "errors": 2},
        ]
    }
    metrics = extract_intent(payload)
    # 同名后端有多行（dev / v2 / v1）：必须按 split=test 过滤，并取最后一条
    assert metrics["intent.local.macro_f1"] == 0.7366
    assert metrics["intent.local.local_dir"] == "data/intent/models/v1"
    assert metrics["intent.local.queries"] == 701
    assert metrics["intent.local.p95_ms"] == 1470.0
    assert metrics["intent.local.errors"] == 2
    assert extract_intent({"backends": [{"backend": "fewshot"}]}) == {}   # 没有 local 就返回空
    assert extract_intent(payload, split=None)["intent.local.macro_f1"] == 0.7366


def test_extract_intent_drops_latency_when_everything_cached():
    """全命中缓存时时延是 0，报出去会被读成"变快了"，所以干脆不产出这个指标。"""
    payload = {
        "backends": [
            {"backend": "local", "split": "test", "queries": 701, "cached": 701,
             "macro_f1": 0.7366, "latency": {"p95_ms": 0.0}},
        ]
    }
    metrics = extract_intent(payload)
    assert "intent.local.p95_ms" not in metrics
    assert metrics["intent.local.cached"] == 701

    partial = {
        "backends": [
            {"backend": "local", "split": "test", "queries": 701, "cached": 700,
             "macro_f1": 0.7366, "latency": {"p95_ms": 1470.0}},
        ]
    }
    assert extract_intent(partial)["intent.local.p95_ms"] == 1470.0


def test_extract_ood_reads_three_scopes():
    payload = {
        "model_only": {"ood_recall": 0.94, "false_kill_rate": 0.0684},
        "with_signals": {"ood_recall": 0.88, "false_kill_rate": 0.0501},
    }
    metrics = extract_ood(payload)
    assert metrics["ood.with_signals.recall"] == 0.88
    assert metrics["ood.model_only.false_kill_rate"] == 0.0684
    assert "ood.with_signals_no_veto.recall" not in metrics


def test_extract_multi_reads_strategies_and_multilabel():
    payload = {
        "strategies": {
            "priority": {"both_covered_rate": 0.24, "secondary_coverage": 0.38,
                         "primary_hit_rate": 0.84, "multilabel": {"macro": 0.7784}},
            "parallel": {"both_covered_rate": 0.76, "secondary_coverage": 0.9,
                         "primary_hit_rate": 0.84, "multilabel": {"macro": 0.7784}},
        }
    }
    metrics = extract_multi(payload)
    assert metrics["multi.parallel.both_covered_rate"] == 0.76
    assert metrics["multi.multilabel_macro"] == 0.7784            # 识别层与策略无关


def test_extract_dialogues_computes_tool_success_rate():
    payload = {
        "metrics": {
            "dialogues": 2, "behavior_ok_rate": 1.0, "resolution_rate": 1.0,
            "expected_unresolved_ok_rate": 1.0, "handoff_recall": 1.0, "handoff_false_rate": 0.0,
        },
        "records": [
            {"tools": {"calls": 2, "success": 2}},
            {"tools": {"calls": 2, "success": 1}},
        ],
    }
    metrics = extract_dialogues(payload)
    assert metrics["dialogues.tool_success_rate"] == 0.75          # 3/4
    assert metrics["dialogues.behavior_ok_rate"] == 1.0


def test_extract_judge_merges_calibration_when_present():
    payload = {
        "metrics": {"resolved_rate": 1.0, "partial_rate": 0.0, "unresolved_rate": 0.0,
                    "judge_vs_rule_agreement": 1.0, "failed": 0, "cost_total": 0.0515},
        "calibration": {"agreement": 0.9, "compared": 10},
    }
    metrics = extract_judge(payload)
    assert metrics["judge.agreement"] == 0.9 and metrics["judge.compared"] == 10
    assert metrics["judge.cost_total"] == 0.0515
    assert "judge.agreement" not in extract_judge({"metrics": payload["metrics"]})  # 没校准就不报


# ---------------- 门禁：方向、容差、退化 ----------------


def test_compare_metric_improvement_and_regression():
    up = MetricSpec("x", "x", "s", "up", 0.01)
    assert compare_metric(up, 0.80, 0.85)["status"] == "改善"
    assert compare_metric(up, 0.80, 0.795)["status"] == "持平"     # 容差内的小幅下降
    assert compare_metric(up, 0.80, 0.74)["status"] == "退化"
    assert compare_metric(up, 0.80, 0.74)["delta"] == -0.06

    down = MetricSpec("y", "y", "s", "down", 0.01)
    assert compare_metric(down, 0.05, 0.01)["status"] == "改善"
    assert compare_metric(down, 0.05, 0.055)["status"] == "持平"
    assert compare_metric(down, 0.05, 0.09)["status"] == "退化"


def test_compare_metric_relative_tolerance_for_latency():
    """时延适合相对容差：1000ms 的 25% 是 250ms。"""
    spec = MetricSpec("lat", "lat", "s", "down", 0.0, 0.25)
    assert compare_metric(spec, 1000.0, 1200.0)["status"] == "持平"
    assert compare_metric(spec, 1000.0, 1300.0)["status"] == "退化"
    assert compare_metric(spec, 1000.0, 800.0)["status"] == "改善"


def test_compare_metric_handles_info_missing_and_new():
    info = MetricSpec("cost", "cost", "s", "info")
    assert compare_metric(info, 0.05, 0.06)["status"] == "仅记录"
    up = MetricSpec("x", "x", "s", "up", 0.01)
    assert compare_metric(up, None, 0.8)["status"] == "新增"
    assert compare_metric(up, 0.8, None)["status"] == "缺失"


def test_compare_all_covers_every_registered_spec():
    rows = compare_all(METRIC_SPECS, {"intent.local.macro_f1": 0.7366}, {"intent.local.macro_f1": 0.75})
    assert len(rows) == len(METRIC_SPECS)
    keys = {row["key"] for row in rows}
    assert "dialogues.behavior_ok_rate" in keys and "judge.agreement" in keys
    assert "retrieval.hybrid+rw+score.recall@5" in keys          # 生产口径也要进门禁
    assert all("status" in row for row in rows)


def test_merge_baseline_keeps_metrics_not_remeasured():
    """只跑了一两个阶段时更新基线，不能把没跑的指标抹掉。"""
    baseline = {
        "created_at": "2026-10-15 22:09:10",
        "label": "retrieval、intent、ood",
        "metrics": {"intent.local.macro_f1": 0.7178, "dialogues.behavior_ok_rate": 1.0},
        "code_fingerprint": {"scripts/a.py": "aaaa"},
    }
    current = {
        "created_at": "2026-10-15 23:00:00",
        "label": "intent",
        "metrics": {"intent.local.macro_f1": 0.7366},
        "code_fingerprint": {"scripts/a.py": "bbbb"},
    }
    merged = merge_baseline(baseline, current)
    assert merged["metrics"] == {"intent.local.macro_f1": 0.7366, "dialogues.behavior_ok_rate": 1.0}
    assert merged["metrics"]["dialogues.behavior_ok_rate"] == 1.0     # 没重测的沿用旧值
    assert merged["code_fingerprint"] == {"scripts/a.py": "bbbb"}     # 指纹用本次的
    assert merged["merged_from"]["reused_metrics"] == ["dialogues.behavior_ok_rate"]


def test_merge_baseline_without_previous_returns_current():
    current = {"created_at": "x", "metrics": {"a": 1.0}}
    assert merge_baseline(None, current) == current


def test_partial_run_marks_untested_stages():
    """没跑的阶段标"未测"，不要标"缺失"——否则报告像全线崩了。"""
    rows = compare_all(
        METRIC_SPECS,
        {"retrieval.hybrid+rw.recall@5": 0.83, "dialogues.behavior_ok_rate": 1.0},
        {"retrieval.hybrid+rw.recall@5": 0.83},
    )
    ran = {"retrieval"}
    for row in rows:
        if row["stage"] not in ran and row["status"] in {"缺失", "新增"}:
            row["status"] = "未测"
    statuses = {row["key"]: row["status"] for row in rows}
    assert statuses["retrieval.hybrid+rw.recall@5"] == "持平"
    assert statuses["dialogues.behavior_ok_rate"] == "未测"     # 阶段没跑
    assert statuses["retrieval.hybrid+rw.recall@1"] == "缺失"   # 阶段跑了，但这个指标没抽到


# ---------------- 代码指纹与归因 ----------------


def test_code_fingerprint_and_diff(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "a.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "scripts" / "keep.py").write_text("x = 1\n", encoding="utf-8")
    first = code_fingerprint(tmp_path, dirs=("scripts",))
    assert set(first) == {"scripts/a.py", "scripts/keep.py"}

    (tmp_path / "scripts" / "a.py").write_text("print(2)\n", encoding="utf-8")
    (tmp_path / "scripts" / "b.py").write_text("y = 2\n", encoding="utf-8")
    second = code_fingerprint(tmp_path, dirs=("scripts",))
    diff = diff_fingerprint(first, second)
    assert diff["added"] == ["scripts/b.py"]
    assert diff["modified"] == ["scripts/a.py"]
    assert diff["removed"] == []

    (tmp_path / "scripts" / "keep.py").unlink()
    diff2 = diff_fingerprint(second, code_fingerprint(tmp_path, dirs=("scripts",)))
    assert diff2["removed"] == ["scripts/keep.py"]


# ---------------- 阶段编排 ----------------


def test_build_stages_order_and_judge_toggle(tmp_path):
    names = [stage.name for stage in build_stages(with_judge=True, review_file=tmp_path / "无.jsonl")]
    assert names == ["retrieval", "intent", "ood", "multi", "dialogues", "judge"]
    without = [stage.name for stage in build_stages(with_judge=False, review_file=tmp_path / "无.jsonl")]
    assert without == ["retrieval", "intent", "ood", "multi", "dialogues"]


def test_build_stages_passes_review_file_only_when_it_exists(tmp_path):
    review = tmp_path / "judge_review.jsonl"
    review.write_text("{}\n", encoding="utf-8")
    stages = {stage.name: stage for stage in build_stages(with_judge=True, review_file=review)}
    assert "--review" in stages["judge"].args
    assert str(review) in stages["judge"].args
    plain = {stage.name: stage for stage in build_stages(with_judge=True, review_file=tmp_path / "x.jsonl")}
    assert "--review" not in plain["judge"].args


def test_run_stage_dry_run_does_not_execute():
    stage = build_stages(with_judge=False, review_file=Path("无.jsonl"))[0]
    record = run_stage(stage, python="python3", timeout=0, dry_run=True)
    assert record["ok"] is True and record["metrics"] == {}
    assert record["error"] == "dry-run：未执行"
    assert record["command"][1].endswith("eval_retrieval.py")


# ---------------- 报告 ----------------


def test_write_report_marks_red_items_and_changed_files(tmp_path):
    comparison = [
        compare_metric(MetricSpec("intent.local.macro_f1", "意图 macro-F1", "intent", "up", 0.01),
                       0.7366, 0.7000),
        compare_metric(MetricSpec("dialogues.behavior_ok_rate", "判据通过率", "dialogues", "up", 0.01),
                       1.0, 1.0),
    ]
    stages = [{"name": "intent", "title": "意图", "duration_s": 12.0, "returncode": 0,
               "ok": True, "error": "", "output_tail": []}]
    current = {"created_at": "2026-10-15 18:00:00", "git": {"available": False, "note": "不是仓库"}}
    baseline = {"created_at": "2026-10-15 17:00:00", "label": "上次"}
    path = tmp_path / "regression.md"
    write_report(
        path,
        comparison=comparison,
        stages=stages,
        baseline_payload=baseline,
        current_payload=current,
        fingerprint_diff={"added": [], "removed": [], "modified": ["scripts/regression.py"]},
        commit_lines=[],
    )
    text = path.read_text(encoding="utf-8")
    assert "**有红项**" in text
    assert "| 意图 macro-F1 | 0.7366 | 0.7000 | -0.0366 | ±0.01 | **退化** |" in text
    assert "修改 scripts/regression.py" in text
    assert "无 git 信息" in text


def test_write_report_without_baseline(tmp_path):
    path = tmp_path / "regression.md"
    write_report(
        path,
        comparison=[],
        stages=[],
        baseline_payload=None,
        current_payload={"created_at": "2026-10-15 18:00:00", "git": {"available": False, "note": "-"}},
        fingerprint_diff={"added": [], "removed": [], "modified": []},
        commit_lines=[],
    )
    text = path.read_text(encoding="utf-8")
    assert "本次已记为基线" in text
    assert "本次没有基线" in text
