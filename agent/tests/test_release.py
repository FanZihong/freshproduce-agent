"""发布门禁的测试：路径匹配、决策规则、决策卡、active 版本与 CLI 编排。"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

from agent.intent import active as active_module
from scripts.release import (
    _same_dir,
    decide,
    find_row,
    main,
    regression_summary,
    render_card,
)


def make_baseline_payload(**rows) -> dict:
    return {"backends": list(rows.values())}


def make_row(local_dir: str, f1: float, per_class: dict | None = None, split: str = "test") -> dict:
    return {
        "backend": "local",
        "split": split,
        "local_dir": local_dir,
        "macro_f1": f1,
        "per_class": per_class or {},
    }


def good_classes(f1: float) -> dict:
    return {name: {"f1": f1} for name in ("quality_qa", "chitchat", "human_handoff")}


# ---------------- 路径匹配 ----------------


def test_same_dir_matches_relative_and_absolute():
    assert _same_dir("data/intent/models/v1", "/mnt/d/生鲜售后agent/agent/data/intent/models/v1")
    assert _same_dir("data/intent/models/v1", "data/intent/models/v1/")
    assert not _same_dir("data/intent/models/v1", "data/intent/models/v3")
    assert not _same_dir("", "data/intent/models/v1")


def test_find_row_needs_matching_backend_and_split():
    payload = make_baseline_payload(
        a=make_row("data/intent/models/v1", 0.7366),
        b=make_row("data/intent/models/v1", 0.7, split="dev"),
        c={"backend": "fewshot", "split": "test", "macro_f1": 0.85},
    )
    assert find_row(payload, "data/intent/models/v1")["macro_f1"] == 0.7366
    assert find_row(payload, "data/intent/models/v1", split="dev")["macro_f1"] == 0.7
    assert find_row(payload, "data/intent/models/v9") is None


# ---------------- 决策规则 ----------------


def test_decide_rejects_when_gate_missing_rows():
    verdict = decide({"passed": False, "reason": "缺少可比对的评测行", "rows": []}, None, regression_required=True)
    assert verdict["approved"] is False
    assert any("评测行" in reason for reason in verdict["reasons"])


def test_decide_rejects_when_gate_failed():
    gate = {"passed": False, "reason": "以下指标劣化：human_handoff F1", "rows": [{"metric": "x"}]}
    verdict = decide(gate, {"passed": True, "comparison": []}, regression_required=True)
    assert verdict["approved"] is False
    assert "human_handoff" in verdict["reasons"][0]
    # 门禁已挂时不额外加"没跑回归"这种噪音理由
    assert len(verdict["reasons"]) == 1


def test_decide_rejects_without_regression():
    gate = {"passed": True, "rows": [{"metric": "x"}]}
    verdict = decide(gate, None, regression_required=True)
    assert verdict["approved"] is False
    assert "回归" in verdict["reasons"][0]


def test_decide_rejects_on_regression_red_items():
    gate = {"passed": True, "rows": [{"metric": "x"}]}
    regression = {
        "passed": False,
        "comparison": [
            {"label": "检索 recall@5", "status": "退化"},
            {"label": "判据通过率", "status": "持平"},
        ],
    }
    verdict = decide(gate, regression, regression_required=True)
    assert verdict["approved"] is False
    assert "检索 recall@5" in verdict["reasons"][0]


def test_decide_approves_when_both_green():
    gate = {"passed": True, "rows": [{"metric": "x"}]}
    regression = {"passed": True, "comparison": [{"label": "判据通过率", "status": "持平"}]}
    assert decide(gate, regression, regression_required=True)["approved"] is True


def test_regression_summary_counts_improved_and_failed():
    payload = {
        "comparison": [
            {"label": "检索 recall@5", "status": "改善", "baseline": 0.83, "current": 0.87, "delta": 0.03},
            {"label": "判据通过率", "status": "持平", "baseline": 1.0, "current": 1.0, "delta": 0.0},
            {"label": "目标解决率", "status": "退化", "baseline": 1.0, "current": 0.9, "delta": -0.1},
            {"label": "意图 F1", "status": "未测"},
        ]
    }
    summary = regression_summary(payload)
    assert summary["counts"] == {"改善": 1, "持平": 1, "退化": 1, "未测": 1}
    assert [row["label"] for row in summary["improved"]] == ["检索 recall@5"]
    assert summary["failed"] == ["目标解决率"]


# ---------------- 决策卡 ----------------


def test_render_card_rejection_lists_reasons_and_rollback():
    gate = {
        "passed": False,
        "reason": "以下指标劣化：human_handoff F1",
        "rows": [
            {"metric": "macro-F1", "baseline": 0.7366, "candidate": 0.7085, "delta": -0.0281, "ok": False},
            {"metric": "human_handoff F1", "baseline": 0.6176, "candidate": 0.3111, "delta": -0.3065, "ok": False},
        ],
    }
    card = render_card(
        candidate=Path("data/intent/models/v2"),
        baseline=Path("data/intent/models/v1"),
        gate=gate,
        regression=None,
        regression_source="none",
        decision={"approved": False, "reasons": ["意图门禁不通过"]},
        active=None,
        dry_run=False,
    )
    assert "❌ 拒绝" in card and "保留现役版本" in card
    assert "human_handoff F1" in card and "-0.3065" in card
    assert "release.py --rollback" in card
    assert "没有跑全链路回归" in card


def test_render_card_approved_with_reused_regression_notes_source():
    gate = {"passed": True, "reason": "全部门禁指标不劣化", "rows": [
        {"metric": "macro-F1", "baseline": 0.7366, "candidate": 0.75, "delta": 0.0134, "ok": True}
    ]}
    regression = {
        "created_at": "2026-10-19 21:30:00",
        "passed": True,
        "comparison": [{"label": "判据通过率", "status": "持平"}],
    }
    card = render_card(
        candidate=Path("data/intent/models/v3"),
        baseline=Path("data/intent/models/v1"),
        gate=gate,
        regression=regression,
        regression_source="reused",
        decision={"approved": True, "reasons": []},
        active={"version": "v3", "adapter_dir": "data/intent/models/v3", "activated_at": "2026-10-19 22:00:00"},
        dry_run=False,
    )
    assert "✅ 放行" in card and "v3" in card
    assert "沿用 2026-10-19 21:30:00 的回归结果" in card   # 复用旧证据必须写明


# ---------------- active 版本 ----------------


def test_set_active_roundtrip_and_resolution(tmp_path):
    adapter = tmp_path / "models" / "v3"
    adapter.mkdir(parents=True)
    path = tmp_path / "active.json"
    payload = active_module.set_active("v3", adapter, evidence={"gate": {"passed": True}}, path=path)
    assert payload["version"] == "v3" and payload["evidence"]["gate"]["passed"] is True

    loaded = active_module.load_active(path)
    # 不在 AGENT_ROOT 下就存绝对路径（存的是 posix 写法，比较时统一成 Path）
    assert Path(loaded["adapter_dir"]) == adapter.resolve()
    assert active_module.active_adapter_dir(path) == adapter.resolve()

    adapter.rename(tmp_path / "models" / "v3-moved")            # 目录没了就不该再指向它
    assert active_module.active_adapter_dir(path) is None
    assert active_module.clear_active(path) is True
    assert active_module.load_active(path) is None


def test_resolve_adapter_dir_prefers_env_over_active(tmp_path, monkeypatch):
    from agent.intent.local_model import resolve_adapter_dir

    adapter = tmp_path / "models" / "v3"
    adapter.mkdir(parents=True)
    path = tmp_path / "active.json"
    active_module.set_active("v3", adapter, path=path)
    monkeypatch.setattr(active_module, "ACTIVE_PATH", path)
    monkeypatch.delenv("INTENT_LOCAL_DIR", raising=False)

    assert resolve_adapter_dir() == adapter.resolve()            # 没 env -> 用已发布版本
    monkeypatch.setenv("INTENT_LOCAL_DIR", str(tmp_path / "models" / "v9"))
    assert str(resolve_adapter_dir()).endswith("v9")             # env 更高优先（实验可覆盖）


# ---------------- CLI 编排 ----------------


def _prepare(tmp_path, *, candidate_f1: float, candidate_classes: float):
    models = tmp_path / "models"
    (models / "v1").mkdir(parents=True)
    candidate = models / "v3"
    candidate.mkdir(parents=True)
    (candidate / "adapter_config.json").write_text("{}", encoding="utf-8")

    baseline_json = tmp_path / "intent_baseline.json"
    baseline_json.write_text(
        json.dumps(
            {
                "backends": [
                    make_row(str(models / "v1"), 0.7366, good_classes(0.60)),
                    make_row(str(candidate), candidate_f1, good_classes(candidate_classes)),
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    regression_json = tmp_path / "regression.json"
    regression_json.write_text(
        json.dumps({"created_at": "2026-10-19 21:00:00", "passed": True,
                    "comparison": [{"label": "判据通过率", "status": "持平"}]},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    return models, candidate, baseline_json, regression_json


def _argv(tmp_path, baseline_json, regression_json, active_file, *extra):
    models = tmp_path / "models"
    return [
        "release.py",
        "--candidate", str(models / "v3"),
        "--baseline", str(models / "v1"),
        "--skip-eval",
        "--intent-baseline", str(baseline_json),
        "--regression-json", str(regression_json),
        "--active-file", str(active_file),
        "--report", str(tmp_path / "release.md"),
        "--json", str(tmp_path / "release.json"),
        *extra,
    ]


def test_cli_rejects_when_gate_fails_and_keeps_active(tmp_path, monkeypatch):
    """负样本演练：候选劣化 -> 退出码 1，active.json 不写。"""
    _, _, baseline_json, regression_json = _prepare(
        tmp_path, candidate_f1=0.70, candidate_classes=0.30
    )
    active_file = tmp_path / "active.json"
    monkeypatch.setattr(
        sys, "argv", _argv(tmp_path, baseline_json, regression_json, active_file)
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert main() == 1
    assert not active_file.exists()
    assert "拒绝放行" in buffer.getvalue()


def test_cli_dry_run_approves_without_switching(tmp_path, monkeypatch):
    _, _, baseline_json, regression_json = _prepare(
        tmp_path, candidate_f1=0.75, candidate_classes=0.65
    )
    active_file = tmp_path / "active.json"
    monkeypatch.setattr(
        sys, "argv", _argv(tmp_path, baseline_json, regression_json, active_file, "--dry-run")
    )
    with contextlib.redirect_stdout(io.StringIO()):
        assert main() == 0
    assert not active_file.exists()                      # dry-run 不落 active
    payload = json.loads((tmp_path / "release.json").read_text(encoding="utf-8"))
    assert payload["decision"]["approved"] is True and payload["dry_run"] is True


def test_cli_switches_active_when_gate_and_regression_pass(tmp_path, monkeypatch):
    _, candidate, baseline_json, regression_json = _prepare(
        tmp_path, candidate_f1=0.75, candidate_classes=0.65
    )
    active_file = tmp_path / "active.json"
    monkeypatch.setattr(
        sys, "argv", _argv(tmp_path, baseline_json, regression_json, active_file)
    )
    with contextlib.redirect_stdout(io.StringIO()):
        assert main() == 0

    activated = json.loads(active_file.read_text(encoding="utf-8"))
    assert activated["version"] == "v3"
    assert activated["evidence"]["gate"]["passed"] is True
    assert activated["evidence"]["regression"]["counts"] == {"持平": 1}
    assert (tmp_path / "release.md").exists()


def test_cli_refuses_when_regression_result_missing(tmp_path, monkeypatch):
    _, _, baseline_json, regression_json = _prepare(
        tmp_path, candidate_f1=0.75, candidate_classes=0.65
    )
    regression_json.write_text("", encoding="utf-8")       # 没有可复用的回归结果
    active_file = tmp_path / "active.json"
    monkeypatch.setattr(
        sys, "argv", _argv(tmp_path, baseline_json, regression_json, active_file)
    )
    with contextlib.redirect_stdout(io.StringIO()):
        assert main() == 1
    assert not active_file.exists()


def test_cli_returns_2_when_adapter_missing(tmp_path, monkeypatch):
    _, candidate, baseline_json, regression_json = _prepare(
        tmp_path, candidate_f1=0.75, candidate_classes=0.65
    )
    (candidate / "adapter_config.json").unlink()
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(tmp_path, baseline_json, regression_json, tmp_path / "active.json"),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        assert main() == 2


def test_cli_rollback_removes_active_file(tmp_path, monkeypatch):
    active_file = tmp_path / "active.json"
    active_file.write_text(json.dumps({"version": "v3", "adapter_dir": "x"}), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["release.py", "--rollback", "--active-file", str(active_file)]
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert main() == 0
    assert not active_file.exists()
    assert "已回滚" in buffer.getvalue()
