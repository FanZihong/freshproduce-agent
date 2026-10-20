"""训练报告中转站测试：从 checkpoint 还原曲线与 adapter 文件。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_train_report import export_report, find_checkpoint


def make_run_dir(tmp_path: Path, step: int = 327) -> Path:
    run_dir = tmp_path / "v1"
    checkpoint = run_dir / "checkpoints" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": step,
                "epoch": 1.0,
                "log_history": [
                    {"step": 10, "epoch": 0.03, "loss": 1.617},
                    {"step": 20, "epoch": 0.06, "loss": 1.332},
                    {"step": 327, "epoch": 1.0, "loss": 0.468},
                ],
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"fake-weights")
    return run_dir


def test_find_checkpoint_picks_largest_step(tmp_path):
    run_dir = make_run_dir(tmp_path, step=100)
    make_run_dir(tmp_path, step=327)
    found = find_checkpoint(run_dir)
    assert found.name == "checkpoint-327"


def test_find_checkpoint_raises_without_state(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        find_checkpoint(empty)


def test_export_report_restores_curve_and_adapter(tmp_path):
    run_dir = make_run_dir(tmp_path)
    result = export_report(run_dir, max_len=384)

    assert (run_dir / "adapter_config.json").exists()
    assert (run_dir / "adapter_model.safetensors").exists()
    assert set(result["copied"]) >= {"adapter_config.json", "adapter_model.safetensors"}

    for name in ("loss_curve.md", "training_log.jsonl", "train_config.json", "train_summary.json", "README.md"):
        assert (run_dir / name).exists(), name

    curve = (run_dir / "loss_curve.md").read_text(encoding="utf-8")
    assert "1.617" in curve and "0.468" in curve
    config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
    assert config["max_len"] == 384
    assert config["prompt_detail"] == "minimal"
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["recovered"] is True
    assert summary["best_train_loss"] == 0.468


def test_export_report_does_not_overwrite_existing_adapter(tmp_path):
    run_dir = make_run_dir(tmp_path)
    (run_dir / "adapter_model.safetensors").write_bytes(b"already-here")
    result = export_report(run_dir)
    assert "adapter_model.safetensors" not in result["copied"]
    assert (run_dir / "adapter_model.safetensors").read_bytes() == b"already-here"
