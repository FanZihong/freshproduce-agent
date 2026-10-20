"""从 checkpoint 里补出训练报告（进程被中断时的救援工具）。

典型场景：训练已经跑完，但保存 adapter / 写报告那一步之前进程被杀了
（WSL 断连、关终端、休眠都可能）。此时 `checkpoints/checkpoint-N/` 里其实躺着完整的
adapter 和 `trainer_state.json`（含全部 loss 记录），不用重训，跑一下这个脚本就能补齐：

- 把 checkpoint 里的 adapter 复制到运行目录（推理侧默认找这里）；
- 重新生成 `loss_curve.md` / `training_log.jsonl` / `train_summary.json`；
- 写 `train_config.json`，保证推理与训练的提示词一致。

用法（在 agent/ 目录下执行）：
    python scripts/export_train_report.py --run-dir data/intent/models/v1
    python scripts/export_train_report.py --run-dir data/intent/models/v1 --max-len 384
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from scripts.train_intent import write_outputs  # noqa: E402

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "README.md")


def find_checkpoint(run_dir: Path, explicit: str | None = None) -> Path:
    """定位要恢复的 checkpoint：显式指定 > checkpoints 下步数最大的那个 > run_dir 自己。"""
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = run_dir / explicit
        if not (path / "trainer_state.json").exists():
            raise SystemExit(f"{path} 里没有 trainer_state.json，无法恢复报告")
        return path

    candidates = sorted(
        (run_dir / "checkpoints").glob("checkpoint-*"),
        key=lambda item: int(item.name.split("-")[-1]) if item.name.split("-")[-1].isdigit() else 0,
    )
    for candidate in reversed(candidates):
        if (candidate / "trainer_state.json").exists():
            return candidate
    if (run_dir / "trainer_state.json").exists():
        return run_dir
    raise SystemExit(f"{run_dir} 下没找到 checkpoint（先确认训练至少保存过一次）")


def export_report(
    run_dir: Path,
    checkpoint: str | None = None,
    max_len: int = 384,
    prompt_detail: str = "minimal",
    label_format: str = "key",
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> dict:
    """把 checkpoint 里的训练状态还原成报告 + 可用的 adapter。"""
    run_dir = Path(run_dir)
    source = find_checkpoint(run_dir, checkpoint)
    state = json.loads((source / "trainer_state.json").read_text(encoding="utf-8"))
    history = list(state.get("log_history", []))
    losses = [entry["loss"] for entry in history if "loss" in entry]
    eval_losses = [entry["eval_loss"] for entry in history if "eval_loss" in entry]
    accuracies = [
        entry["eval_answer_token_accuracy"]
        for entry in history
        if "eval_answer_token_accuracy" in entry
    ]

    run_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ADAPTER_FILES:
        src = source / name
        dst = run_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(name)

    config = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "recovered_from": str(source),
        "base_model": base_model,
        "train_file": str(AGENT_ROOT / "data" / "intent" / "dataset_v1_balanced" / "train_sft.jsonl"),
        "eval_file": None,
        "prompt_detail": prompt_detail,
        "label_format": label_format,
        "max_len": max_len,
        "lora": {"note": "超参见 checkpoint 里的 training_args.bin"},
        "lr": None,
        "epochs": state.get("epoch"),
        "batch_size": None,
        "grad_accum": None,
        "class_weights_mode": "sample",
        "device": "cpu",
        "dtype": "fp32",
        "seed": None,
    }
    summary = {
        "recovered": True,
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "train_samples": None,
        "eval_samples": None,
        "steps": state.get("global_step"),
        "final_train_loss": round(losses[-1], 4) if losses else None,
        "first_train_loss": round(losses[0], 4) if losses else None,
        "best_train_loss": round(min(losses), 4) if losses else None,
        "loss_points": len(losses),
        "eval_loss": round(eval_losses[-1], 4) if eval_losses else None,
        "eval_loss_history": eval_losses,
        "eval_answer_token_accuracy_history": accuracies,
        "adapter_files": copied,
        "note": "训练收尾阶段进程被中断，报告由 trainer_state.json 还原；验证 loss 可能缺失",
    }
    write_outputs(_Namespace(run_dir=run_dir), run_dir, config, history, summary)
    return {"source": str(source), "copied": copied, "summary": summary}


class _Namespace:
    """write_outputs 只用到 out_dir 这一个属性。"""

    def __init__(self, run_dir: Path) -> None:
        self.out_dir = run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="从 checkpoint 恢复训练报告")
    parser.add_argument("--run-dir", type=Path, required=True, help="例如 data/intent/models/v1")
    parser.add_argument("--checkpoint", default=None, help="指定 checkpoint 目录（默认取步数最大的）")
    parser.add_argument("--max-len", type=int, default=384, help="训练时实际使用的 max_len")
    parser.add_argument("--prompt-detail", default="minimal")
    parser.add_argument("--label-format", default="key")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    result = export_report(
        args.run_dir,
        checkpoint=args.checkpoint,
        max_len=args.max_len,
        prompt_detail=args.prompt_detail,
        label_format=args.label_format,
        base_model=args.base_model,
    )
    summary = result["summary"]
    print(f"\n来源 checkpoint：{result['source']}")
    print(f"补回的 adapter 文件：{result['copied'] or '（运行目录里已存在）'}")
    print(
        f"loss {summary['first_train_loss']} → {summary['final_train_loss']}"
        f"（最低 {summary['best_train_loss']}，{summary['loss_points']} 个记录点）"
    )
    print(f"adapter 就绪：{args.run_dir}")
    print("\n下一步（三方对比）：")
    print("  python scripts/eval_intent.py --backends rule fewshot local --splits test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
