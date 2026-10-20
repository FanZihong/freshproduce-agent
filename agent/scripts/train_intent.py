"""Qwen2.5 + LoRA 微调意图识别模型。

分三步走，先体检再训练，避免"跑了 40 分钟才发现答案被截断"：

    # 0) 先下底座（国内直连 HF 会断，走镜像）
    export HF_ENDPOINT=https://hf-mirror.com
    python scripts/train_intent.py --download-only

    # 1) 数据体检（不加载 torch，几十秒出结果：长度分布、截断风险、类别分布）
    python scripts/train_intent.py --dry-run

    # 2) 小样本验链路（300 条 x 1 epoch，确认能跑通、能存盘）
    python scripts/train_intent.py --limit 300 --epochs 1 --out-dir data/intent/models/smoke

    # 3) 正式训练
    python scripts/train_intent.py

关键设计：
- **只对答案算 loss**：prompt 部分 label 全部置 -100，小模型才不会去学"复述提示词"；
- **训练/推理提示词一致性**：prompt_detail 与 target 风格写进 train_config.json，
  后续 local_model.py 读同一份配置，不靠人记；
- **长度先算后训**：超长默认直接报错（--on-too-long error），而不是悄悄截断把答案切掉；
- CPU 也能跑：--device auto，bf16 交给 GPU，CPU 走 fp32；batch 8 拆成 2x4 效果等价。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DATA_DIR  # noqa: E402
from agent.intent.baseline import build_messages, build_training_system_prompt  # noqa: E402
from agent.intent.taxonomy import load_taxonomy  # noqa: E402

DATASET_DIRS: tuple[Path, ...] = (
    DATA_DIR / "intent" / "dataset_v1_balanced",
    DATA_DIR / "intent" / "dataset_v1",
)
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUT_DIR = DATA_DIR / "intent" / "models" / "v1"
DEFAULT_REPORT = AGENT_ROOT / "eval" / "reports" / "intent_finetune_report.md"
PROMPT_DETAILS = ("keep", "minimal", "compact", "catalog")
SPARK = "▁▂▃▄▅▆▇█"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_dataset_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in DATASET_DIRS:
        if (candidate / "train_sft.jsonl").exists():
            return candidate
    raise SystemExit(
        f"找不到 SFT 数据（预期 {DATASET_DIRS[0] / 'train_sft.jsonl'}），"
        "先跑 python scripts/prepare_sft_data.py"
    )


def approx_tokens(text: str) -> int:
    """没装 tokenizers 时的估算：中日韩字符≈1 token，其余≈0.35 token。"""
    wide = sum(1 for char in text if "\u4e00" <= char <= "\u9fff" or "\u3000" <= char <= "\u303f")
    return int(wide + (len(text) - wide) * 0.35) + 1


def count_tokens(tokenizer, text: str) -> int:
    if tokenizer is None:
        return approx_tokens(text)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def prompt_for_row(row: dict, detail: str, taxonomy, style: str, max_labels: int) -> str | None:
    """detail == keep 时沿用数据里的 system 提示词，其余按指定详略重建。"""
    if detail == "keep":
        return None
    return build_training_system_prompt(
        taxonomy, target_style=style, max_labels=max_labels, detail=detail
    )


def render_texts(row: dict, system_prompt: str | None, taxonomy, max_labels: int) -> tuple[str, str]:
    """返回 (prompt 文本, 完整文本)：prompt 是"模型要读的部分"，完整文本多一段答案。"""
    system = system_prompt or (row["messages"][0]["content"] if row.get("messages") else "")
    messages = build_messages(
        row.get("text", ""),
        row.get("context"),
        taxonomy=taxonomy,
        system_prompt=system,
        output_hint="请输出意图标签：",
    )
    prompt_text = "\n".join(message["content"] for message in messages)
    answer = row.get("target") or ""
    return prompt_text, f"{prompt_text}\n{answer}"


def percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(1, int(len(ordered) * ratio + 0.9999))
    return ordered[min(index, len(ordered)) - 1]


def round_up(value: int, unit: int = 64) -> int:
    return int((value + unit - 1) // unit * unit)


def auto_max_len(p99_tokens: int, cap: int = 1024) -> int:
    """按实测长度定 max_len：P99 + 32 余量，向上取整到 64，下限 256、上限 cap。

    计划里写的 256 是"短指令 + 短答案"的配置；我们的提示词带 10 类定义，实测 P99 在
    600 token 上下，硬压到 256 会把类别定义整段截掉——那是模型学标签空间的依据。
    """
    return min(cap, max(256, round_up(p99_tokens + 32)))


def length_report(
    rows: list[dict], tokenizer, taxonomy, max_labels: int, max_len: int, styles: tuple[str, ...]
) -> list[dict]:
    """对每种提示词详略算长度分布：这是"选哪个 detail"的数据依据。"""
    report: list[dict] = []
    for detail in PROMPT_DETAILS:
        totals: list[int] = []
        prompts: list[int] = []
        over = 0
        for row in rows:
            system = prompt_for_row(row, detail, taxonomy, styles[0], max_labels)
            prompt_text, full_text = render_texts(row, system, taxonomy, max_labels)
            prompt_tokens = count_tokens(tokenizer, prompt_text)
            total = count_tokens(tokenizer, full_text) + 4  # 预留 assistant 头尾 token
            prompts.append(prompt_tokens)
            totals.append(total)
            if total > max_len:
                over += 1
        report.append(
            {
                "detail": detail,
                "prompt_p50": percentile(prompts, 0.5),
                "prompt_p95": percentile(prompts, 0.95),
                "total_p50": percentile(totals, 0.5),
                "total_p95": percentile(totals, 0.95),
                "total_p99": percentile(totals, 0.99),
                "total_max": max(totals) if totals else 0,
                "over_max_len": over,
                "over_ratio": round(over / len(totals), 4) if totals else 0.0,
            }
        )
    return report


def print_length_report(report: list[dict], max_len: int, tokenizer_available: bool) -> None:
    source = "真实 tokenizer" if tokenizer_available else "字符估算（未装 transformers）"
    limit = f"max_len={max_len}" if max_len else "max_len=自动"
    print(f"长度体检（{source}，{limit}）：")
    header = f"  {'提示词详略':<12}{'prompt P50':<12}{'prompt P95':<12}{'总长 P50':<10}{'总长 P95':<10}{'总长 P99':<10}{'最长':<7}{'超长条数':<10}"
    print(header)
    for item in report:
        print(
            f"  {item['detail']:<12}{item['prompt_p50']:<12}{item['prompt_p95']:<12}"
            f"{item['total_p50']:<10}{item['total_p95']:<10}{item['total_p99']:<10}"
            f"{item['total_max']:<7}{item['over_max_len']:<10}"
        )
    if not max_len:
        print("\n  自动 max_len 会按 P99 + 32 余量取整到 64：")
        for item in report:
            print(f"    {item['detail']:<10} -> {auto_max_len(item['total_p99'])}")
        return

    safe = [item for item in report if item["over_max_len"] == 0]
    if safe:
        print("\n  建议：" + "、".join(item["detail"] for item in safe)
              + f" 都能塞进 max_len={max_len}（答案不会被截断）")
    else:
        best = min(report, key=lambda item: item["total_p99"])
        print(f"\n  警告：max_len={max_len} 太小，没有一种详略能全部塞进去。")
        print(f"        改用 --max-len {auto_max_len(best['total_p99'])}（自动口径），"
              "或者把提示词压到更短再训。截断会把类别定义整段切掉，模型学不到标签空间。")


def build_optimizer_summary(args: argparse.Namespace, steps: int) -> list[str]:
    return [
        f"底座模型：{args.base_model}",
        f"LoRA：r={args.r} alpha={args.alpha} dropout={args.dropout}",
        f"target_modules：{args.target_modules}",
        f"超参：lr={args.lr} epochs={args.epochs} batch={args.batch_size} "
        f"grad_accum={args.grad_accum}（有效 batch={args.batch_size * args.grad_accum}）",
        f"max_len={args.max_len or '自动测量'}；优化步数约 {steps} 步",
        f"提示词详略：{args.prompt_detail}；目标串风格：{args.label_format}",
        "第一次 logging 之后就能用实测 it/s 外推总时长（脚本会在日志里打印）",
    ]


def download_base_model(args: argparse.Namespace) -> None:
    """把底座权重拉到本地缓存，避免训练到一半断网。"""
    if not os.environ.get("HF_ENDPOINT"):
        print("[提示] 没设 HF_ENDPOINT，国内直连 huggingface.co 大概率超时。建议：")
        print("       export HF_ENDPOINT=https://hf-mirror.com")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("缺 huggingface_hub（transformers 的依赖），先 pip install transformers")

    print(f"开始下载/校验：{args.base_model}（缓存目录 {args.cache_dir or '默认 HF 缓存'}）")
    started = time.time()
    path = snapshot_download(
        repo_id=args.base_model,
        cache_dir=args.cache_dir or None,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        max_workers=4,
    )
    print(f"完成，用时 {time.time() - started:.1f}s\n本地路径：{path}")
    print(f"\n后续训练可以用：--base-model {path}")


def _imports():
    """惰性导入重依赖，保证 --dry-run 在没装 torch 的机器上也能跑。"""
    try:
        import torch  # noqa: PLC0415
        from datasets import Dataset  # noqa: PLC0415
        from peft import LoraConfig, get_peft_model  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise SystemExit(
            f"缺少训练依赖（{exc.name}）。先执行：\n"
            "  pip install transformers datasets peft accelerate"
        )
    return torch, Dataset, LoraConfig, get_peft_model, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def pick_device_dtype(torch, requested: str) -> tuple[str, str]:
    if requested != "auto":
        device = requested
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    else:
        dtype = "fp32"  # CPU 上 bf16/fp16 训练又慢又不稳
    return device, dtype


def print_versions() -> None:
    """把关键库版本打进日志：换版本时的字段不兼容全靠这行定位。"""
    parts: list[str] = []
    for module_name in ("torch", "transformers", "peft", "datasets", "accelerate"):
        try:
            module = __import__(module_name)
            parts.append(f"{module_name} {getattr(module, '__version__', '?')}")
        except ImportError:
            parts.append(f"{module_name} 缺失")
    print("  " + " | ".join(parts))


def resolve_batch(args: argparse.Namespace, device: str) -> None:
    """CPU 上把 batch 8 拆成 2x4：有效 batch 不变，但 logits 峰值内存降到 1/4。

    8×256 token 的 logits 在 fp32 下约 1.2GB（vocab 15 万），CPU 机器很容易被这一步卡死。
    """
    if args.batch_size is not None:  # 用户显式指定，尊重它
        args.grad_accum = args.grad_accum or 1
        return
    if args.grad_accum is not None:  # 只指定了累积步数，batch 用设备默认
        args.batch_size = 2 if device == "cpu" else 8
        return
    if device == "cpu":
        args.batch_size, args.grad_accum = 2, 4
        print("  已按 CPU 自动配比：--batch-size 2 --grad-accum 4（有效 batch 仍是 8，省内存）")
    else:
        args.batch_size, args.grad_accum = 8, 1


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    answer_tokens: int


def encode_example(
    tokenizer, row: dict, system_prompt: str | None, max_len: int, on_too_long: str
) -> EncodedExample:
    """把一条样本编码成 input_ids + labels（prompt 段全部 -100）。

    用小模型的 chat template 生成完整序列，再定位 assistant 段起点；
    这样可以避免手写特殊 token 造成的错位。
    """
    messages = row["messages"]
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}, *messages[1:]]

    # return_dict=False 必须显式传：v4 默认 False、v5 默认 True，不传就拿不到 list[int]
    full_ids = to_id_list(
        tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, add_generation_prompt=False
        )
    )
    header_ids = to_id_list(tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))
    start = _last_sublist_index(full_ids, header_ids)
    if start == -1:  # 兜底：按 prompt 前缀长度对齐
        prompt_ids = to_id_list(
            tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, return_dict=False, add_generation_prompt=True
            )
        )
        start = len(prompt_ids) if prompt_ids == full_ids[: len(prompt_ids)] else max(0, len(full_ids) - 16)
    answer_start = start + len(header_ids)

    if len(full_ids) > max_len:
        if on_too_long == "error":
            raise SystemExit(
                f"样本超长（{len(full_ids)} > {max_len}），先别急着训：\n"
                f"  - 用更短的提示词详略：--prompt-detail minimal（推荐，长度体检里能看到差多少）\n"
                f"  - 或者 --max-len {max_len * 2}（CPU 会慢一倍左右）\n"
                f"  - 想强行裁剪（会损失提示词内容但保住答案）：--on-too-long trim-system"
            )
        # trim-system：从系统提示词里裁掉超出的 token，保住用户句和答案
        overflow = len(full_ids) - max_len
        keep_prompt = max(0, answer_start - overflow)
        if keep_prompt <= 0:
            raise SystemExit(f"样本连答案都放不下（{len(full_ids)} > {max_len}）：{row.get('text')[:30]}")
        system_ids = tokenizer.encode(messages[0]["content"], add_special_tokens=False)
        trimmed = system_ids[: max(8, len(system_ids) - overflow)]
        messages = [{"role": "system", "content": tokenizer.decode(trimmed)}, *messages[1:]]
        full_ids = to_id_list(
            tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=False, add_generation_prompt=False
            )
        )
        start = _last_sublist_index(full_ids, header_ids)
        answer_start = (start if start != -1 else len(full_ids) - 16) + len(header_ids)

    labels = [-100] * answer_start + full_ids[answer_start:]
    return EncodedExample(
        input_ids=full_ids,
        labels=labels,
        answer_tokens=len(full_ids) - answer_start,
    )


def _last_sublist_index(haystack: list[int], needle: list[int]) -> int:
    for index in range(len(haystack) - len(needle), -1, -1):
        if haystack[index : index + len(needle)] == needle:
            return index
    return -1


def to_id_list(value) -> list[int]:
    """把 tokenizer 可能返回的各种形态统一成 list[int]。

    坑就在这儿：transformers v5 把 `apply_chat_template` 的 `return_dict` 默认值从 False
    改成了 True，同样一行代码在 v4 返回 list[int]、在 v5 返回 BatchEncoding；
    如果不显式指定，值会一路混进 dataset 再在 collator 里炸成
    "too many dimensions 'str'"。
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):  # 带 batch 维，取第一条
            return to_id_list(value[0])
        if value and isinstance(value[0], str):
            raise SystemExit(
                "tokenizer 返回的是字符串 token 而不是 id（多数是 apply_chat_template 的 "
                f"return_dict 默认值变了）：{list(value)[:5]}... 请在调用处显式传 return_dict=False"
            )
        return [int(item) for item in value]
    if isinstance(value, dict) and "input_ids" in value:  # BatchEncoding
        return to_id_list(value["input_ids"])
    if hasattr(value, "input_ids"):
        return to_id_list(value.input_ids)
    if hasattr(value, "tolist"):
        return to_id_list(value.tolist())
    raise SystemExit(f"无法识别的 tokenizer 返回类型：{type(value)}")


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return SPARK[0] * len(values)
    return "".join(SPARK[min(len(SPARK) - 1, int((value - low) / (high - low) * (len(SPARK) - 1)))] for value in values)


def verify_encoded(dataset, limit: int = 3) -> None:
    """训练前抽样体检：类型/掩码不对就立刻停，别等 collator 抛出 torch 的怪错误。

    这一条专门用来拦"tokenizer 返回类型变了"这类坑：症状出现在 collator，
    根因却在 tokenize 那一步，隔了几百行很难查。
    """
    problems: list[str] = []
    for index in range(min(limit, len(dataset))):
        record = dataset[index]
        for field in ("input_ids", "attention_mask", "labels"):
            values = record.get(field)
            if not isinstance(values, list) or not all(isinstance(item, int) for item in values):
                problems.append(
                    f"样本 {index} 的 {field} 不是 int 列表（实际是 {type(values).__name__}）"
                )
                continue
        labels = record.get("labels")
        if isinstance(labels, list) and labels:
            masked = sum(1 for item in labels if item == -100)
            answer = len(labels) - masked
            if answer <= 0:
                problems.append(f"样本 {index} 的答案 token 数为 0，prompt 掩码把答案也盖住了")
            elif index == 0:
                print(
                    f"  样本体检：input_ids {len(record['input_ids'])} token，"
                    f"prompt 段 {masked} 个 -100，答案 {answer} token"
                )
    if problems:
        raise SystemExit("编码结果不合预期，先别训练：\n  - " + "\n  - ".join(problems))


def grad_accum_loss_scale(grad_accum_steps: int, already_scaled: bool) -> float:
    """自定义 loss 的梯度累积系数。

    transformers v5 引入了 `loss_is_scaled_for_ga`：如果 Trainer 认为 loss 已经按梯度累积
    归一化过（例如模型自己吃了 num_items_in_batch），它就不再除以累积步数。我们的加权 loss
    是自己算的 micro-batch 均值，所以要在这里补上除法——否则 CPU 的 2×4 配比会有 4 倍
    等效学习率，和 GPU 的 8×1 跑出来不是一个东西。
    """
    if grad_accum_steps > 1 and already_scaled:
        return 1.0 / grad_accum_steps
    return 1.0


def resolve_weight_mode(mode: str, has_weight_column: bool) -> bool:
    """决定这次训练要不要用样本加权。

    放在 main 里算（纯逻辑、不依赖 torch），run_training 只读结果——
    否则"用到时才定义"很容易写出引用未赋值变量的函数。
    """
    if mode == "sample":
        return bool(has_weight_column)
    return False


def save_checkpoint_policy(save_steps: int, save_epochs: bool) -> tuple[str, int]:
    """checkpoint 落盘策略。

    CPU 上 1 epoch 要两小时，而 `save_strategy="epoch"` 在 1 epoch 时等于"只在最后存一次"——
    中途断电就全没了。所以长跑必须能按步数存（--save-steps 100 约等于每 35 分钟一次）。
    """
    if save_steps > 0:
        return "steps", save_steps
    if save_epochs:
        return "epoch", 0
    return "no", 0


def accepted_training_arguments(training_arguments_cls) -> set[str]:
    """列出当前 transformers 的 TrainingArguments 真正接受的参数名。

    v5 删掉了 overwrite_output_dir / warmup_ratio，v4.41 之前叫 evaluation_strategy——
    写死参数名会在换版本时直接崩，所以这里按实际签名判断。
    """
    import dataclasses
    import inspect

    try:
        accepted = set(inspect.signature(training_arguments_cls.__init__).parameters)
    except (TypeError, ValueError):  # pragma: no cover - 极端情况
        accepted = set()
    if not accepted and dataclasses.is_dataclass(training_arguments_cls):
        accepted = {field.name for field in dataclasses.fields(training_arguments_cls) if field.init}
    return accepted


def make_training_arguments(training_arguments_cls, arguments: dict):
    """跨版本构造 TrainingArguments：先做字段改名，再丢掉不认识的参数。"""
    accepted = accepted_training_arguments(training_arguments_cls)
    if not accepted or "kwargs" in accepted:
        # 内省不出来（或它本来就吃 **kwargs），交给它自己判断
        return training_arguments_cls(**arguments)
    if "eval_strategy" not in accepted and "evaluation_strategy" in accepted:
        arguments["evaluation_strategy"] = arguments.pop("eval_strategy", "no")
    dropped = sorted(key for key in arguments if key not in accepted)
    if dropped:
        print(f"  [提示] 当前版本不认识这些 TrainingArguments 参数，已忽略：{', '.join(dropped)}")
        arguments = {key: value for key, value in arguments.items() if key in accepted}
    return training_arguments_cls(**arguments)


def write_outputs(
    args: argparse.Namespace,
    out_dir: Path,
    config: dict,
    log_history: list[dict],
    summary: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "training_log.jsonl").write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in log_history) + "\n",
        encoding="utf-8",
    )

    train_points = [entry["loss"] for entry in log_history if "loss" in entry]
    eval_points = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]
    accuracy_points = [
        entry["eval_answer_token_accuracy"]
        for entry in log_history
        if "eval_answer_token_accuracy" in entry
    ]
    lines = [
        "# 训练曲线（loss）",
        "",
        f"- 训练 loss（{len(train_points)} 个记录点）：`{sparkline(train_points)}`",
        f"- 首点 {train_points[0]:.4f} → 末点 {train_points[-1]:.4f}"
        + (f"，最低 {min(train_points):.4f}" if train_points else ""),
    ]
    if eval_points:
        lines.append(f"- 验证 loss（{len(eval_points)} 个 epoch）：`{sparkline(eval_points)}` → {eval_points}")
    if accuracy_points:
        lines.append(
            f"- 答案 token 准确率（逐 epoch）：`{sparkline(accuracy_points)}` → "
            + "、".join(f"{value:.4f}" for value in accuracy_points)
        )
    lines += ["", "## 采样点", "", "| step | epoch | loss | lr |", "| --- | --- | --- | --- |"]
    stride = max(1, len(train_points) // 20)
    index = 0
    for entry in log_history:
        if "loss" not in entry:
            continue
        if index % stride == 0:
            lines.append(
                f"| {entry.get('step', '')} | {round(entry.get('epoch', 0), 2)} | "
                f"{entry['loss']:.4f} | {entry.get('learning_rate', 0):.2e} |"
            )
        index += 1
    lines += [
        "",
        "怎么看：",
        "- 训练 loss 单调下降且验证 loss 不抬头 = 没跑偏；验证 loss 先降后升就是过拟合，减少 epoch。",
        "- loss 几乎不动：多半是学习率太小或 prompt 被截断（先跑 `--dry-run` 看长度体检）。",
        "- 0.5B 在这种短输出任务上，loss 通常从 3~5 降到 0.1 以下就该停了。",
        "",
    ]
    (out_dir / "loss_curve.md").write_text("\n".join(lines), encoding="utf-8")

    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# 意图识别 LoRA adapter（v1）",
                "",
                f"- 底座：{config['base_model']}",
                f"- 训练数据：{config['train_file']}（{summary['train_samples']} 条）",
                f"- 提示词详略：{config['prompt_detail']}；目标串风格：{config['label_format']}",
                f"- max_len={config['max_len']}；类别加权：{config['class_weights_mode']}",
                "",
                "## 推理时必须保持一致",
                "",
                "`local_model.py` 会用 `train_config.json` 里的 prompt_detail / label_format 重建同一段",
                "system prompt——训练与推理的提示词不一致是本项目最容易踩的坑。",
                "",
                "## 加载",
                "",
                "```python",
                "from peft import PeftModel",
                "from transformers import AutoModelForCausalLM, AutoTokenizer",
                "base = AutoModelForCausalLM.from_pretrained(train_config['base_model'])",
                "model = PeftModel.from_pretrained(base, 'data/intent/models/v1')",
                "```",
                "",
                "## 下一步",
                "",
                "```bash",
                "python scripts/eval_intent.py --backends rule fewshot local",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\n训练曲线：{out_dir / 'loss_curve.md'}")
    print(f"训练日志：{out_dir / 'training_log.jsonl'}")


def run_training(args: argparse.Namespace, rows: list[dict], eval_rows: list[dict], taxonomy) -> int:
    (torch, Dataset, LoraConfig, get_peft_model, AutoModelForCausalLM, AutoTokenizer, Trainer,
     TrainingArguments) = _imports()

    device, dtype = pick_device_dtype(torch, args.device)
    print(f"\n设备：{device}（{dtype}）")
    print_versions()
    resolve_batch(args, device)
    if device == "cpu":
        print("  提示：0.5B + LoRA 在 CPU 上 4 epoch 通常 30–60 分钟；"
              "内存吃紧就再降 --batch-size 1 并把 --grad-accum 提到 8")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir or None)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # max_len 默认自动测量：先看真实长度，再决定截断边界（--max-len 可手动覆盖）
    max_len = args.max_len
    policy = args.on_too_long
    if not max_len:
        step = max(1, len(rows) // 200)
        sample = rows[::step][:200]
        lengths = []
        for row in sample:
            system = prompt_for_row(row, args.prompt_detail, taxonomy, args.label_format, 2)
            _, full_text = render_texts(row, system, taxonomy, 2)
            lengths.append(count_tokens(tokenizer, full_text) + 4)
        p99 = percentile(lengths, 0.99)
        max_len = auto_max_len(p99)
        if policy == "auto":
            policy = "trim-system"  # 自动口径已经覆盖 P99，剩下的长尾裁提示词即可
        print(f"  max_len 自动测量：{len(sample)} 条样本 P99={p99} token -> max_len={max_len}")
    elif policy == "auto":
        policy = "error"
    print(f"  超长策略：{policy}"
          + ("（超长直接报错，避免静默截断）" if policy == "error" else "（裁剪系统提示词尾部，保住答案）"))

    torch_dtype = (
        torch.bfloat16
        if dtype == "bf16"
        else (torch.float16 if dtype == "fp16" else torch.float32)
    )
    try:
        # transformers 5.x 把 torch_dtype 改名成 dtype
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, cache_dir=args.cache_dir or None, dtype=torch_dtype
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, cache_dir=args.cache_dir or None, torch_dtype=torch_dtype
        )
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, lora)
    if args.grad_checkpoint:
        # 冻结底座 + 梯度检查点：不开这个开关，梯度根本回不到 LoRA 参数上
        model.enable_input_require_grads()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    weighted = args.use_sample_weights  # 由 main() 按数据里有没有 sample_weight 决定
    print(f"可训练参数：{trainable:,} / {total:,}（{trainable / total:.2%}）")
    print(
        f"loss 口径：{'样本加权 CE' if weighted else '标准 CE'}"
        f"（只在答案 token 上计算，梯度累积 {args.grad_accum} 步已做等效归一化）"
    )

    def tokenize(row: dict) -> dict:
        system = prompt_for_row(row, args.prompt_detail, taxonomy, args.label_format, 2)
        encoded = encode_example(tokenizer, row, system, max_len, policy)
        payload = {
            "input_ids": encoded.input_ids,
            "attention_mask": [1] * len(encoded.input_ids),
            "labels": encoded.labels,
        }
        if weighted:
            payload["sample_weight"] = float(row.get("sample_weight", 1.0))
        return payload

    raw_train = Dataset.from_list(rows)
    train_ds = raw_train.map(tokenize, remove_columns=raw_train.column_names)
    eval_ds = None
    if eval_rows:
        raw_eval = Dataset.from_list(eval_rows)
        eval_ds = raw_eval.map(tokenize, remove_columns=raw_eval.column_names)
    verify_encoded(train_ds)
    if eval_ds is not None:
        verify_encoded(eval_ds, limit=1)

    class SFTDataCollator:
        """动态 padding + 透传 sample_weight（默认 collator 会把权重丢掉）。"""

        def __call__(self, features):
            max_length = max(len(feature["input_ids"]) for feature in features)
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            weights = []
            for feature in features:
                pad = max_length - len(feature["input_ids"])
                batch["input_ids"].append(list(feature["input_ids"]) + [tokenizer.pad_token_id] * pad)
                batch["attention_mask"].append(list(feature["attention_mask"]) + [0] * pad)
                batch["labels"].append(list(feature["labels"]) + [-100] * pad)
                weights.append(float(feature.get("sample_weight", 1.0)))
            tensors = {
                key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()
            }
            if weighted:
                tensors["sample_weight"] = torch.tensor(weights, dtype=torch.float)
            return tensors

    class WeightedTrainer(Trainer):
        """按样本权重加权 CE：长尾类（complaint/short_weight）不会被高频类淹没。"""

        def _grad_accum_steps(self) -> int:
            value = getattr(self, "current_gradient_accumulation_steps", None)
            if isinstance(value, (int, float)) and value:
                return int(value)
            return int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weights = inputs.pop("sample_weight", None)
            outputs = model(**inputs)
            if weights is None:
                return (outputs.loss, outputs) if return_outputs else outputs.loss

            logits = outputs.logits
            labels = inputs["labels"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            losses = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.size())
            mask = (shift_labels != -100).float()
            per_sample = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            loss = (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)
            scale = grad_accum_loss_scale(
                self._grad_accum_steps(),
                bool(getattr(self, "loss_is_scaled_for_ga", False)),
            )
            if scale != 1.0:
                loss = loss * scale
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        mask = labels != -100
        if mask.sum() == 0:
            return {"answer_token_accuracy": 0.0}
        accuracy = float((predictions[mask] == labels[mask]).mean())
        return {"answer_token_accuracy": round(accuracy, 4)}

    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.grad_accum))
    total_steps = int(steps_per_epoch * args.epochs)
    print()
    for line in build_optimizer_summary(args, total_steps):
        print(f"  {line}")

    arguments = dict(
        output_dir=str(args.out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        # v5 删了 warmup_ratio，改用 warmup_steps（4.x 两个都有），按比例折算保证等价的暖启动
        warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        weight_decay=0.0,
        logging_steps=args.logging_steps,
        save_strategy=save_checkpoint_policy(args.save_steps, args.save_epochs)[0],
        save_steps=save_checkpoint_policy(args.save_steps, args.save_epochs)[1] or 500,
        save_total_limit=2,
        eval_strategy="epoch" if eval_ds is not None else "no",
        bf16=dtype == "bf16",
        fp16=dtype == "fp16",
        use_cpu=device == "cpu",
        dataloader_num_workers=0,
        remove_unused_columns=False,  # 保留 sample_weight，否则权重会被丢掉
        report_to=[],
        seed=args.seed,
        gradient_checkpointing=args.grad_checkpoint,
    )
    training_args = make_training_arguments(TrainingArguments, arguments)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=SFTDataCollator(),
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )

    started = time.time()
    print(f"\n开始训练：{len(train_ds)} 条样本，{total_steps} 步\n")
    resume = args.resume_from_checkpoint
    if resume == "latest":
        candidates = sorted((args.out_dir / "checkpoints").glob("checkpoint-*"))
        resume = str(candidates[-1]) if candidates else None
        if resume:
            print(f"  从最近一次 checkpoint 续训：{resume}")
    trainer.train(resume_from_checkpoint=resume or None)
    duration = time.time() - started

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.out_dir))
    # 双保险：显式存一次 adapter，避免某些版本把整个底座也塞进 checkpoints
    model.save_pretrained(str(args.out_dir))
    tokenizer.save_pretrained(str(args.out_dir))

    log_history = list(trainer.state.log_history)
    eval_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]
    train_losses = [entry["loss"] for entry in log_history if "loss" in entry]
    config = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": args.base_model,
        "train_file": str(args.train_file),
        "eval_file": str(args.eval_file) if args.eval_file else None,
        "prompt_detail": args.prompt_detail,
        "label_format": args.label_format,
        "max_len": max_len,
        "on_too_long": policy,
        "lora": {
            "r": args.r,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target_modules": args.target_modules,
        },
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "class_weights_mode": args.class_weights,
        "device": device,
        "dtype": dtype,
        "seed": args.seed,
    }
    summary = {
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds) if eval_ds is not None else 0,
        "steps": total_steps,
        "duration_seconds": round(duration, 1),
        "duration_minutes": round(duration / 60, 2),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": round(trainable / total, 6),
        "max_len": max_len,
        "final_train_loss": round(train_losses[-1], 4) if train_losses else None,
        "first_train_loss": round(train_losses[0], 4) if train_losses else None,
        "eval_loss": round(eval_losses[-1], 4) if eval_losses else None,
        "eval_loss_history": eval_losses,
        "eval_answer_token_accuracy_history": [
            entry["eval_answer_token_accuracy"]
            for entry in log_history
            if "eval_answer_token_accuracy" in entry
        ],
    }
    write_outputs(args, args.out_dir, config, log_history, summary)

    print(f"\n训练完成：{summary['duration_minutes']} 分钟，"
          f"loss {summary['first_train_loss']} → {summary['final_train_loss']}"
          + (f"，验证 loss {summary['eval_loss']}" if summary["eval_loss"] else ""))
    if summary["eval_answer_token_accuracy_history"]:
        print(f"答案 token 准确率（逐 epoch）：{summary['eval_answer_token_accuracy_history']}")
    print(f"adapter 已保存：{args.out_dir}")
    print("\n下一步：python scripts/eval_intent.py --backends rule fewshot local")
    return 0


def dry_run(args: argparse.Namespace, rows: list[dict], eval_rows: list[dict], taxonomy) -> int:
    """数据体检：不需要 torch，先把长度、分布、截断风险看清楚。"""
    if args.batch_size is None:
        args.batch_size, args.grad_accum = 8, 1
        print("[提示] --dry-run 不知道设备，按 GPU 的 8×1 估步数；"
              "CPU 上会自动改成 2×4（有效 batch 相同）")
    tokenizer = None
    tokenizer_available = False
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir or None)
        tokenizer_available = True
    except Exception as exc:  # noqa: BLE001 - 没装/没下载时退回字符估算
        print(f"[提示] 没拿到 tokenizer（{type(exc).__name__}: {exc}）")
        print("       下面用字符估算；`--download-only` 可以先拉底座，或先 export HF_ENDPOINT=https://hf-mirror.com")

    print("\n数据分布：")
    counts: dict[str, int] = {}
    multi = 0
    for row in rows:
        label = (row.get("labels") or ["unknown"])[0]
        counts[label] = counts.get(label, 0) + 1
        if len(row.get("labels") or []) > 1:
            multi += 1
    print("  " + "，".join(f"{key} {value}" for key, value in sorted(counts.items(), key=lambda item: -item[1])))
    print(f"  多标签样本 {multi} 条；加权字段 sample_weight "
          f"{'存在' if rows and 'sample_weight' in rows[0] else '缺失（--class-weights off）'}")

    report = length_report(rows, tokenizer, taxonomy, 2, args.max_len, PROMPT_DETAILS)
    print()
    print_length_report(report, args.max_len, tokenizer_available)

    steps_per_epoch = max(1, len(rows) // (args.batch_size * args.grad_accum))
    print()
    for line in build_optimizer_summary(args, steps_per_epoch * args.epochs):
        print(f"  {line}")

    if eval_rows:
        print(f"\n验证集：{len(eval_rows)} 条（--eval-limit {args.eval_limit}）；"
              f"每个 epoch 会算一次验证 loss 与答案 token 准确率")
    print("\n体检通过后：python scripts/train_intent.py --limit 300 --epochs 1   # 先验链路")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen2.5 + LoRA 微调意图识别")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--prompt-detail", choices=PROMPT_DETAILS, default="minimal")
    parser.add_argument("--label-format", choices=["key", "name", "json"], default="key")
    parser.add_argument(
        "--max-len",
        type=int,
        default=0,
        help="0=按实测长度自动（P99+32 取整到 64，下限 256）；计划里的 256 需手动指定",
    )
    parser.add_argument(
        "--on-too-long",
        choices=["auto", "error", "trim-system"],
        default="auto",
        help="auto：手动指定 max_len 时报错（避免静默截断答案），自动 max_len 时裁提示词尾部",
    )
    parser.add_argument("--class-weights", choices=["off", "sample"], default="sample")
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 条训练（冒烟用）")
    parser.add_argument("--eval-limit", type=int, default=200, help="验证集最多用多少条")
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="默认 CPU 上自动用 2（配 --grad-accum 4，有效 batch 仍是 8），GPU 上 8",
    )
    parser.add_argument("--grad-accum", type=int, default=None, help="默认随 --batch-size 自动配")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Qwen2 的注意力 + MLP 全挂上，效果比只挂 q,v 明显更好",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--grad-checkpoint", action="store_true", help="显存不够时开；CPU 上会更慢")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-epochs", action="store_true", help="每个 epoch 存一次 checkpoint")
    parser.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="每 N 步存一次 checkpoint（CPU 长跑建议 100，1 epoch 也能中途保命）",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="断点续训：填 checkpoints/checkpoint-N，或直接填 latest（CPU 长跑很实用）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只做数据体检，不加载 torch、不训练")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    if args.download_only:
        download_base_model(args)
        return 0

    dataset_dir = resolve_dataset_dir(args.data_dir)
    train_file = args.train_file or dataset_dir / "train_sft.jsonl"
    eval_file = args.eval_file or dataset_dir / "dev_sft.jsonl"
    args.train_file = train_file
    args.eval_file = eval_file if eval_file.exists() else None

    rows = load_jsonl(train_file)
    if not rows:
        raise SystemExit(f"训练数据为空：{train_file}（先跑 python scripts/prepare_sft_data.py）")
    if args.limit and args.limit < len(rows):
        # 系统抽样（每 stride 条取 1 条）而不是直接切头：小样本冒烟也要覆盖到所有类别
        stride = max(1, len(rows) // args.limit)
        rows = rows[::stride][: args.limit]
    # 加权 / 输出目录等"纯逻辑"决定都放在这里，run_training 只读结果
    has_weight_column = bool(rows) and "sample_weight" in rows[0]
    args.use_sample_weights = resolve_weight_mode(args.class_weights, has_weight_column)
    if args.class_weights == "sample" and not has_weight_column:
        print("[warn] 数据里没有 sample_weight 字段，已自动关闭类别加权"
              "（重新跑 prepare_sft_data.py 会带上该字段）")
    eval_rows = load_jsonl(eval_file)[: args.eval_limit] if args.eval_file else []
    taxonomy = load_taxonomy()

    print("=" * 78)
    print("Qwen2.5 + LoRA 意图识别微调")
    print("=" * 78)
    print(
        f"训练数据：{train_file}（{len(rows)} 条"
        + ("，系统抽样自 --limit" if args.limit else "")
        + "）"
    )
    print(f"验证数据：{eval_file if args.eval_file else '（无）'}（{len(eval_rows)} 条）")
    print(f"输出目录：{args.out_dir}")
    print(f"底座模型：{args.base_model}")

    if args.dry_run:
        return dry_run(args, rows, eval_rows, taxonomy)
    return run_training(args, rows, eval_rows, taxonomy)


if __name__ == "__main__":
    raise SystemExit(main())
