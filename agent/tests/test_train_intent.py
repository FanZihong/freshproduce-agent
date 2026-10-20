"""LoRA 训练脚本的纯逻辑测试：label 掩码、超长截断、长度体检、曲线渲染。

不加载 torch/transformers：用一个"按字符切 token"的假 tokenizer 验证最容易出错的
那一段（prompt 段必须全部 -100，否则小模型会去学复述提示词）。
"""

from __future__ import annotations

import argparse
import inspect
import json

import pytest

from agent.intent.taxonomy import load_taxonomy
from scripts.train_intent import (
    _last_sublist_index,
    accepted_training_arguments,
    auto_max_len,
    build_optimizer_summary,
    encode_example,
    length_report,
    make_training_arguments,
    percentile,
    render_texts,
    grad_accum_loss_scale,
    resolve_weight_mode,
    save_checkpoint_policy,
    sparkline,
    to_id_list,
    verify_encoded,
    write_outputs,
)
from scripts.train_intent import run_training


class FakeBatchEncoding(dict):
    """复刻 transformers v5 的 BatchEncoding：dict 形态，len 是字段数而不是 token 数。"""


class FakeTokenizer:
    """把每个字符映射成一个 token id 的假 tokenizer，行为与 Qwen chat 模板对齐。"""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
        self.eos_token = "<|im_end|>"
        self._to_id: dict[str, int] = {}

    def _id(self, char: str) -> int:
        if char not in self._to_id:
            self._to_id[char] = 100 + len(self._to_id)
        return self._to_id[char]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id(char) for char in text]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": self.encode(text, add_special_tokens)}

    def decode(self, ids) -> str:
        inverse = {value: key for key, value in self._to_id.items()}
        return "".join(inverse.get(int(item), "") for item in ids)

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_dict: bool = True,
    ):
        """v5 的语义：return_dict 默认 True，返回 BatchEncoding 而不是 list[int]。"""
        parts = [f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in messages]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        text = "".join(parts)
        if not tokenize:
            return text
        ids = self.encode(text)
        if return_dict:
            return FakeBatchEncoding(input_ids=ids, attention_mask=[1] * len(ids))
        return ids


def make_row(target: str = "主:invoice", text: str = "怎么开发票", system: str = "你是意图识别器。") -> dict:
    return {
        "id": "t1",
        "text": text,
        "context": None,
        "labels": ["invoice"],
        "target": target,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
            {"role": "assistant", "content": target},
        ],
    }


def test_encode_example_masks_every_prompt_token():
    tokenizer = FakeTokenizer()
    encoded = encode_example(tokenizer, make_row(), None, max_len=256, on_too_long="error")

    masked = [index for index, value in enumerate(encoded.labels) if value == -100]
    assert masked == list(range(len(masked)))          # prompt 段必须连续在最前面
    assert encoded.labels[len(masked) :] == encoded.input_ids[len(masked) :]
    answer_text = tokenizer.decode(encoded.labels[len(masked) :])
    assert answer_text.startswith("主:invoice")
    assert encoded.answer_tokens == len(encoded.labels) - len(masked)
    assert encoded.answer_tokens >= 5


def test_encode_example_overrides_system_prompt_when_given():
    tokenizer = FakeTokenizer()
    encoded = encode_example(tokenizer, make_row(), "新的系统提示词", max_len=256, on_too_long="error")
    prompt_ids = encoded.input_ids[: len(encoded.labels) - encoded.answer_tokens]
    assert "新的系统提示词" in tokenizer.decode(prompt_ids)
    assert "你是意图识别器。" not in tokenizer.decode(prompt_ids)


def test_encode_example_raises_when_too_long_by_default():
    tokenizer = FakeTokenizer()
    with pytest.raises(SystemExit):
        encode_example(tokenizer, make_row(), None, max_len=20, on_too_long="error")


def test_encode_example_trim_system_keeps_answer():
    tokenizer = FakeTokenizer()
    long_system = "系统提示词" * 60
    row = make_row(system=long_system)
    full = to_id_list(tokenizer.apply_chat_template(row["messages"], return_dict=False))
    trimmed = encode_example(tokenizer, row, None, max_len=len(full) - 40, on_too_long="trim-system")

    assert len(trimmed.input_ids) <= len(full) - 40 + 20  # 允许重新套模板的少量开销
    answer_text = tokenizer.decode(trimmed.labels[len(trimmed.labels) - trimmed.answer_tokens :])
    assert answer_text.startswith("主:invoice")           # 答案必须完整保留


def test_last_sublist_index_returns_last_match():
    assert _last_sublist_index([1, 2, 3, 1, 2], [1, 2]) == 3
    assert _last_sublist_index([1, 2, 3], [9]) == -1


def test_auto_max_len_rounds_and_clamps():
    assert auto_max_len(500) == 576      # 500+32=532 -> 取整到 576
    assert auto_max_len(100) == 256      # 下限
    assert auto_max_len(5000) == 1024    # 上限，防止显存/内存爆掉


def test_percentile_nearest_rank():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile(list(range(1, 101)), 0.99) == 99
    assert percentile([], 0.9) == 0


def test_sparkline_maps_range():
    line = sparkline([3.0, 2.0, 1.0])
    assert len(line) == 3
    assert line[0] != line[-1]           # 高 loss 与低 loss 的符号不同
    assert sparkline([1.0, 1.0]) == sparkline([1.0, 1.0])
    assert sparkline([]) == ""


def test_render_texts_keep_uses_data_system_prompt():
    taxonomy = load_taxonomy()
    row = make_row(system="数据里的系统提示词")
    prompt_text, full_text = render_texts(row, None, taxonomy, 2)
    assert "数据里的系统提示词" in prompt_text
    assert full_text.endswith("主:invoice")


def test_length_report_flags_over_max_len_for_short_limit():
    tokenizer = FakeTokenizer()
    rows = [make_row() for _ in range(10)]
    report = length_report(rows, tokenizer, load_taxonomy(), 2, max_len=30, styles=("key",))
    minimal = next(item for item in report if item["detail"] == "minimal")
    assert minimal["over_max_len"] == 10
    assert minimal["total_p50"] > 30


def test_build_optimizer_summary_reports_effective_batch():
    args = argparse.Namespace(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        r=16,
        alpha=32,
        dropout=0.05,
        target_modules="q_proj,v_proj",
        lr=2e-4,
        epochs=4,
        batch_size=2,
        grad_accum=4,
        max_len=0,
        prompt_detail="minimal",
        label_format="key",
    )
    summary = "\n".join(build_optimizer_summary(args, 1304))
    assert "有效 batch=8" in summary
    assert "自动测量" in summary
    assert "1304" in summary


def test_write_outputs_produces_curve_and_config(tmp_path):
    args = argparse.Namespace(out_dir=tmp_path)
    config = {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "train_file": "train_sft.jsonl",
        "prompt_detail": "minimal",
        "label_format": "key",
        "max_len": 512,
        "class_weights_mode": "sample",
    }
    log_history = [
        {"step": 10, "epoch": 0.1, "loss": 3.2, "learning_rate": 2e-4},
        {"step": 20, "epoch": 0.2, "loss": 1.1, "learning_rate": 1.8e-4},
        {"step": 30, "epoch": 0.3, "loss": 0.2, "learning_rate": 1.2e-4},
        {"step": 30, "epoch": 0.3, "eval_loss": 0.35, "eval_answer_token_accuracy": 0.9},
    ]
    summary = {"train_samples": 2614, "eval_samples": 200}
    write_outputs(args, tmp_path, config, log_history, summary)

    for name in ("loss_curve.md", "training_log.jsonl", "train_config.json", "train_summary.json", "README.md"):
        assert (tmp_path / name).exists(), name
    curve = (tmp_path / "loss_curve.md").read_text(encoding="utf-8")
    assert "3.2000" in curve and "0.2000" in curve
    assert "验证 loss" in curve
    assert json.loads((tmp_path / "train_config.json").read_text(encoding="utf-8"))["max_len"] == 512


def test_make_training_arguments_drops_removed_fields_in_v5(capsys):
    """transformers 5.x 删了 overwrite_output_dir / warmup_ratio，不能让脚本直接崩。"""
    import dataclasses

    @dataclasses.dataclass
    class ArgsV5:
        output_dir: str = ""
        warmup_steps: float = 0
        eval_strategy: str = "no"
        logging_steps: int = 10

    arguments = {
        "output_dir": "/tmp/out",
        "overwrite_output_dir": True,
        "warmup_ratio": 0.03,
        "warmup_steps": 3,
        "eval_strategy": "epoch",
        "logging_steps": 10,
    }
    built = make_training_arguments(ArgsV5, dict(arguments))

    assert built.output_dir == "/tmp/out"
    assert built.warmup_steps == 3
    assert built.eval_strategy == "epoch"
    warning = capsys.readouterr().out
    assert "overwrite_output_dir" in warning and "warmup_ratio" in warning


def test_make_training_arguments_renames_eval_strategy_for_old_versions():
    """transformers 4.41 之前只有 evaluation_strategy。"""

    class ArgsOld:
        def __init__(self, output_dir="", evaluation_strategy="no", learning_rate=5e-5):
            self.output_dir = output_dir
            self.evaluation_strategy = evaluation_strategy
            self.learning_rate = learning_rate

    built = make_training_arguments(
        ArgsOld, {"output_dir": "/tmp/old", "eval_strategy": "epoch", "learning_rate": 2e-4}
    )
    assert built.evaluation_strategy == "epoch"
    assert built.learning_rate == 2e-4


def test_accepted_training_arguments_falls_back_to_dataclass_fields():
    import dataclasses

    @dataclasses.dataclass
    class Sample:
        alpha: int = 1
        beta: str = "b"

    accepted = accepted_training_arguments(Sample)
    assert {"alpha", "beta"} <= accepted


def test_make_training_arguments_skips_filtering_when_class_takes_kwargs():
    """内省不出来时不要擅自丢参数，交给对方自己判断。"""

    class ArgsKwargs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    built = make_training_arguments(ArgsKwargs, {"whatever": 1})
    assert built.kwargs == {"whatever": 1}


def test_encode_example_requests_list_output_from_chat_template():
    """回归测试：transformers v5 的 return_dict 默认值是 True。

    这一条等价于用户踩的坑——假 tokenizer 按 v5 语义（默认返回 BatchEncoding）实现，
    如果 encode_example 忘了显式传 return_dict=False，input_ids 就会变成 dict/str。
    """
    tokenizer = FakeTokenizer()
    row = make_row()
    assert isinstance(
        tokenizer.apply_chat_template(row["messages"], tokenize=True), FakeBatchEncoding
    )  # 确认假 tokenizer 真的在模拟 v5

    encoded = encode_example(tokenizer, row, None, max_len=256, on_too_long="error")
    assert isinstance(encoded.input_ids, list)
    assert all(isinstance(item, int) for item in encoded.input_ids)
    assert all(isinstance(item, int) for item in encoded.labels)


def test_to_id_list_normalizes_tokenizer_return_shapes():
    assert to_id_list([1, 2, 3]) == [1, 2, 3]
    assert to_id_list([[1, 2], [3, 4]]) == [1, 2]           # 带 batch 维
    assert to_id_list({"input_ids": [5, 6]}) == [5, 6]      # BatchEncoding
    assert to_id_list(FakeBatchEncoding(input_ids=[7, 8])) == [7, 8]

    class TensorLike:
        def tolist(self):
            return [[9, 10]]

    assert to_id_list(TensorLike()) == [9, 10]
    with pytest.raises(SystemExit):
        to_id_list(["151644", "198"])


def test_verify_encoded_catches_string_columns():
    """dataset 里存成字符串（就是用户遇到的现象）必须被拦住并给出可读报错。"""
    good = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 4, 5]}]
    verify_encoded(good)  # 正常数据不报错

    bad = [{"input_ids": "123", "attention_mask": "111", "labels": "-10045"}]
    with pytest.raises(SystemExit) as excinfo:
        verify_encoded(bad)
    assert "不是 int 列表" in str(excinfo.value)


def test_verify_encoded_catches_fully_masked_answer():
    bad = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, -100, -100]}]
    with pytest.raises(SystemExit) as excinfo:
        verify_encoded(bad)
    assert "答案 token 数为 0" in str(excinfo.value)


def test_grad_accum_loss_scale_keeps_cpu_and_gpu_equivalent():
    """CPU 的 2×4 与 GPU 的 8×1 必须等价，否则等效学习率会差 4 倍。"""
    assert grad_accum_loss_scale(4, True) == 0.25   # v5 认为已归一化 -> 自己补除法
    assert grad_accum_loss_scale(4, False) == 1.0   # Trainer 会自己除
    assert grad_accum_loss_scale(1, True) == 1.0
    assert grad_accum_loss_scale(1, False) == 1.0


def test_resolve_weight_mode_requires_the_column():
    assert resolve_weight_mode("sample", True) is True
    assert resolve_weight_mode("sample", False) is False
    assert resolve_weight_mode("off", True) is False


def test_run_training_defines_weighted_before_first_use():
    """回归测试：weighted 曾在赋值前被打印使用，直接 UnboundLocalError。

    这类"顺序写错"的问题语法检查抓不到、本地又跑不起 torch，所以用源码顺序兜住。
    """
    source = inspect.getsource(run_training).splitlines()
    assign_index = next(
        index for index, line in enumerate(source) if line.strip().startswith("weighted =")
    )
    use_index = next(
        index
        for index, line in enumerate(source)
        if "weighted" in line and not line.strip().startswith("weighted =")
    )
    assert assign_index < use_index, "weighted 在赋值之前就被使用了"


def test_save_checkpoint_policy_prefers_steps():
    """1 epoch 的 CPU 长跑里 save_strategy="epoch" 等于"只在最后存一次"，所以按步数优先。"""
    assert save_checkpoint_policy(100, False) == ("steps", 100)
    assert save_checkpoint_policy(100, True) == ("steps", 100)
    assert save_checkpoint_policy(0, True) == ("epoch", 0)
    assert save_checkpoint_policy(0, False) == ("no", 0)
