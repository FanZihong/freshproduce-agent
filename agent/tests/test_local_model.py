"""本地 LoRA 后端测试：置信度数学、配置读取、缺失模型报错、批量推理解析。

torch / transformers / peft 全部用假模块注入（local_model 只在 _ensure_loaded 里惰性导入），
所以这些用例在没有 GPU、没装 torch 的机器上也能跑。
"""

from __future__ import annotations

import json
import math
import sys
import types

import pytest

from agent.intent import local_model
from agent.intent.local_model import (
    LocalConfig,
    LocalIntentClassifier,
    ModelNotFoundError,
    adapter_exists,
    load_local_config,
    resolve_adapter_dir,
    sequence_confidence,
    target_mean_logprob,
    target_logprob,
)
from agent.intent.taxonomy import load_taxonomy


# ---------------- 纯函数：置信度与对数似然 ----------------


def test_sequence_confidence_is_geometric_mean_of_token_probs():
    # 词表 3 个 token、logits 全等 -> 每个 token 概率 1/3，几何平均仍是 1/3
    step_logits = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    confidence = sequence_confidence([1, 2], step_logits, eos_id=None)
    assert confidence == pytest.approx(round(1 / 3, 4))


def test_sequence_confidence_stops_at_eos_and_handles_empty():
    eos = 9
    step_logits = [[0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    # 遇到 eos 就停：只算第一步，概率 = e^4 / (e^4 + 1 + 1)
    expected = math.exp(4) / (math.exp(4) + 2)
    assert sequence_confidence([1, eos, 2], step_logits, eos_id=eos) == pytest.approx(expected, abs=0.001)
    assert sequence_confidence([], [], eos_id=eos) == 0.0


def test_target_logprob_sums_only_target_tokens():
    # 词表 4、logits 全是 [0,1,0,0]：token 1 的概率 e/(e+3)，token 3 的概率 1/(e+3)
    logits = [[0.0, 1.0, 0.0, 0.0]] * 4
    value = target_logprob([7, 1, 3], 1, logits)
    denominator = math.exp(1) + 3
    assert value < 0
    assert value == pytest.approx(math.log(math.exp(1) / denominator) + math.log(1 / denominator), rel=1e-6)


def test_target_mean_logprob_removes_length_bias():
    """同样"每个 token 都很好"的情况下，长标签不该因为更长而被压低。"""
    logits = [[0.0, 4.0, 0.0]] * 8
    short = target_mean_logprob([7, 1, 1], 1, logits)
    long = target_mean_logprob([7, 1, 1, 1, 1, 1, 1, 1], 1, logits)
    assert short == pytest.approx(long, rel=1e-9)      # 均值口径下长度无关


# ---------------- 配置与可用性 ----------------


def test_resolve_adapter_dir_prefers_explicit_then_env(monkeypatch, tmp_path):
    assert resolve_adapter_dir(tmp_path) == tmp_path
    monkeypatch.setenv("INTENT_LOCAL_DIR", str(tmp_path))
    assert resolve_adapter_dir(None) == tmp_path


def test_load_local_config_reads_train_config(tmp_path):
    (tmp_path / "train_config.json").write_text(
        json.dumps(
            {
                "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "prompt_detail": "catalog",
                "label_format": "name",
                "max_len": 640,
            }
        ),
        encoding="utf-8",
    )
    config = load_local_config(tmp_path)
    assert config.prompt_detail == "catalog"
    assert config.label_format == "name"
    assert config.max_len == 640
    assert config.adapter_dir == str(tmp_path)


def test_load_local_config_defaults_when_file_missing(tmp_path):
    config = load_local_config(tmp_path)
    assert config.prompt_detail == "minimal"
    assert config.label_format == "key"
    assert config.adapter_dir == str(tmp_path)


def test_adapter_exists_detects_peft_layout(tmp_path):
    assert adapter_exists(tmp_path) is False
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert adapter_exists(tmp_path) is True

    nested = tmp_path / "checkpoints" / "checkpoint-100"
    nested.mkdir(parents=True)
    (nested / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert adapter_exists(tmp_path / "checkpoints") is True


def test_missing_adapter_raises_actionable_error(tmp_path):
    classifier = LocalIntentClassifier(adapter_dir=tmp_path)
    assert classifier.available is False
    with pytest.raises(ModelNotFoundError) as excinfo:
        classifier.classify("怎么开发票")
    message = str(excinfo.value)
    assert "train_intent.py" in message      # 报错里直接给训练命令
    assert str(tmp_path) in message


# ---------------- 假 torch / transformers / peft ----------------


class FakeTensor:
    def __init__(self, data):
        self.data = data

    def tolist(self):
        return self.data

    def __getitem__(self, item):
        return FakeTensor(self.data[item])

    def to(self, device):
        return self

    def __len__(self):
        return len(self.data)


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    """按字符切 token 的假 tokenizer，chat template 与 Qwen 语义一致。"""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.padding_side = "right"   # 真实 tokenizer 默认就是 right，靠 local_model 纠正
        self._to_id: dict[str, int] = {}
        self._inverse: dict[int, str] = {}

    def _id(self, char: str) -> int:
        if char not in self._to_id:
            new = 10 + len(self._to_id)
            self._to_id[char] = new
            self._inverse[new] = char
        return self._to_id[char]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id(char) for char in text]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(self._inverse.get(int(item), "") for item in ids)

    def apply_chat_template(self, messages, tokenize=False, return_dict=True, add_generation_prompt=False):
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        text = "".join(parts)
        return self.encode(text) if tokenize else text

    def __call__(self, texts, return_tensors=None, padding=False, truncation=False,
                 max_length=None, add_special_tokens=True):
        if isinstance(texts, str):
            ids = self.encode(texts)
            return FakeBatch(input_ids=FakeTensor(ids), attention_mask=FakeTensor([1] * len(ids)))
        batch_ids = [self.encode(text) if isinstance(text, str) else list(text) for text in texts]
        if not padding:
            return FakeBatch(
                input_ids=FakeTensor(batch_ids),
                attention_mask=FakeTensor([[1] * len(ids) for ids in batch_ids]),
            )
        width = max(len(ids) for ids in batch_ids)
        padded, masks = [], []
        for ids in batch_ids:
            pad = width - len(ids)
            if self.padding_side == "left":
                padded.append([self.pad_token_id] * pad + ids)
                masks.append([0] * pad + [1] * len(ids))
            else:
                padded.append(ids + [self.pad_token_id] * pad)
                masks.append([1] * len(ids) + [0] * pad)
        return FakeBatch(input_ids=FakeTensor(padded), attention_mask=FakeTensor(masks))


class FakeModel:
    """假模型：按 prompt 里出现的关键字决定输出，logits 让正确标签得分最高。"""

    VOCAB = 256  # 假 tokenizer 的 id 随字符数增长，词表给小了会越界

    def __init__(self, tokenizer: FakeTokenizer, answers: dict[str, str]):
        self.tokenizer = tokenizer
        self.answers = answers
        self.calls = 0

    def _answer_for(self, prompt_text: str) -> str:
        for keyword, answer in self.answers.items():
            if keyword in prompt_text:
                return answer
        return "看不懂"

    def generate(self, input_ids, attention_mask=None, **kwargs):
        self.calls += 1
        batch_ids = input_ids.tolist()
        if batch_ids and not isinstance(batch_ids[0], list):
            batch_ids = [batch_ids]
        vocab = self.VOCAB
        width = len(batch_ids[0])            # 左填充后 prompt 等宽
        answers = []
        for ids in batch_ids:
            answer_ids = self.tokenizer.encode(
                self._answer_for(self.tokenizer.decode(ids))
            ) + [self.tokenizer.eos_token_id]
            answers.append(answer_ids)

        max_gen = max(len(ids) for ids in answers)
        sequences = [
            list(ids) + answer_ids + [self.tokenizer.pad_token_id] * (max_gen - len(answer_ids))
            for ids, answer_ids in zip(batch_ids, answers)
        ]

        scores = []
        for step in range(max_gen):
            rows = []
            for answer_ids in answers:
                row = [0.0] * vocab
                token = answer_ids[step] if step < len(answer_ids) else self.tokenizer.eos_token_id
                row[min(token, vocab - 1)] = 5.0
                rows.append(row)
            scores.append(FakeTensor(rows))
        assert width == len(batch_ids[0])
        return types.SimpleNamespace(sequences=FakeTensor(sequences), scores=scores)

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        data = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        ids = data[0] if data and isinstance(data[0], list) else data
        vocab = self.VOCAB
        correct_label = self._answer_for(self.tokenizer.decode(ids))
        good_ids = {self.tokenizer._id(char) for char in correct_label}
        logits = []
        for position in range(len(ids)):
            row = [0.0] * vocab
            for token in good_ids:
                row[min(token, vocab - 1)] = 4.0
            logits.append(row)
        return types.SimpleNamespace(logits=FakeTensor([logits]))

    def to(self, device):
        return self

    def eval(self):
        return self


class FakePeftModel(FakeModel):
    @classmethod
    def from_pretrained(cls, base, path):
        # 真实代码会把 base 传进来；这里直接复用测试准备好的实例
        return _FAKE_PEFT_INSTANCE


_FAKE_PEFT_INSTANCE = None


def install_fake_stack(answers: dict[str, str]) -> FakeTokenizer:
    """把假 torch / transformers / peft 塞进 sys.modules，供 local_model 惰性导入。"""
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, answers)

    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.no_grad = lambda: __import__("contextlib").nullcontext()
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
    transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda *a, **k: model)

    global _FAKE_PEFT_INSTANCE
    _FAKE_PEFT_INSTANCE = model
    peft = types.ModuleType("peft")
    peft.PeftModel = FakePeftModel

    sys.modules["torch"] = torch
    sys.modules["transformers"] = transformers
    sys.modules["peft"] = peft
    return tokenizer


def make_adapter_dir(tmp_path, detail: str = "minimal") -> "object":
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "train_config.json").write_text(
        json.dumps({"prompt_detail": detail, "label_format": "key", "max_len": 512}),
        encoding="utf-8",
    )
    return tmp_path


def test_classify_batch_parses_generated_labels(tmp_path, monkeypatch):
    tokenizer = install_fake_stack({"草莓坏了": "主:bad_fruit_claim", "怎么开发票": "主:invoice"})
    classifier = LocalIntentClassifier(adapter_dir=make_adapter_dir(tmp_path), batch_size=2)
    assert classifier.available is True

    # 两条样本长度差得多：右填充时短的那条会被 pad 影响，这里就是要盯住这一点
    short_text = "草莓坏了"
    long_text = "我上周买的那个订单怎么开发票啊，麻烦帮我看看抬头能不能改"
    results, latencies = classifier.classify_batch([short_text, long_text])

    assert tokenizer.padding_side == "left", "解码器模型必须左填充，否则批里除最长的一条外输出都是错的"
    assert [result.label for result in results] == ["bad_fruit_claim", "invoice"]
    assert all(0.0 < result.confidence <= 1.0 for result in results)
    assert len(latencies) == 2
    assert all(value >= 0 for value in latencies)


def test_classify_uses_train_config_prompt_detail(tmp_path):
    install_fake_stack({"草莓坏了": "主:bad_fruit_claim"})
    classifier = LocalIntentClassifier(adapter_dir=make_adapter_dir(tmp_path, detail="compact"))
    classifier.classify("草莓坏了")
    assert classifier._loaded is True
    assert "可选类别" in classifier.system_prompt      # 用的是重建后的 system prompt


def test_unparsable_output_raises_so_graph_can_fall_back(tmp_path):
    install_fake_stack({"草莓坏了": "完全看不懂的答案"})
    classifier = LocalIntentClassifier(adapter_dir=make_adapter_dir(tmp_path))
    with pytest.raises(ValueError):
        classifier.classify("草莓坏了")


def test_score_labels_returns_normalized_distribution(tmp_path):
    install_fake_stack({"草莓坏了": "主:bad_fruit_claim"})
    classifier = LocalIntentClassifier(adapter_dir=make_adapter_dir(tmp_path))
    scores = classifier.score_labels("草莓坏了")

    assert set(scores) == set(load_taxonomy().keys)
    assert sum(scores.values()) == pytest.approx(1.0, abs=0.01)
    assert max(scores, key=scores.get) == "bad_fruit_claim"


def test_local_config_dataclass_fields_serializable():
    payload = LocalConfig().to_dict()
    assert payload["prompt_detail"] == "minimal"
    assert "adapter_dir" in payload
