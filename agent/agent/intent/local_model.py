"""本地小模型后端：加载 Qwen2.5 + LoRA adapter 做意图识别。

三件事必须和训练侧对齐，否则指标会莫名下降：
1. **提示词**：从 adapter 目录的 `train_config.json` 读 `prompt_detail` 与 `label_format`，
   用同一个 `build_training_system_prompt` 重建 system prompt；
2. **目标串格式**：解析复用 `agent.intent.target.parse_target`（宽容解析，容忍漂移）；
3. **截断口径**：沿用训练时的 `max_len`。

置信度给了两条路：
- `classify()` 默认用**生成答案序列的平均 token 对数概率**（一次生成就能拿到，便宜）；
- `score_labels()` 给出 10 类的 **softmax 概率**（对每个候选标签各算一次对数似然，
  贵约 10 倍，留给 Day 3 的温度校准与拒答阈值实验用）。

模型缺失时抛 `ModelNotFoundError`，消息里直接给出训练命令——图那一层会捕获它并回落规则版。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from agent.config import get_env
from agent.db import DATA_DIR
from agent.intent.baseline import build_messages, build_training_system_prompt
from agent.intent.classifier import IntentResult
from agent.intent.target import format_target, parse_target
from agent.intent.taxonomy import load_taxonomy

DEFAULT_ADAPTER_DIR = DATA_DIR / "intent" / "models" / "v1"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class ModelNotFoundError(RuntimeError):
    """adapter 还没训练出来（或路径不对）。"""


def local_context(context: Any, *, enabled: bool | None = None) -> Any:
    """本地 SFT 模型的上下文开关：默认不带上下文。

    训练数据是单轮 prompt，没有"上一轮用户问题 / 已知记忆"这些行。
    推理时硬塞进去，模型会把提示词原样复读（实测会输出"上一轮客服回答：…"），
    然后解析失败。默认关闭不是功能缺失，是**训练/推理一致**。
    想验证上下文是否有效，用 `LOCAL_INTENT_CONTEXT=on`（并最好按带上下文的样本再训一版）。
    """
    if enabled is None:
        mode = (get_env("LOCAL_INTENT_CONTEXT", "off") or "off").strip().lower()
        enabled = mode in {"on", "1", "true", "yes"}
    return context if enabled else None


@dataclass
class LocalConfig:
    """推理配置；默认值来自 adapter 目录里的 train_config.json。"""

    base_model: str = DEFAULT_BASE_MODEL
    adapter_dir: str = ""
    prompt_detail: str = "minimal"
    label_format: str = "key"
    max_len: int = 512
    max_new_tokens: int = 24
    batch_size: int = 8
    device: str = "auto"

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_adapter_dir(adapter_dir: Path | str | None = None) -> Path:
    """优先级：显式传参 > INTENT_LOCAL_DIR > 已发布版本（active.json）> 默认 models/v1。

    环境变量排第二是有意的：评测、A/B、临时实验要能临时压过"线上已发布版本"；
    不设环境变量时，跑哪个版本由 `scripts/release.py` 写下的 active.json 决定。
    """
    if adapter_dir:
        return Path(adapter_dir)
    from agent.config import get_env

    configured = get_env("INTENT_LOCAL_DIR", "")
    if configured:
        return Path(configured)
    from agent.intent.active import active_adapter_dir

    released = active_adapter_dir()
    return released if released else DEFAULT_ADAPTER_DIR


def adapter_exists(adapter_dir: Path) -> bool:
    """adapter 的落盘标志：PEFT 总会写 adapter_config.json。"""
    if not adapter_dir.exists():
        return False
    if (adapter_dir / "adapter_config.json").exists():
        return True
    # 也允许直接指向训练时的 checkpoints/checkpoint-N
    return any(adapter_dir.glob("checkpoint-*/adapter_config.json"))


def load_local_config(adapter_dir: Path | str | None = None) -> LocalConfig:
    """读 adapter 目录里的 train_config.json；文件缺失时用默认值并把路径带上。"""
    path = resolve_adapter_dir(adapter_dir)
    config = LocalConfig(adapter_dir=str(path))
    train_config = path / "train_config.json"
    if train_config.exists():
        payload = json.loads(train_config.read_text(encoding="utf-8"))
        config.base_model = payload.get("base_model") or config.base_model
        config.prompt_detail = payload.get("prompt_detail") or config.prompt_detail
        config.label_format = payload.get("label_format") or config.label_format
        config.max_len = int(payload.get("max_len") or config.max_len)
    return config


def sequence_confidence(
    generated_ids: Sequence[int], step_logits: Sequence[Sequence[float]], eos_id: int | None
) -> float:
    """生成答案的平均 token 概率（几何平均），作为置信度。

    step_logits[t] 是第 t 步该样本的 logits（长度 = 词表大小）。
    停止条件：遇到 eos 或步数用尽；空答案返回 0。
    """
    logprobs: list[float] = []
    for step, token_id in enumerate(generated_ids):
        if step >= len(step_logits):
            break
        if eos_id is not None and token_id == eos_id:
            break
        row = step_logits[step]
        if not row:
            break
        max_logit = max(row)
        total = sum(math.exp(value - max_logit) for value in row)
        logprob = (row[int(token_id)] - max_logit) - math.log(total)
        logprobs.append(logprob)
    if not logprobs:
        return 0.0
    return round(math.exp(sum(logprobs) / len(logprobs)), 4)


def target_logprob(
    full_ids: Sequence[int], prompt_len: int, logits: Sequence[Sequence[float]]
) -> float:
    """候选目标串的对数似然：对目标段每个 token 取 log_softmax 后求和。"""
    total = 0.0
    for position in range(prompt_len, len(full_ids)):
        row = logits[position - 1]
        if not row:
            continue
        max_logit = max(row)
        normalizer = max_logit + math.log(sum(math.exp(value - max_logit) for value in row))
        total += row[int(full_ids[position])] - normalizer
    return total


def target_mean_logprob(
    full_ids: Sequence[int], prompt_len: int, logits: Sequence[Sequence[float]]
) -> float:
    """候选目标串的**平均** token 对数概率。

    这里必须用均值而不是求和：求和会让长标签（bad_fruit_claim 比 chitchat 长得多）被
    系统性压低，分类结果直接偏向短标签。实践中这也是"标签似然打分"最容易踩的坑。
    """
    count = max(0, len(full_ids) - prompt_len)
    if count == 0:
        return float("-inf")
    return target_logprob(full_ids, prompt_len, logits) / count


class LocalIntentClassifier:
    """本地 LoRA 模型后端，接口与规则版 / few-shot 版一致。"""

    name = "local-lora"

    def __init__(
        self,
        adapter_dir: Path | str | None = None,
        config: LocalConfig | None = None,
        batch_size: int | None = None,
        strict: bool = False,
    ) -> None:
        self.config = config or load_local_config(adapter_dir)
        self.adapter_dir = Path(self.config.adapter_dir or resolve_adapter_dir(adapter_dir))
        self.batch_size = batch_size or self.config.batch_size
        self.strict = strict
        self._loaded = False
        self._tokenizer = None
        self._model = None
        self._torch = None
        self.system_prompt: str | None = None

    # ---------- 可用性 ----------

    @property
    def available(self) -> bool:
        return adapter_exists(self.adapter_dir)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.available:
            raise ModelNotFoundError(
                f"没找到 LoRA adapter：{self.adapter_dir}\n"
                "  先训练：python scripts/train_intent.py --epochs 1\n"
                "  （长跑建议加 --save-steps 100，中断也能续）\n"
                "  或者用 INTENT_LOCAL_DIR 指向已有的 adapter 目录"
            )
        # 这一句要 flush：项目在 /mnt/d（WSL 的 drvfs）上时，import torch 会做大量文件系统
        # 元数据查询，首次加载可能几十秒——没有提示的话会被误当成卡死。
        print(
            "  正在加载本地意图模型（首次需 import torch/transformers/peft + 读 adapter，"
            "约 10–60 秒；同一进程内后续调用是毫秒级）…",
            flush=True,
        )
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise ModelNotFoundError(
                f"缺少推理依赖（{exc.name}）：pip install transformers peft torch"
            ) from exc

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # 解码器-only 模型**必须左填充**：右填充时一批里除最长那条外，模型会从 pad 位置
        # 继续生成，输出全错（transformers 只给一行 warning，指标却会莫名其妙很差）。
        self._tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(self.config.base_model, dtype=dtype)
        self._model = PeftModel.from_pretrained(base, str(self.adapter_dir))
        self._model.to(device)
        self._model.eval()
        self._torch = torch
        self.system_prompt = build_training_system_prompt(
            load_taxonomy(),
            target_style=self.config.label_format,
            detail=self.config.prompt_detail,
        )
        self._loaded = True

    def preload(self) -> bool:
        """显式预热：把首次加载的几十秒放到启动阶段，而不是用户的第一句话上。

        加载失败返回 False（不抛异常），调用方可以据此提示"本次按规则版运行"。
        """
        try:
            self._ensure_loaded()
        except Exception:  # noqa: BLE001 - 预热失败不该影响服务启动
            return False
        return True

    # ---------- 推理 ----------

    def build_prompt(self, text: str, context: Any = None) -> str:
        """和训练时完全一致的 prompt（system + user 两段拼起来喂给模型）。

        上下文默认不拼：SFT 数据是单轮的，见 `local_context`。
        """
        self._ensure_loaded()
        messages = build_messages(
            text,
            local_context(context),
            taxonomy=load_taxonomy(),
            system_prompt=self.system_prompt,
            output_hint="请输出意图标签：",
        )
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, return_dict=False, add_generation_prompt=True
        )

    def classify_batch(self, texts: Sequence[str], contexts: Sequence[Any] | None = None):
        """批量推理；返回 (results, latency_ms_per_sample)。"""
        self._ensure_loaded()
        torch = self._torch
        taxonomy = load_taxonomy()
        valid = set(taxonomy.keys)
        max_labels = int(taxonomy.multi_intent_policy.get("max_labels", 2))
        contexts = list(contexts) if contexts else [None] * len(texts)

        results: list[IntentResult] = []
        per_sample_ms: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            chunk_contexts = contexts[start : start + self.batch_size]
            prompts = [self.build_prompt(text, ctx) for text, ctx in zip(chunk, chunk_contexts)]
            started = time.perf_counter()
            encoded = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_len,
            ).to(self.device)
            with torch.no_grad():
                output = self._model.generate(
                    **encoded,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            elapsed_ms = (time.perf_counter() - started) * 1000
            per_sample_ms.extend([elapsed_ms / max(1, len(chunk))] * len(chunk))

            sequences = _to_list(output.sequences)
            scores = [_to_list(step) for step in (output.scores or [])]
            attention = _to_list(encoded["attention_mask"])
            # 左填充后 prompt 在批内等宽，生成段从"填充后的宽度"开始（不是 1 的个数）
            prompt_width = len(attention[0]) if attention else 0
            for index, prompt_ids in enumerate(sequences):
                generated = list(prompt_ids[prompt_width:])
                raw = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
                confidence = sequence_confidence(
                    generated,
                    [step[index] for step in scores],  # 每步取该样本的 logits 行
                    self._tokenizer.eos_token_id,
                )
                results.append(self._to_result(raw, confidence, valid, max_labels, taxonomy))
        return results, per_sample_ms

    def classify(self, text: str, context: Any = None) -> IntentResult:
        results, _ = self.classify_batch([text], [context])
        return results[0]

    def _to_result(
        self, raw: str, confidence: float, valid: set[str], max_labels: int, taxonomy
    ) -> IntentResult:
        try:
            labels, _, _ = parse_target(raw, valid, taxonomy=taxonomy, max_labels=max_labels)
        except ValueError as exc:
            # 解析失败对外抛错：图那层会捕获并回落规则版，评测脚本会记为"未识别"
            raise ValueError(f"本地模型输出无法解析（{exc}）：{raw[:60]}") from exc
        return IntentResult(labels[0], confidence, f"local:{raw[:40]}", labels)

    def score_labels(
        self, text: str, context: Any = None, labels: Sequence[str] | None = None
    ) -> dict[str, float]:
        """对每个候选标签算对数似然再过 softmax —— 真正的 10 类概率分布。

        比 classify() 贵约 10 倍（每个候选一次前向），只用于校准与误差分析。
        """
        self._ensure_loaded()
        torch = self._torch
        taxonomy = load_taxonomy()
        candidates = list(labels or taxonomy.keys)
        prompt = self.build_prompt(text, context)
        prompt_ids = _to_list(
            self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )

        logprobs: dict[str, float] = {}
        for label in candidates:
            target = format_target([label], style=self.config.label_format, taxonomy=taxonomy)
            target_ids = _to_list(
                self._tokenizer(target, add_special_tokens=False)["input_ids"]
            )
            full_ids = prompt_ids + target_ids
            encoded = self._tokenizer(
                self._tokenizer.decode(full_ids), return_tensors="pt", truncation=True,
                max_length=self.config.max_len,
            ).to(self.device)
            with torch.no_grad():
                logits = _to_list(self._model(**encoded).logits[0])
            # 用平均对数概率，避免长标签被长度偏置压低
            logprobs[label] = target_mean_logprob(full_ids, len(prompt_ids), logits)

        peak = max(logprobs.values())
        weights = {label: math.exp(value - peak) for label, value in logprobs.items()}
        total = sum(weights.values()) or 1.0
        return {label: round(value / total, 4) for label, value in weights.items()}


def _to_list(value) -> list:
    """统一 torch tensor / numpy / list 的取值方式。"""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


_DEFAULT_CLASSIFIER: LocalIntentClassifier | None = None


def get_local_classifier(**kwargs) -> LocalIntentClassifier:
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = LocalIntentClassifier(**kwargs)
    return _DEFAULT_CLASSIFIER
