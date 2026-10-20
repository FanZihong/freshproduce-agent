# 意图识别 LoRA adapter（v1）

- 底座：Qwen/Qwen2.5-0.5B-Instruct
- 训练数据：/mnt/d/生鲜售后agent/agent/data/intent/dataset_v1_balanced/train_sft.jsonl（None 条）
- 提示词详略：minimal；目标串风格：key
- max_len=384；类别加权：sample

## 推理时必须保持一致

`local_model.py` 会用 `train_config.json` 里的 prompt_detail / label_format 重建同一段
system prompt——训练与推理的提示词不一致是本项目最容易踩的坑。

## 加载

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained(train_config['base_model'])
model = PeftModel.from_pretrained(base, 'data/intent/models/v1')
```

## 下一步

```bash
python scripts/eval_intent.py --backends rule fewshot local
```
