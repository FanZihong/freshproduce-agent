# badcase 挖掘与 v3 训练数据（v3）

- 生成时间：2026-10-16 11:01:47
- 来源：pool（trace 池：线上新数据，不涉及验收集泄漏）
- adapter：（本次没跑 dev，未用到）
- dev 错例：0 条；trace 池可用：2 条（映射不了 0 条）
- 生成的训练样本 2 条；去重后实际新增 2 条
- 基础训练集 2614 条 → 合并后 2616 条：`train_sft_v3.jsonl`


## 二、trace 池（人工标注的线上样本）

> 人工填的是自然语言，脚本翻译成体系 key 的方式记在「匹配方式」列。凡是走「关键词兜底」的行都要人再过一眼。

| id | 用户原话 | 人工期望 → key | 匹配方式 | 模型当时判定 | 结果 |
| --- | --- | --- | --- | --- | --- |
| trace-trace-handoff-t1 | 你们客服太差了，我要投诉 | human_handoff | 名字/别名精确匹配 | human_handoff | 一致（正样本） |
| trace-trace-ood-t1 | 你们能帮我写一段 python 代码吗 | chitchat | 关键词兜底（域外） | chitchat | 一致（正样本） |

## 下一步（训练完成后跑门禁）

```bash
python scripts/train_intent.py --train-file /mnt/d/生鲜售后agent/agent/data/intent/dataset_v1_balanced/train_sft_v3.jsonl \
  --prompt-detail catalog --max-len 0 --epochs 1 --save-steps 100 --eval-limit 0 \
  --out-dir data/intent/models/v3
python scripts/eval_intent.py --backends local --local-dir data/intent/models/v3 --splits test
python scripts/evolve_intent.py --gate --baseline-dir  --candidate-dir data/intent/models/v3
```

