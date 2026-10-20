# 意图识别基线报告

- 生成时间：2026-10-19 10:45:41
- 数据集：dataset_v1_balanced（test 701 条）
- 评测命令：`python scripts/eval_intent.py --backends local --splits test --report /mnt/d/生鲜售后agent/agent/eval/reports/intent_baseline.md --json /mnt/d/生鲜售后agent/agent/eval/reports/intent_baseline.json`
- 计价假设：输入 ¥0.0/百万 token、输出 ¥0.0/百万 token（改动请设 INTENT_PRICE_IN / INTENT_PRICE_OUT，报告口径随之变化）
- scikit-learn：已启用（F1 走 sklearn 口径）
- test 集锁死只用于最终评估，dev 集用于调阈值；P95 时延只统计真实调用（缓存命中不计入）。

## 总览

| 后端 | 条数 | macro-F1 | accuracy | 多标签完全一致 | 多标签 F1(macro/micro) | P95 时延(ms) | 单次成本(¥) | 千次成本(¥) | 失败 | 缓存命中 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local LoRA（test） | 701 | 0.7366 | 0.7375 | 0.7332 | 0.7409 / 0.7417 | 0.0 | 0.000000 | 0.0000 | 2 | 701 |

## 逐类指标（local LoRA（test））

| 类别 | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| query_order | 0.8373 | 0.7675 | 0.8009 | 228 |
| urge_logistics | 0.6726 | 0.8172 | 0.7379 | 93 |
| bad_fruit_claim | 0.9167 | 0.7097 | 0.8000 | 31 |
| short_weight | 0.8421 | 0.9412 | 0.8889 | 17 |
| refund_consult | 0.7570 | 0.8617 | 0.8060 | 94 |
| invoice | 1.0000 | 0.9310 | 0.9643 | 29 |
| quality_qa | 0.5435 | 0.4902 | 0.5155 | 51 |
| complaint | 0.5652 | 0.7222 | 0.6341 | 18 |
| chitchat | 0.6040 | 0.5980 | 0.6010 | 102 |
| human_handoff | 0.7000 | 0.5526 | 0.6176 | 38 |

主要混淆（真值 → 误判为）：
- query_order → urge_logistics：25 条
- quality_qa → chitchat：12 条
- chitchat → quality_qa：12 条

token 用量：平均输入 0.0 / 平均输出 0.0，时延 P50 0.0ms / 均值 0.0ms
多标签样本 9 条，多标签完全一致率 0.7332（主标签 macro-F1 0.7366 不含辅助标签）

## 错例（主标签判错，每个后端最多 10 条）

### local LoRA（test）

| id | 问句 | 真值 | 预测 | 置信度 |
| --- | --- | --- | --- | --- |
| c-05781cf7fb | 一点包装都没有了也可以吗?#E-s[数字x] | bad_fruit_claim | quality_qa | 0.83 |
| c-8af0afbd03 | 就是要洗的尺寸错了 | bad_fruit_claim | chitchat | 0.86 |
| c-691eb92439 | 2301929 快递员能送货的同时 售后处理这个有问题的么 | bad_fruit_claim | quality_qa | 0.89 |
| c-8dd2ab4f66 | 我买了六瓶酒 结果快递给我打碎了两瓶 | bad_fruit_claim | quality_qa | 0.75 |
| c-d7b758d1ae | 你好 这个表带边上小圆圈有点瑕疵 能给补发一个吗 | bad_fruit_claim | quality_qa | 0.81 |
| c-43081d758d | [数字x]月买的，我提交了给我关闭了 | bad_fruit_claim | query_order | 0.78 |
| c-c46d64a138 | 门框边缘有磨损痕迹 | bad_fruit_claim | quality_qa | 0.84 |
| c-ad69fcefd8 | 我把有问题的东西邮过去，运费不得给我补偿吗 | bad_fruit_claim | refund_consult | 0.94 |
| c-c626a2a4b1 | 请问需要等多久 | bad_fruit_claim | urge_logistics | 0.85 |
| c-bf36d7810a | 我用的那个是直接减[数字x]的，我付款以后发现还有一张买手表减[数字x]的没用 | chitchat | refund_consult | 0.83 |

## 结论与下一步

- local LoRA（test）：macro-F1 0.7366，最弱三类 quality_qa(F1 0.52/真值 51)、chitchat(F1 0.60/真值 102)、human_handoff(F1 0.62/真值 38)
- local LoRA（test）：2 条调用失败（0.3%），这些样本按判错计入指标；超时/限流居多时可调大 INTENT_TIMEOUT 或降低 --concurrency，重跑会自动补上（失败不写缓存）
- local LoRA（test）：本次全部命中缓存，P95 时延不可用；要拿真实时延请加 --no-cache 重跑
- 最需要盯的混淆：query_order 被误判为 urge_logistics（25 条），先查 taxonomy 边界定义
- 最需要盯的混淆：quality_qa 被误判为 chitchat（12 条），先查 taxonomy 边界定义
- 最需要盯的混淆：chitchat 被误判为 quality_qa（12 条），先查 taxonomy 边界定义

复现命令：

```bash
python scripts/eval_intent.py --backends local --splits test --report /mnt/d/生鲜售后agent/agent/eval/reports/intent_baseline.md --json /mnt/d/生鲜售后agent/agent/eval/reports/intent_baseline.json
```
