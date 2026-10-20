# 意图识别基线报告

- 生成时间：2026-10-14 12:14:32
- 数据集：dataset_v1_balanced（test 701 条）
- 评测命令：`python scripts/eval_intent.py --backends rule fewshot local --local-dir data/intent/models/v2 --splits test`
- 计价假设：输入 ¥2.0/百万 token、输出 ¥8.0/百万 token（改动请设 INTENT_PRICE_IN / INTENT_PRICE_OUT，报告口径随之变化）
- scikit-learn：已启用（F1 走 sklearn 口径）
- test 集锁死只用于最终评估，dev 集用于调阈值；P95 时延只统计真实调用（缓存命中不计入）。

## 总览

| 后端 | 条数 | macro-F1 | accuracy | 多标签完全一致 | 多标签 F1(macro/micro) | P95 时延(ms) | 单次成本(¥) | 千次成本(¥) | 失败 | 缓存命中 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rule-based-v1（test） | 701 | 0.2013 | 0.3923 | 0.3894 | 0.2014 / 0.3912 | 0.0 | 0.000000 | 0.0000 | 0 | 0 |
| few-shot deepseek-chat（1-shot，test） | 701 | 0.8490 | 0.8459 | 0.8402 | 0.8465 / 0.8440 | 0.0 | 0.004212 | 4.2121 | 0 | 701 |
| local LoRA（test） | 701 | 0.7085 | 0.7347 | 0.7332 | 0.7147 / 0.7371 | 1955.3 | 0.000000 | 0.0000 | 2 | 0 |

## 逐类指标（rule-based-v1（test））

| 类别 | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| query_order | 0.6278 | 0.6140 | 0.6208 | 228 |
| urge_logistics | 0.0000 | 0.0000 | 0.0000 | 93 |
| bad_fruit_claim | 0.6364 | 0.6774 | 0.6562 | 31 |
| short_weight | 0.0000 | 0.0000 | 0.0000 | 17 |
| refund_consult | 0.0000 | 0.0000 | 0.0000 | 94 |
| invoice | 0.0000 | 0.0000 | 0.0000 | 29 |
| quality_qa | 0.1739 | 0.3137 | 0.2238 | 51 |
| complaint | 0.0000 | 0.0000 | 0.0000 | 18 |
| chitchat | 0.2840 | 0.9412 | 0.4364 | 102 |
| human_handoff | 0.1333 | 0.0526 | 0.0755 | 38 |

主要混淆（真值 → 误判为）：
- query_order → chitchat：67 条
- refund_consult → chitchat：46 条
- urge_logistics → query_order：44 条

token 用量：平均输入 0.0 / 平均输出 0.0，时延 P50 0.0ms / 均值 0.0ms
多标签样本 9 条，多标签完全一致率 0.3894（主标签 macro-F1 0.2013 不含辅助标签）

## 逐类指标（few-shot deepseek-chat（1-shot，test））

| 类别 | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| query_order | 0.8075 | 0.9386 | 0.8682 | 228 |
| urge_logistics | 1.0000 | 0.6559 | 0.7922 | 93 |
| bad_fruit_claim | 0.9600 | 0.7742 | 0.8571 | 31 |
| short_weight | 1.0000 | 1.0000 | 1.0000 | 17 |
| refund_consult | 0.8936 | 0.8936 | 0.8936 | 94 |
| invoice | 1.0000 | 0.9655 | 0.9825 | 29 |
| quality_qa | 0.8125 | 0.7647 | 0.7879 | 51 |
| complaint | 0.8235 | 0.7778 | 0.8000 | 18 |
| chitchat | 0.7890 | 0.8431 | 0.8152 | 102 |
| human_handoff | 0.7027 | 0.6842 | 0.6933 | 38 |

主要混淆（真值 → 误判为）：
- urge_logistics → query_order：30 条
- chitchat → query_order：8 条
- quality_qa → chitchat：6 条

token 用量：平均输入 1949.1 / 平均输出 39.2，时延 P50 0.0ms / 均值 0.0ms
多标签样本 9 条，多标签完全一致率 0.8402（主标签 macro-F1 0.8490 不含辅助标签）

## 逐类指标（local LoRA（test））

| 类别 | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| query_order | 0.7886 | 0.8509 | 0.8186 | 228 |
| urge_logistics | 0.8429 | 0.6344 | 0.7239 | 93 |
| bad_fruit_claim | 0.9545 | 0.6774 | 0.7925 | 31 |
| short_weight | 1.0000 | 0.8235 | 0.9032 | 17 |
| refund_consult | 0.7315 | 0.8404 | 0.7822 | 94 |
| invoice | 1.0000 | 0.9310 | 0.9643 | 29 |
| quality_qa | 0.5400 | 0.5294 | 0.5347 | 51 |
| complaint | 0.5417 | 0.7222 | 0.6190 | 18 |
| chitchat | 0.5649 | 0.7255 | 0.6352 | 102 |
| human_handoff | 1.0000 | 0.1842 | 0.3111 | 38 |

主要混淆（真值 → 误判为）：
- urge_logistics → query_order：21 条
- human_handoff → chitchat：21 条
- quality_qa → chitchat：13 条

token 用量：平均输入 0.0 / 平均输出 0.0，时延 P50 1556.5ms / 均值 1649.1ms
多标签样本 9 条，多标签完全一致率 0.7332（主标签 macro-F1 0.7085 不含辅助标签）

## 错例（主标签判错，每个后端最多 10 条）

### rule-based-v1（test）

| id | 问句 | 真值 | 预测 | 置信度 |
| --- | --- | --- | --- | --- |
| c-05781cf7fb | 一点包装都没有了也可以吗?#E-s[数字x] | bad_fruit_claim | chitchat | 0.35 |
| c-8af0afbd03 | 就是要洗的尺寸错了 | bad_fruit_claim | chitchat | 0.35 |
| c-691eb92439 | 2301929 快递员能送货的同时 售后处理这个有问题的么 | bad_fruit_claim | query_order | 0.66 |
| c-8dd2ab4f66 | 我买了六瓶酒 结果快递给我打碎了两瓶 | bad_fruit_claim | query_order | 0.66 |
| c-fc372667d1 | 938625 刚收到，是[姓名x]，但发现带子比车上的薄，希望退货 | bad_fruit_claim | quality_qa | 0.68 |
| c-d7b758d1ae | 你好 这个表带边上小圆圈有点瑕疵 能给补发一个吗 | bad_fruit_claim | chitchat | 0.35 |
| c-43081d758d | [数字x]月买的，我提交了给我关闭了 | bad_fruit_claim | chitchat | 0.35 |
| c-c46d64a138 | 门框边缘有磨损痕迹 | bad_fruit_claim | chitchat | 0.35 |
| c-ad69fcefd8 | 我把有问题的东西邮过去，运费不得给我补偿吗 | bad_fruit_claim | quality_qa | 0.68 |
| c-c626a2a4b1 | 请问需要等多久 | bad_fruit_claim | chitchat | 0.35 |

### few-shot deepseek-chat（1-shot，test）

| id | 问句 | 真值 | 预测 | 置信度 |
| --- | --- | --- | --- | --- |
| c-05781cf7fb | 一点包装都没有了也可以吗?#E-s[数字x] | bad_fruit_claim | chitchat | 0.30 |
| c-8af0afbd03 | 就是要洗的尺寸错了 | bad_fruit_claim | chitchat | 0.30 |
| c-691eb92439 | 2301929 快递员能送货的同时 售后处理这个有问题的么 | bad_fruit_claim | query_order | 0.60 |
| c-43081d758d | [数字x]月买的，我提交了给我关闭了 | bad_fruit_claim | refund_consult | 0.40 |
| c-c46d64a138 | 门框边缘有磨损痕迹 | bad_fruit_claim | chitchat | 0.30 |
| c-ad69fcefd8 | 我把有问题的东西邮过去，运费不得给我补偿吗 | bad_fruit_claim | refund_consult | 0.75 |
| c-c626a2a4b1 | 请问需要等多久 | bad_fruit_claim | query_order | 0.60 |
| c-17573c8368 | 刚才电话跟客服沟通的时候好像说的是xl号 | chitchat | human_handoff | 0.40 |
| c-1d92e41566 | 短信来了再去吗 | chitchat | query_order | 0.60 |
| c-b1a79a26c7 | 就是京东派是吧 | chitchat | query_order | 0.55 |

### local LoRA（test）

| id | 问句 | 真值 | 预测 | 置信度 |
| --- | --- | --- | --- | --- |
| c-05781cf7fb | 一点包装都没有了也可以吗?#E-s[数字x] | bad_fruit_claim | quality_qa | 0.94 |
| c-8af0afbd03 | 就是要洗的尺寸错了 | bad_fruit_claim | chitchat | 0.97 |
| c-691eb92439 | 2301929 快递员能送货的同时 售后处理这个有问题的么 | bad_fruit_claim | quality_qa | 0.86 |
| c-8dd2ab4f66 | 我买了六瓶酒 结果快递给我打碎了两瓶 | bad_fruit_claim | quality_qa | 0.87 |
| c-fc372667d1 | 938625 刚收到，是[姓名x]，但发现带子比车上的薄，希望退货 | bad_fruit_claim | quality_qa | 0.81 |
| c-d7b758d1ae | 你好 这个表带边上小圆圈有点瑕疵 能给补发一个吗 | bad_fruit_claim | chitchat | 0.93 |
| c-43081d758d | [数字x]月买的，我提交了给我关闭了 | bad_fruit_claim | query_order | 0.82 |
| c-c46d64a138 | 门框边缘有磨损痕迹 | bad_fruit_claim | chitchat | 0.90 |
| c-ad69fcefd8 | 我把有问题的东西邮过去，运费不得给我补偿吗 | bad_fruit_claim | refund_consult | 0.98 |
| c-c626a2a4b1 | 请问需要等多久 | bad_fruit_claim | query_order | 0.76 |

## 结论与下一步

- rule-based-v1（test）：macro-F1 0.2013，最弱三类 urge_logistics(F1 0.00/真值 93)、short_weight(F1 0.00/真值 17)、refund_consult(F1 0.00/真值 94)
- few-shot deepseek-chat（1-shot，test）：macro-F1 0.8490，最弱三类 human_handoff(F1 0.69/真值 38)、quality_qa(F1 0.79/真值 51)、urge_logistics(F1 0.79/真值 93)
- few-shot deepseek-chat（1-shot，test）：本次全部命中缓存，P95 时延不可用；要拿真实时延请加 --no-cache 重跑
- few-shot deepseek-chat（1-shot，test）：千次调用 ¥4.21，P95 时延 0ms —— 这两个数字就是本地小模型要打败的目标
- local LoRA（test）：macro-F1 0.7085，最弱三类 human_handoff(F1 0.31/真值 38)、quality_qa(F1 0.53/真值 51)、complaint(F1 0.62/真值 18)
- local LoRA（test）：2 条调用失败（0.3%），这些样本按判错计入指标；超时/限流居多时可调大 INTENT_TIMEOUT 或降低 --concurrency，重跑会自动补上（失败不写缓存）
- 最需要盯的混淆：urge_logistics 被误判为 query_order（21 条），先查 taxonomy 边界定义
- 最需要盯的混淆：human_handoff 被误判为 chitchat（21 条），先查 taxonomy 边界定义
- 最需要盯的混淆：quality_qa 被误判为 chitchat（13 条），先查 taxonomy 边界定义

复现命令：

```bash
python scripts/eval_intent.py --backends rule fewshot local --local-dir data/intent/models/v2 --splits test
```
