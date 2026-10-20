# 域外检测评测报告

- 生成时间：2026-10-15 22:00:35
- 后端：local-lora
- 域外测试集：ood_golden.jsonl（50 条）
- 误杀率统计集：dataset_v1_balanced 里真值不是 chitchat 的样本（599 条）
- 信号阈值：置信度 < 0.5、相似度 < 0.35、至少 2 路信号
- 语义相似度信号：启用

## 结果

| 口径 | 域外召回 | 已知类误杀率 | 误杀条数 |
| --- | --- | --- | --- |
| 纯模型（local-lora） | 0.9400 | 0.0684 | 41/599 |
| 信号组合（不用规则否决） | 0.8800 | 0.0534 | 32/599 |
| 信号组合 + 规则否决 ★ | 0.8800 | 0.0501 | 30/599 |

## 验收判定

- 纯模型域外召回 0.9400、误杀率 0.0684（模型自己就够好）
- 三路信号：域外召回 0.8800（≥0.9？✗）；已知类误杀率 0.0501（<0.05？✗）
- 参照：不用规则否决时 召回 0.8800、误杀率 0.0534（规则否决的代价是召回 −0.0000，收益是误杀 −0.0033）
- 误杀样本的典型原因：模型自身判为 chitchat×30
- 漏判原因分布：仅命中 1/4 路信号×3、仅命中 1/2 路信号×2、命中领域词×1

## 漏判的域外问句（6 条全列）

| id | 问句 | 模型标签 | 置信度 | 未判为域外的原因 |
| --- | --- | --- | --- | --- |
| ood-03 | 北京现在多少度 | （空） | 0.000 | 仅命中 1/2 路信号（需要 2 路），判域内 |
| ood-15 | 手机充不进电是哪里坏了 | ['chitchat'] | 0.981 | 命中领域词（坏），强制判域内 |
| ood-16 | 衣服尺码买小了能换吗 | ['quality_qa'] | 0.920 | 仅命中 1/4 路信号（需要 2 路），判域内 |
| ood-26 | 帮我给老板写封邮件 | ['invoice'] | 0.610 | 仅命中 1/4 路信号（需要 2 路），判域内 |
| ood-38 | 我这样算不算违法 | ['complaint'] | 0.906 | 仅命中 1/4 路信号（需要 2 路），判域内 |
| ood-39 | 个人所得税怎么计算 | （空） | 0.000 | 仅命中 1/2 路信号（需要 2 路），判域内 |

## 被误杀的已知类问句（30 条全列）

> 逐条复核这些样本的**真值是否可信**：JDDC 带进来的其他品类噪音、缺上下文的碎片，被判域外其实是正确行为。把存疑的 id 写进 suspects 清单，重跑时用 --suspects 引用，报告会多出一行「剔除标签存疑后」的指标。

| id | 问句 | 真值 | 模型标签 | 判为域外的原因 |
| --- | --- | --- | --- | --- |
| c-8af0afbd03 | 就是要洗的尺寸错了 | ['bad_fruit_claim'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-f3ce7c6d50 | 希望不要再有短信了 | ['complaint'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-a1155fa2d9 | 电话一直没人接 | ['human_handoff'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-e828dd64ff | 你们自己看看吧 | ['human_handoff'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-bcb2f2309d | 你这又了解不到 | ['human_handoff'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-b912259df5 | 什么意思?我现在要怎么做呢 | ['human_handoff'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-00e3ca040a | 问题是为什么电话也打不通 | ['human_handoff'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-0622d1fa10 | 我原来的手机号不用了 收不到验证码了 | ['human_handoff'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-52f8bf419e | 改成 怀远县淝南镇双沟 | ['invoice'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-b54179a505 | 我现在炒菜还有，感觉没去干净 | ['quality_qa'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-b69a142db6 | 一年这个周期是什么时候开始计算 | ['quality_qa'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-49aa5f7f86 | 所有都在，就是找不到说明书了 | ['quality_qa'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-8a743c50f1 | 现在基本上配件都换了 | ['quality_qa'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-5e5008e85e | 我记得P会员好像次数会多一点 | ['quality_qa'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-1925ae9020 | 我下单看到的是赠一个滤芯 | ['quality_qa'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-de8e65bd90 | 用e卡付款食品的话，有没有京享值呢 | ['quality_qa'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-5e2c1abbc1 | 不是一年换新吗 | ['quality_qa'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-d39dd5b73f | 我看上面还没有刷新 | ['query_order'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-4be289d2a0 | 不是有个网址可以查吗 | ['query_order'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-c59fc5d142 | 微信端没办法查到吗 | ['query_order'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-893253c27d | 我刚才拍了一个手机 | ['query_order'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-bcc11b406c | 你好，你帮我看一下，我买了两个 | ['query_order'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-b2024d7aff | 你刚才说让我拒收 | ['query_order'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-f2a8fa2a86 | 好，那我那个周围哪块有 | ['query_order'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-a266f62521 | 可刚刚收到师傅电话 | ['query_order'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-fffadfe0be | 其他的，不是这个账号买的 | ['query_order'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-31b862d6dd | 可以给现金额吗 | ['refund_consult'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-c432fd2940 | 我怎么转入银行卡 | ['refund_consult'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-36aaa47a69 | 没有其他费用了吧 | ['refund_consult'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |
| c-043850c41b | 但是我还想要货 | ['urge_logistics'] | ['chitchat'] | 模型自身判为 chitchat（第一路信号） |

复现命令：

```bash
python scripts/eval_ood.py --backend local --with-embedder --report /mnt/d/生鲜售后agent/agent/eval/reports/intent_ood.md --json /mnt/d/生鲜售后agent/agent/eval/reports/intent_ood.json
```

