# 多意图路由策略对比报告

- 生成时间：2026-10-15 22:04:31
- 后端：local-lora
- 金标集：multi_intent.jsonl（50 条多意图问句）
- 策略：priority（优先动作型，取 routing_priority 最靠前的分支）/ parallel（两个分支都执行，回复合并）

## 结果

| 后端 / 策略 | 主分支命中率 | 次诉求覆盖率 | 双诉求覆盖率 | 多标签完全一致率 |
| --- | --- | --- | --- | --- |
| local-lora · priority | 0.8400 | 0.3800 | 0.2400 | 0.6200 |
| local-lora · parallel | 0.8400 | 0.9000 | 0.7600 | 0.6200 |

> 识别指标（macro-F1 / 多标签 F1）与策略无关：策略只决定「走哪些分支」，不改变模型输出的标签集合。所以策略要靠覆盖指标比，而不是靠 F1。

## 逐条明细（预测 vs 真值 + 分支覆盖）

| id | 问句 | 真值 | 预测 | priority 分支 | parallel 分支 |
| --- | --- | --- | --- | --- | --- |
| mi-01 | 车厘子坏了一半，而且三天了还没发货 | bad_fruit_claim/urge_logistics | bad_fruit_claim/urge_logistics | claim | claim→order |
| mi-02 | 草莓发霉了，顺便帮我催一下快递 | bad_fruit_claim/urge_logistics | bad_fruit_claim/urge_logistics | claim | claim→order |
| mi-03 | 蓝莓压坏了不少，另外包裹怎么还没到 | bad_fruit_claim/urge_logistics | bad_fruit_claim/urge_logistics | claim | claim→order |
| mi-04 | 苹果烂了两个，还有什么时候能送到 | bad_fruit_claim/urge_logistics | bad_fruit_claim/urge_logistics | claim | claim→order |
| mi-05 | 到货的芒果全坏了，麻烦催下物流 | bad_fruit_claim/urge_logistics | urge_logistics | order | order |
| mi-06 | 这个订单的菜坏了，还有多久能送到 | bad_fruit_claim/urge_logistics | urge_logistics | order | order |
| mi-07 | 番茄压烂了，同时也想问问我的另一单发没发 | bad_fruit_claim/urge_logistics | bad_fruit_claim/query_order | claim | claim→order |
| mi-08 | 牛油果变质了，快递也一直没动 | bad_fruit_claim/urge_logistics | bad_fruit_claim/urge_logistics | claim | claim→order |
| mi-09 | 鸡蛋碎了三个，另外能不能帮我催下配送 | bad_fruit_claim/urge_logistics | bad_fruit_claim/urge_logistics | claim | claim→order |
| mi-10 | 青菜到货就蔫了，顺便问下什么时候派送 | bad_fruit_claim/urge_logistics | bad_fruit_claim/urge_logistics | claim | claim→order |
| mi-11 | 少了一盒不说，退款还要等多久 | short_weight/refund_consult | short_weight/refund_consult | claim | claim→faq |
| mi-12 | 说好两斤只有一斤半，退款一般几天到账 | short_weight/refund_consult | refund_consult | faq | faq |
| mi-13 | 重量比标注少了一截，另外退款失败怎么办 | short_weight/refund_consult | short_weight/refund_consult | claim | claim→faq |
| mi-14 | 数量不对少了两个，退款是原路返回吗 | short_weight/refund_consult | short_weight/refund_consult | claim | claim→faq |
| mi-15 | 缺斤少两了，想问下退款时效是多久 | short_weight/refund_consult | refund_consult | faq | faq |
| mi-16 | 少称了半斤，顺便问下退款流程 | short_weight/refund_consult | short_weight/refund_consult | claim | claim→faq |
| mi-17 | 你们客服太差了，我要投诉，顺便帮我转人工 | human_handoff/complaint | complaint/human_handoff | handoff | handoff |
| mi-18 | 再这样我就给差评了，麻烦转人工处理 | human_handoff/complaint | complaint | handoff | handoff |
| mi-19 | 我要曝光你们，叫个真人来跟我谈 | human_handoff/complaint | human_handoff | handoff | handoff |
| mi-20 | 这种服务我要投诉到底，转人工 | human_handoff/complaint | complaint | handoff | handoff |
| mi-21 | 对你们很不满意，不跟机器人说了，找人工 | human_handoff/complaint | complaint | handoff | handoff |
| mi-22 | 准备给差评，别机器人了叫真人 | human_handoff/complaint | complaint | handoff | handoff |
| mi-23 | 投诉你们客服，顺便转人工 | human_handoff/complaint | complaint/human_handoff | handoff | handoff |
| mi-24 | 要求处罚商家，同时帮我转人工跟进 | human_handoff/complaint | complaint/human_handoff | handoff | handoff |
| mi-25 | 我的订单到哪了，另外发票怎么开 | query_order/invoice | query_order/invoice | order | order→faq |
| mi-26 | 帮我查一下物流，同时想开一张电子发票 | query_order/invoice | query_order/invoice | order | order→faq |
| mi-27 | 这个订单签收了吗，顺便问下抬头能不能改 | query_order/invoice | query_order | order | order |
| mi-28 | 货什么时候到，还有发票能补开吗 | query_order/invoice | urge_logistics/invoice | order | order→faq |
| mi-29 | 想问下订单进度，另外税号写错了要不要重开 | query_order/invoice | query_order/invoice | order | order→faq |
| mi-30 | 查下我的订单，顺便把发票寄给我 | query_order/invoice | query_order/invoice | order | order→faq |
| mi-31 | 车厘子坏了能赔多少，退款又要等几天 | bad_fruit_claim/refund_consult | bad_fruit_claim/refund_consult | claim | claim→faq |
| mi-32 | 草莓烂了要理赔，另外退款失败了怎么办 | bad_fruit_claim/refund_consult | bad_fruit_claim/refund_consult | claim | claim→faq |
| mi-33 | 这批货变质了，想问下退款到账时间 | bad_fruit_claim/refund_consult | refund_consult | faq | faq |
| mi-34 | 里面有异物怎么赔，顺便问问退款进度 | bad_fruit_claim/refund_consult | bad_fruit_claim/refund_consult | claim | claim→faq |
| mi-35 | 果子上有霉点，赔付之后钱几天到 | bad_fruit_claim/refund_consult | refund_consult | faq | faq |
| mi-36 | 东西坏了要赔付，退款是原路返回吗 | bad_fruit_claim/refund_consult | bad_fruit_claim/refund_consult | claim | claim→faq |
| mi-37 | 车厘子怎么保存，另外帮我把发票开了 | invoice/quality_qa | quality_qa/invoice | faq | faq |
| mi-38 | 冷冻肉能放多久，顺便问下能不能开发票 | invoice/quality_qa | quality_qa/invoice | faq | faq |
| mi-39 | 生鲜可以无理由退货吗，还有发票抬头写错了 | invoice/quality_qa | quality_qa/invoice | faq | faq |
| mi-40 | 这个能放几天，另外要开个发票 | invoice/quality_qa | quality_qa/invoice | faq | faq |
| mi-41 | 催一下发货，另外发票什么时候能开出来 | urge_logistics/invoice | urge_logistics/invoice | order | order→faq |
| mi-42 | 都两天了还不发，顺便把发票开了 | urge_logistics/invoice | urge_logistics/invoice | order | order→faq |
| mi-43 | 帮我催下快递，另外抬头要改成公司名 | urge_logistics/invoice | urge_logistics/human_handoff | handoff | handoff→order |
| mi-44 | 快点发货吧，还有发票能不能补开 | urge_logistics/invoice | urge_logistics/invoice | order | order→faq |
| mi-45 | 投诉商家缺斤少两，必须给个说法 | complaint/short_weight | complaint | handoff | handoff |
| mi-46 | 我要投诉你们发的货全是坏的 | complaint/bad_fruit_claim | bad_fruit_claim | claim | claim |
| mi-47 | 太差了我要投诉，而且这单草莓都烂了 | complaint/bad_fruit_claim | complaint | handoff | handoff |
| mi-48 | 要求处罚商家，货也是烂的 | complaint/bad_fruit_claim | complaint | handoff | handoff |
| mi-49 | 这服务我必须投诉，另外东西确实坏了要赔 | complaint/bad_fruit_claim | complaint/bad_fruit_claim | handoff | handoff→claim |
| mi-50 | 东西坏了要赔，同时我要投诉你们的配送 | complaint/bad_fruit_claim | bad_fruit_claim/complaint | handoff | handoff→claim |

## 验收判定

- 识别层：多标签完全一致率 0.6200、多标签 macro-F1 0.7784（这部分与策略无关，是模型能力上限）
- priority 双诉求覆盖率 0.2400，parallel 0.7600（提升 +0.5200）
- parallel 把双诉求覆盖率从 0.2400 提到 0.7600（+0.5200）：代价是回复更长、要跑两个分支。
- 策略建议：priority 适合「动作型 + 咨询型」组合（先给赔付结论，次诉求补一句入口）；parallel 适合「两个动作型」组合（都要真做事）。上线前以本报告的数字为准。

复现命令：

```bash
python scripts/eval_multi_intent.py --backend local --report /mnt/d/生鲜售后agent/agent/eval/reports/intent_multi.md --json /mnt/d/生鲜售后agent/agent/eval/reports/intent_multi.json
```

