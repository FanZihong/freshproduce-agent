# 语料碎片拆分报告

- 生成时间：2026-10-12 10:13:06
- 输入文件：calibrated_v1.jsonl（7227 条）
- 信息自足：4701 条（65.0%）
- 依赖上下文：2526 条（35.0%）

## 自足样本类别分布

| 类别 | 条数 |
| --- | --- |
| query_order | 1522 |
| chitchat | 683 |
| refund_consult | 628 |
| urge_logistics | 623 |
| quality_qa | 343 |
| human_handoff | 258 |
| bad_fruit_claim | 211 |
| invoice | 197 |
| complaint | 120 |
| short_weight | 116 |

- 自足样本上的预标注采纳率：84.8%
- 自足样本上的人工一致性：完全一致 84.8%，Cohen's kappa 0.632

## 结论

碎片样本单看没有独立意图，人工给它们打标签属于凭上下文猜测；
应改为：进多轮样本（带 context）或在推理时走澄清追问，而不是进单句分类训练集。
