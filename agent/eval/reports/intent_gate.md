# 意图版本门禁报告

- 生成时间：2026-10-15 14:10:56
- 基线：data/intent/models/v1
- 候选：data/intent/models/v2
- 判定：❌ 不通过（保留基线）
- 理由：以下指标劣化：macro-F1、human_handoff F1

| 指标 | 基线 | 候选 | Δ | 是否达标 |
| --- | --- | --- | --- | --- |
| macro-F1 | 0.7366 | 0.7085 | -0.0281 | ✗ |
| quality_qa F1 | 0.5155 | 0.5347 | +0.0192 | ✓ |
| chitchat F1 | 0.6010 | 0.6352 | +0.0342 | ✓ |
| human_handoff F1 | 0.6176 | 0.3111 | -0.3065 | ✗ |

门禁口径：macro-F1 **与**关键类（quality_qa / chitchat / human_handoff）都不劣化才放行；
只看 macro-F1 会让「高频类涨分、短板悄悄退化」的版本蒙混过关。

