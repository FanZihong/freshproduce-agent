"""意图识别层。

- `taxonomy`：10 类意图体系 + 标签到路由分支的映射（单一事实来源）；
- `classifier`：统一入口，按 INTENT_BACKEND 切换 rule / fewshot / local 后端；
- `baseline`：大模型 few-shot 基线（后续 LoRA 微调的对照物）；
- `slot_extractor`：订单号、比例等槽位的抽取规则。
"""
