# 护栏规则

每条规则一个 YAML 条目，文件可以是列表，也可以是 `{version: 1, rules: [...]}`。

## 字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| id | 是 | 小写 kebab-case / 点号，全局唯一 |
| scope | 是 | input / dialog / execution / retrieval / output |
| priority | 否 | 越大越先执行，默认 50 |
| match.event | 是 | user_message / intent / retrieval / tool_call / tool_result / reply / handoff / * |
| match.keywords | 否 | 触发词列表；命中按长度加权打分 |
| match.regex | 否 | 正则，命中加 3 分 |
| match.not_regex | 否 | 反向正则（不匹配才命中），用于“格式不对”的校验 |
| match.conditions | 否 | 对事件 payload 的字段条件（field / op / value，或 value_field 比较两个字段） |
| match.min_score | 否 | 触发词门槛，默认 3.0（一个两字词即可） |
| action | 是 | block / rewrite / handoff / flag |
| reason | 否 | 给 handoff 用的原因，保持和旧代码的 reason 文案一致 |
| message | 视动作 | block / handoff / rewrite 必填；rewrite 支持 `{text}` 占位 |

关键词、正则、条件至少要有一样——不允许“匹配一切”的规则。

## 已接入的位置

| 类别 | 接入点 | 行为 |
| --- | --- | --- |
| 输入 | rails_input 节点（intent 前） | 拦截诱导泄漏；争议词/转人工交给 intent 后路由；PII/OOD 只做标记 |
| 对话 | route_after_claim / decision.decide | loop≥3、澄清达上限→handoff |
| 检索 | faq_node | 无依据→转人工（默认不启用低相关度规则） |
| 执行 | gateway 校验 + rails_execution 节点 | 金额超限、工单号格式异常→handoff |
| 输出 | rails_output 节点（respond 前） | 手机号脱敏、承诺/泄漏文案改写 |

`RAILS_ENABLED=off` 时所有判断回退到迁移前的内置逻辑，便于对照。

## 检查

```bash
python scripts/rail_index.py
python scripts/rail_index.py --json
python scripts/rail_index.py --strict
```
