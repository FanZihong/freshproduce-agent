# Skills 目录

每个子目录是一个技能包，必须包含 `SKILL.md`：

```
data/skills/
├─ refund-policy/
│  └─ SKILL.md
├─ compensation/
│  └─ SKILL.md
└─ ...
```

## 最小格式

```markdown
---
name: refund-policy
description: 一句话说明这个技能解决什么问题（用于路由，8–300 字）
when_to_use: 用户问退款、退货、到账时间时
version: 0.1.0
tags: [faq, policy]
triggers: [退款, 退货, 到账]
references: []
---

正文：命中后才会被读进上下文，写"先判断什么、按什么步骤做、什么情况转人工"。
```

## 校验规则

- 目录名必须和 `name` 一致，`name` 只能是小写 kebab-case；
- `description` 和 `when_to_use` 必填，它们是路由依据；
- `triggers`（可选）是路由触发词列表，命中得越具体分数越高；
- 正文不能为空，单文件不超过 64 KB、正文不超过 20000 字；
- `references` 里的文件必须真实存在、且不能跳出技能目录（禁止 `../`）；
- 坏包默认跳过并记录原因，`--strict` 时整体失败。

## 检查

```bash
python scripts/skill_index.py          # 打印目录
python scripts/skill_index.py --json   # 给接口/脚本用
```

## references/（第二层加载）

- 只有 `SKILL.md` 的 `references` 列表里声明过的文件才允许读，不能现场拼路径；
- 文件必须放在技能目录的 `references/` 下，禁止绝对路径和 `../`；
- 单文件 ≤ 256 KB、每个技能最多 5 个、合计 ≤ 512 KB；
- 只有用户追问“具体第几条 / 细则 / 原文”时才加载，注入预算由 `SKILL_DETAIL_TOKENS` 控制。
