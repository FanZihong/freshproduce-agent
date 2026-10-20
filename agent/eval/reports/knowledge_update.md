# 知识库反哺记录

- 生成时间：2026-10-16 12:05:27
- 候选文件：/mnt/d/生鲜售后agent/agent/eval/datasets/kb_candidates.jsonl
- 本次写入 6 条（待填 0 条）

- kb-probe-linqi：追加到 售后政策总则.md 末尾（新建小节「补充问答」，第 1 条）
- kb-probe-yushou：追加到 售后政策总则.md / 补充问答（第 2 条）
- kb-probe-gongdan：追加到 售后政策总则.md / 补充问答（第 3 条）
- kb-synonym-chaifeng：追加到 售后政策总则.md / 补充问答（第 4 条）
- kb-probe-zengpin：追加到 赔付标准表.md 末尾（新建小节「补充问答」，第 1 条）
- kb-probe-qiekai：追加到 赔付标准表.md / 补充问答（第 2 条）

## 下一步

```bash
python scripts/regression.py --only retrieval
```

验收：检索指标不劣化；补了条目的那几个问题，`search_faq` 应该从「无据」变成「有据」。

