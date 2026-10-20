"""知识库反哺脚本的测试：无据判定、小节定位、条目插入、幂等与候选合并。"""

from __future__ import annotations

import contextlib
import io
import json
import sys

from scripts.update_knowledge import (
    append_new_section,
    candidates_from_pool,
    candidates_from_traces,
    entry_exists,
    insert_entry,
    main,
    merge_candidates,
    needs_knowledge,
    next_index_in_section,
    normalize_question,
    render_entry,
    section_bounds,
    suggest_target,
)

SAMPLE_MD = """# 常见问题 FAQ

## 1. 订单与物流

1. 问：我的订单到哪了？答：提供订单号即可查询。
2. 问：多久能发货？答：一般 24 小时内。

## 2. 退款与发票

1. 问：退款多久到账？答：1 至 3 个工作日。
"""


def test_needs_knowledge_matches_no_evidence_replies():
    assert needs_knowledge("我查了知识库，没有找到可靠的依据（知识库中没有「几天」相关内容）。") is True
    assert needs_knowledge(None, "知识库无据：知识库中没有「山竹」相关内容") is True
    assert needs_knowledge("订单 2026093000123 已签收") is False
    assert needs_knowledge(None, "") is False


def test_suggest_target_by_keywords_with_default():
    assert suggest_target("生鲜能开发票吗") == ("常见问题FAQ.md", "3. 退款与发票")
    assert suggest_target("草莓坏了一半怎么赔") == ("常见问题FAQ.md", "2. 品质与赔付")
    assert suggest_target("催单有补偿吗") == ("物流时效说明.md", "4. 催单与补偿")
    assert suggest_target("随便问一句不知道归哪") == ("常见问题FAQ.md", "")


def test_normalize_question_and_render_entry():
    assert normalize_question("生鲜能开发票吗") == "生鲜能开发票吗？"
    assert normalize_question("生鲜能开发票吗？") == "生鲜能开发票吗？"
    assert normalize_question("  ") == ""
    assert render_entry(5, "生鲜能开发票吗", "可以，订单详情页申请。") == (
        "5. 问：生鲜能开发票吗？答：可以，订单详情页申请。"
    )


def test_section_bounds_and_next_index():
    bounds = section_bounds(SAMPLE_MD, "1. 订单与物流")
    assert bounds is not None
    body = SAMPLE_MD[bounds[0] : bounds[1]]
    assert "我的订单到哪了" in body and "退款多久到账" not in body
    assert section_bounds(SAMPLE_MD, "9. 不存在") is None
    assert next_index_in_section(SAMPLE_MD, "1. 订单与物流") == 3
    assert next_index_in_section(SAMPLE_MD, "2. 退款与发票") == 2
    assert next_index_in_section(SAMPLE_MD, "9. 不存在") == 1


def test_insert_entry_keeps_markdown_structure():
    entry = render_entry(3, "可以改地址吗", "未发货可以改，已发货需和快递协商。")
    updated, ok = insert_entry(SAMPLE_MD, "1. 订单与物流", entry)
    assert ok is True
    assert "3. 问：可以改地址吗？" in updated
    first = updated.index("## 1. 订单与物流")
    second = updated.index("## 2. 退款与发票")
    assert first < updated.index("可以改地址吗") < second
    assert "\n\n## 2. 退款与发票" in updated
    assert updated.count("## 2. 退款与发票") == 1


def test_insert_entry_at_end_of_file():
    md = "# 标题\n\n## 1. 小节\n\n1. 问：a？答：b。\n"
    updated, ok = insert_entry(md, "1. 小节", "2. 问：c？答：d。")
    assert ok is True and updated.rstrip().endswith("2. 问：c？答：d。")


def test_insert_entry_reports_missing_section():
    updated, ok = insert_entry(SAMPLE_MD, "9. 不存在", "1. 问：x？答：y。")
    assert ok is False and updated == SAMPLE_MD
    assert "## 补充问答" in append_new_section(SAMPLE_MD, "补充问答", "1. 问：x？答：y。")


def test_entry_exists_is_idempotent_check():
    assert entry_exists(SAMPLE_MD, "退款多久到账") is True
    assert entry_exists(SAMPLE_MD, "退款多久到账？") is True
    assert entry_exists(SAMPLE_MD, "生鲜能开发票吗") is False
    assert entry_exists(SAMPLE_MD, "  ") is False


def test_candidates_from_pool_only_keeps_no_evidence_rows():
    rows = [
        {
            "id": "trace-a-t1",
            "text": "退款一般几天到账",
            "observed": {
                "reply": "我查了知识库，没有找到可靠的依据（…）",
                "handoff_reason": "知识库无据",
            },
        },
        {"id": "trace-b-t1", "text": "我的订单到哪了", "observed": {"reply": "已签收"}},
    ]
    candidates = candidates_from_pool(rows)
    assert len(candidates) == 1
    got = candidates[0]
    assert got["id"] == "kb-trace-a-t1"
    assert got["doc"] == "常见问题FAQ.md" and got["section"] == "3. 退款与发票"
    assert got["answer"] == "" and got["status"] == "pending"


def test_candidates_from_traces_reads_handoff_reason():
    records = [
        {
            "session_id": "cli-1",
            "turn": 2,
            "user_input": "山竹怎么保存",
            "reply": "我查了知识库，没有找到可靠的依据。",
            "handoff_reason": "知识库无据：知识库中没有「山竹」相关内容",
        }
    ]
    candidates = candidates_from_traces(records)
    assert len(candidates) == 1
    assert candidates[0]["source"] == "trace:cli-1#2"
    assert candidates[0]["question"] == "山竹怎么保存"


def test_merge_candidates_keeps_filled_answers():
    existing = [
        {"question": "生鲜能开发票吗？", "answer": "可以，订单详情页申请。", "doc": "常见问题FAQ.md"}
    ]
    fresh = [
        {"question": "生鲜能开发票吗", "answer": "", "doc": "常见问题FAQ.md"},
        {"question": "山竹怎么保存", "answer": "", "doc": "常见问题FAQ.md"},
    ]
    merged, added = merge_candidates(existing, fresh)
    by_q = {row["question"]: row for row in merged}
    assert added == 1
    assert by_q["生鲜能开发票吗？"]["answer"] == "可以，订单详情页申请。"
    assert "山竹怎么保存" in by_q


# ---------------- 落点策略：默认追加到末尾（chunk id 稳定） ----------------


def _write_candidates(path, rows) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )


def test_apply_appends_to_file_end_by_default(tmp_path, monkeypatch):
    """默认追加到文件末尾：既有的小节与编号一个都不动（黄金集的 chunk id 才不会错位）。"""
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    faq = knowledge / "常见问题FAQ.md"
    faq.write_text(SAMPLE_MD, encoding="utf-8")
    candidates = tmp_path / "kb_candidates.jsonl"
    _write_candidates(
        candidates,
        [
            {
                "id": "kb-1",
                "question": "生鲜能开发票吗",
                "answer": "可以，订单详情页申请。",
                "doc": "常见问题FAQ.md",
                "section": "2. 退款与发票",
                "status": "labeled",
            }
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "update_knowledge.py", "--apply", "--no-rebuild", "--no-report",
            "--candidates", str(candidates), "--knowledge-dir", str(knowledge),
        ],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        assert main() == 0

    text = faq.read_text(encoding="utf-8")
    # 新条目落在末尾的「补充问答」，没有被塞进"2. 退款与发票"
    assert text.rstrip().splitlines()[-1].startswith("1. 问：生鲜能开发票吗？")
    assert "## 补充问答" in text
    refund_block = text[text.index("## 2. 退款与发票") : text.index("## 补充问答")]
    assert "生鲜能开发票吗" not in refund_block
    # 原有内容一字未动
    assert text.startswith(SAMPLE_MD.rstrip("\n"))
    assert (knowledge / "常见问题FAQ.md.bak").exists()


def test_apply_in_section_mode_moves_ids_and_says_so(tmp_path, monkeypatch):
    """显式要求按主题插入时，必须提醒"后面的 chunk id 会整体后移"。"""
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "常见问题FAQ.md").write_text(SAMPLE_MD, encoding="utf-8")
    candidates = tmp_path / "kb_candidates.jsonl"
    _write_candidates(
        candidates,
        [
            {
                "id": "kb-2",
                "question": "发票抬头写错了怎么办",
                "answer": "可申请作废重开。",
                "doc": "常见问题FAQ.md",
                "section": "2. 退款与发票",
                "status": "labeled",
            }
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "update_knowledge.py", "--apply", "--no-rebuild", "--no-report",
            "--append-mode", "section",
            "--candidates", str(candidates), "--knowledge-dir", str(knowledge),
        ],
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert main() == 0
    captured = buffer.getvalue()
    assert "chunk id 会整体后移" in captured
    text = (knowledge / "常见问题FAQ.md").read_text(encoding="utf-8")
    # 插进了中间小节，而且没有额外开"补充问答"
    assert text.index("发票抬头写错了怎么办") > text.index("## 2. 退款与发票")
    assert "## 补充问答" not in text


def test_apply_is_idempotent_on_second_run(tmp_path, monkeypatch):
    """同一条问题跑两次不会写重复：第二次直接跳过。"""
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "常见问题FAQ.md").write_text(SAMPLE_MD, encoding="utf-8")
    candidates = tmp_path / "kb_candidates.jsonl"
    _write_candidates(
        candidates,
        [
            {
                "id": "kb-3",
                "question": "临期商品能退吗",
                "answer": "不支持以快到期为由退货。",
                "doc": "常见问题FAQ.md",
                "section": "",
                "status": "labeled",
            }
        ],
    )
    argv = [
        "update_knowledge.py", "--apply", "--no-rebuild", "--no-report",
        "--candidates", str(candidates), "--knowledge-dir", str(knowledge),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with contextlib.redirect_stdout(io.StringIO()):
        assert main() == 0
    first = (knowledge / "常见问题FAQ.md").read_text(encoding="utf-8")
    assert first.count("临期商品能退吗") == 1
    with contextlib.redirect_stdout(io.StringIO()):
        assert main() == 0
    second = (knowledge / "常见问题FAQ.md").read_text(encoding="utf-8")
    assert second == first
