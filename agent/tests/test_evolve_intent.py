"""自进化闭环测试：badcase 挖掘、训练数据合并、评测门禁判定。"""

from __future__ import annotations

import json
from pathlib import Path

from agent.intent.taxonomy import load_taxonomy
from scripts.eval_intent import row_cache_key
from scripts.evolve_intent import (
    build_label_index,
    build_extra_training_rows,
    default_split,
    evaluate_gate,
    find_row,
    format_confusion_counts,
    is_pool_row_labelled,
    merge_training_rows,
    pool_badcases,
    resolve_expected_label,
    select_badcases,
)


def make_row(index: int, text: str, labels: list[str], context=None) -> dict:
    return {
        "id": f"r{index}",
        "text": text,
        "context": context,
        "labels": labels,
        "source": "external:chat.txt",
    }


def test_select_badcases_picks_only_wrong_predictions():
    rows = [
        make_row(1, "车厘子怎么保存", ["quality_qa"]),
        make_row(2, "草莓坏了一半", ["bad_fruit_claim"]),
        make_row(3, "少了半斤", ["short_weight"]),
    ]
    namespace = "data/intent/models/v1"
    cache = {
        row_cache_key("车厘子怎么保存", None, namespace): {"labels": ["quality_qa"], "confidence": 0.9},
        row_cache_key("草莓坏了一半", None, namespace): {"labels": ["chitchat"], "confidence": 0.8},
        row_cache_key("少了半斤", None, namespace): {"labels": ["short_weight"], "confidence": 0.7},
    }
    badcases = select_badcases(rows, cache, namespace, set(load_taxonomy().keys))

    assert [case["id"] for case in badcases] == ["r2"]          # 只挑判错的
    assert badcases[0]["labels"] == ["bad_fruit_claim"]          # 真值保留
    assert badcases[0]["predicted"] == ["chitchat"]              # 错判也记下来，便于复盘
    assert badcases[0]["source"].startswith("badcase:")


def test_select_badcases_ignores_rows_without_cache():
    rows = [make_row(1, "怎么开发票", ["invoice"])]
    assert select_badcases(rows, {}, "ns", set(load_taxonomy().keys)) == []


def test_select_badcases_normalizes_aliases():
    """规则版的旧标签（knowledge_qa）要和体系标签（quality_qa）对齐后再比，否则会误判成错例。"""
    rows = [make_row(1, "车厘子怎么保存", ["quality_qa"])]
    namespace = "ns"
    cache = {row_cache_key("车厘子怎么保存", None, namespace): {"labels": ["knowledge_qa"]}}
    assert select_badcases(rows, cache, namespace, set(load_taxonomy().keys)) == []


def test_build_extra_training_rows_uses_gold_labels():
    badcases = [
        {
            "id": "r2",
            "text": "草莓坏了一半",
            "context": None,
            "labels": ["bad_fruit_claim"],
            "predicted": ["chitchat"],
            "confidence": 0.8,
            "source": "badcase:external",
        }
    ]
    rows = build_extra_training_rows(badcases, load_taxonomy())
    assert len(rows) == 1
    record = rows[0]
    assert record["target"] == "主:bad_fruit_claim"       # 训练目标是真值，不是模型的错判
    assert [message["role"] for message in record["messages"]] == ["system", "user", "assistant"]
    assert record["source"].startswith("badcase:")


def test_merge_training_rows_dedupes_by_text():
    base = [{"id": "a", "text": "怎么开发票", "labels": ["invoice"]}]
    extra = [
        {"id": "b", "text": "怎么开发票", "labels": ["invoice"]},   # 与基础集重复
        {"id": "c", "text": "草莓坏了一半", "labels": ["bad_fruit_claim"]},
    ]
    merged, added = merge_training_rows(base, extra)
    assert added == 1
    assert len(merged) == 2
    assert merged[-1]["id"] == "c"


def test_format_confusion_counts_renders_nested_tuples():
    """回归测试：Counter 的键是"元组套元组"，直接 join 会 TypeError。"""
    from collections import Counter

    counter = Counter(
        {
            (("bad_fruit_claim",), ("chitchat",)): 12,
            (("query_order",), ("urge_logistics",)): 25,
            (("quality_qa",), ()): 3,
        }
    )
    text = format_confusion_counts(counter, top=2)
    assert "query_order→urge_logistics×25" in text
    assert "bad_fruit_claim→chitchat×12" in text
    assert text.index("×25") < text.index("×12")      # 按频次降序
    assert "（空）" in format_confusion_counts(counter, top=3)


def make_metrics(macro_f1: float, **per_class) -> dict:
    return {
        "display": "local LoRA（test）",
        "macro_f1": macro_f1,
        "per_class": {
            label: {"f1": value} for label, value in per_class.items()
        },
    }


def test_gate_passes_when_nothing_degrades():
    baseline = make_metrics(0.7366, quality_qa=0.52, chitchat=0.60, human_handoff=0.62)
    candidate = make_metrics(0.78, quality_qa=0.70, chitchat=0.66, human_handoff=0.64)
    gate = evaluate_gate(baseline, candidate)
    assert gate["passed"] is True
    assert "不劣化" in gate["reason"]
    assert len(gate["rows"]) == 4          # macro-F1 + 3 个关键类


def test_gate_fails_when_key_class_degrades_even_if_macro_f1_improves():
    """回归保护：整体涨分但短板退化，必须拦下来。"""
    baseline = make_metrics(0.7366, quality_qa=0.52, chitchat=0.60, human_handoff=0.62)
    candidate = make_metrics(0.80, quality_qa=0.48, chitchat=0.61, human_handoff=0.63)
    gate = evaluate_gate(baseline, candidate)
    assert gate["passed"] is False
    assert "quality_qa" in gate["reason"]


def test_gate_tolerance_allows_small_regression():
    baseline = make_metrics(0.7366, quality_qa=0.52, chitchat=0.60, human_handoff=0.62)
    candidate = make_metrics(0.7350, quality_qa=0.519, chitchat=0.60, human_handoff=0.62)
    assert evaluate_gate(baseline, candidate, tolerance=0.0)["passed"] is False
    assert evaluate_gate(baseline, candidate, tolerance=0.005)["passed"] is True


def test_gate_reports_missing_rows():
    gate = evaluate_gate({}, {})
    assert gate["passed"] is False
    assert "缺少可比对" in gate["reason"]


def test_find_row_matches_backend_split_and_dir():
    payload = {
        "backends": [
            {"backend": "rule", "split": "test", "local_dir": None},
            {"backend": "local", "split": "test", "local_dir": "data/intent/models/v1"},
            {"backend": "local", "split": "test", "local_dir": "data/intent/models/v3"},
            {"backend": "local", "split": "dev", "local_dir": "data/intent/models/v1"},
        ]
    }
    assert find_row(payload, "data/intent/models/v3")["local_dir"].endswith("v3")
    assert find_row(payload, "data/intent/models/v1", split="dev")["split"] == "dev"
    assert find_row(payload, "data/intent/models/v9") is None


def test_default_split_differs_between_export_and_gate():
    """回归测试：两个命令的默认集合不同，混用会造成"门禁找不到可比对的行"。

    - 导出 badcase 默认 dev（不能用 test，否则数据泄漏）；
    - 门禁默认 test（两个版本都评过的验收口径）。
    """
    assert default_split(None, for_gate=False) == "dev"
    assert default_split(None, for_gate=True) == "test"
    assert default_split("dev", for_gate=True) == "dev"      # 显式指定优先
    assert default_split("test", for_gate=False) == "test"


# ---------------- trace 池：人工自然语言 -> 体系 key ----------------


def test_resolve_expected_label_matches_name_alias_and_keywords():
    taxonomy = load_taxonomy()
    assert resolve_expected_label("转人工", taxonomy)[0] == "human_handoff"       # 中文名精确匹配
    assert resolve_expected_label("human_handoff", taxonomy)[0] == "human_handoff"
    key, how = resolve_expected_label("域外回答拒绝", taxonomy)
    assert key == "chitchat" and "关键词" in how                                  # 兜底命中
    assert resolve_expected_label("  ", taxonomy) == (None, "")
    assert resolve_expected_label("我今天想吃火锅", taxonomy) == (None, "")        # 不猜


def test_build_label_index_covers_keys_names_and_aliases():
    taxonomy = load_taxonomy()
    index = build_label_index(taxonomy)
    for key in taxonomy.keys:
        assert index[key] == key
        assert index[taxonomy.name_of(key)] == key


def make_pool_row(**overrides) -> dict:
    row = {
        "id": "trace-s-1-t1",
        "text": "你们客服太差了，我要投诉",
        "expected_intent": "转人工",
        "expected_action": "转人工处理",
        "note": "用户明确要投诉",
        "status": "labeled",
        "observed": {"intent": "human_handoff", "confidence": 0.78, "reply": "已转接人工"},
    }
    row.update(overrides)
    return row


def test_pool_badcases_turns_labels_into_keys_and_flags_agreement():
    taxonomy = load_taxonomy()
    rows = [
        make_pool_row(),                                                     # 人工与模型一致
        make_pool_row(
            id="trace-s-2-t1",
            text="你们能帮我写一段 python 代码吗",
            expected_intent="域外回答拒绝",
            observed={"intent": "query_order", "confidence": 0.35, "reply": "..."},
        ),                                                                   # 模型判错
        make_pool_row(id="trace-s-3-t1", status="pending", expected_intent=""),
        make_pool_row(id="trace-s-4-t1", expected_intent="我今天想吃火锅"),   # 映射不了
    ]
    usable, unmapped = pool_badcases(rows, taxonomy)
    assert [case["id"] for case in usable] == ["trace-s-1-t1", "trace-s-2-t1"]
    assert usable[0]["labels"] == ["human_handoff"] and usable[0]["agrees_with_model"] is True
    assert usable[1]["labels"] == ["chitchat"] and usable[1]["agrees_with_model"] is False
    assert [row["id"] for row in unmapped] == ["trace-s-4-t1"]
    # 没标注的行默认不进（也不进未映射清单）
    assert "trace-s-3-t1" not in [case["id"] for case in usable] + [row["id"] for row in unmapped]


def test_pool_badcases_can_feed_training_rows():
    taxonomy = load_taxonomy()
    usable, _ = pool_badcases([make_pool_row()], taxonomy)
    records = build_extra_training_rows(usable, taxonomy)
    assert len(records) == 1
    assert "human_handoff" in json.dumps(records[0], ensure_ascii=False)
    assert records[0]["text"] == "你们客服太差了，我要投诉"


def test_is_pool_row_labelled_accepts_any_filled_field():
    assert is_pool_row_labelled(make_pool_row()) is True
    assert is_pool_row_labelled({"status": "pending"}) is False
    assert is_pool_row_labelled({"expected_intent": "转人工"}) is True   # 只填了意图也算标过
