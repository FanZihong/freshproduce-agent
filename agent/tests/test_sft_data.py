"""SFT 数据构造测试：目标串往返、两种多标签编码、长尾过采样、域外配比。"""

from __future__ import annotations

import pytest

from agent.intent.target import format_target, output_contract, parse_target
from agent.intent.taxonomy import load_taxonomy
from scripts.prepare_sft_data import (
    adjust_ood,
    build_split,
    build_two_part,
    distribution,
    oversample,
    strategy_stats,
    target_conflicts,
)

SYSTEM = "你是意图识别器。"


def make_row(index: int, label: str) -> dict:
    return {
        "id": f"r{index}",
        "text": f"第{index}条问句",
        "context": None,
        "labels": [label],
        "source": "test",
    }


def make_rows() -> list[dict]:
    return [
        {
            "id": "1",
            "text": "车厘子坏了而且还没发货",
            "context": None,
            "labels": ["bad_fruit_claim", "urge_logistics"],
            "source": "test",
        },
        {
            "id": "2",
            "text": "怎么开发票",
            "context": {"last_question": "订单相关问题"},
            "labels": ["invoice"],
            "source": "test",
        },
        {
            "id": "3",
            "text": "今天天气怎么样",
            "context": None,
            "labels": ["chitchat"],
            "source": "test",
        },
    ]


def build(style: str = "key", strategy: str = "two_part") -> list[dict]:
    taxonomy = load_taxonomy()
    rows = make_rows()
    if strategy == "two_part":
        return build_two_part(rows, SYSTEM, taxonomy, style, 2)
    return build_split(rows, SYSTEM, taxonomy, style, 2)


def test_format_target_key_style():
    assert format_target(["bad_fruit_claim"]) == "主:bad_fruit_claim"
    assert (
        format_target(["bad_fruit_claim", "urge_logistics"])
        == "主:bad_fruit_claim|辅:urge_logistics"
    )


def test_format_target_name_style_needs_taxonomy():
    taxonomy = load_taxonomy()
    assert format_target(["bad_fruit_claim", "urge_logistics"], "name", taxonomy) == "主:坏果理赔|辅:催物流"
    with pytest.raises(ValueError):
        format_target(["invoice"], "name")


def test_format_target_json_style_and_confidence():
    target = format_target(["invoice"], "json", confidence=0.93)
    assert '"labels": ["invoice"]' in target
    assert '"confidence": 0.93' in target


def test_format_target_dedupes_and_caps_labels():
    assert format_target(["invoice", "invoice"]) == "主:invoice"
    assert format_target(["invoice", "complaint", "chitchat"], max_labels=2) == "主:invoice|辅:complaint"


def test_output_contract_shows_both_formats():
    contract = output_contract("key", 2)
    assert "主:bad_fruit_claim" in contract
    assert "辅:urge_logistics" in contract
    assert "最多 2 个" in contract


@pytest.mark.parametrize("style", ["key", "name", "json"])
def test_target_round_trip(style):
    labels = list(load_taxonomy().keys)
    taxonomy = load_taxonomy()
    target = format_target(["bad_fruit_claim", "urge_logistics"], style, taxonomy)
    parsed, confidence, _ = parse_target(target, labels, taxonomy=taxonomy)
    assert set(parsed) == {"bad_fruit_claim", "urge_logistics"}
    if style == "json":
        assert confidence == 0.0  # 没给 confidence 时按 0 处理


def test_parse_target_is_lenient():
    labels = list(load_taxonomy().keys)
    taxonomy = load_taxonomy()
    # 空格、全角逗号、缺前缀
    parsed, _, _ = parse_target("主: bad_fruit_claim ，辅: urge_logistics", labels, taxonomy=taxonomy)
    assert set(parsed) == {"bad_fruit_claim", "urge_logistics"}
    # 前后夹带说明
    parsed, _, _ = parse_target("我觉得是 refund_consult 吧", labels, taxonomy=taxonomy)
    assert parsed == ("refund_consult",)
    # 旧别名
    parsed, _, _ = parse_target("knowledge_qa", labels, taxonomy=taxonomy)
    assert parsed == ("quality_qa",)
    # 中文名
    parsed, _, _ = parse_target("主:坏果理赔", labels, taxonomy=taxonomy)
    assert parsed == ("bad_fruit_claim",)
    # 代码块包裹的 JSON
    parsed, _, _ = parse_target('```json\n{"labels": ["invoice"], "confidence": 0.8}\n```', labels)
    assert parsed == ("invoice",)


def test_parse_target_rejects_garbage():
    labels = list(load_taxonomy().keys)
    with pytest.raises(ValueError):
        parse_target("我觉得还行", labels)
    with pytest.raises(ValueError):
        parse_target("", labels)


def test_two_part_keeps_auxiliary_label_in_one_sample():
    records = build("key", "two_part")
    assert len(records) == 3
    multi = [record for record in records if len(record["labels"]) == 2]
    assert len(multi) == 1
    assert multi[0]["target"] == "主:bad_fruit_claim|辅:urge_logistics"
    assert target_conflicts(records) == 0
    assert [message["role"] for message in records[0]["messages"]] == ["system", "user", "assistant"]


def test_split_encoding_expands_labels_and_creates_conflicts():
    records = build("key", "split")
    assert len(records) == 4  # 3 条里的 2 标签行拆成 2 条
    assert all(len(record["labels"]) == 1 for record in records)
    assert all("辅:" not in record["target"] for record in records)
    # 拆分方案把同一句话喂给两个不同目标 -> 冲突
    assert target_conflicts(records) == 1


def test_strategy_stats_counts_positives_per_label():
    labels = list(load_taxonomy().keys)
    stats = strategy_stats(build("key", "split"), labels)
    assert stats["samples"] == 4
    assert stats["multi_label_rows"] == 0
    assert stats["per_class_positive"]["bad_fruit_claim"] == 1
    assert stats["per_class_positive"]["urge_logistics"] == 1
    assert stats["per_class_positive"]["chitchat"] == 1
    assert stats["min_class"] == 1


def test_build_two_part_skips_rows_without_labels():
    rows = [{"id": "x", "text": "空标签", "context": None, "labels": []}]
    assert build_two_part(rows, SYSTEM, load_taxonomy(), "key", 2) == []


def test_context_is_rendered_into_user_message():
    records = build("key", "two_part")
    user_message = records[1]["messages"][1]["content"]
    assert "上一轮用户问题：订单相关问题" in user_message
    assert "用户当前问句：怎么开发票" in user_message


def test_oversample_duplicates_rare_class_only():
    records = [make_row(index, "bad_fruit_claim") for index in range(4)] + [make_row(99, "invoice")]
    result, stats = oversample(records, target=3, seed=1)
    assert stats["enabled"] is True
    counter = distribution(result, list(load_taxonomy().keys))
    assert counter["bad_fruit_claim"] == 4  # 已经在目标之上，不动
    assert counter["invoice"] == 3          # 补到目标
    assert stats["added"] == 2
    duplicates = [record for record in result if record.get("dup", 1) > 1]
    assert len(duplicates) == 2
    assert all("#dup" in record["id"] for record in duplicates)


def test_oversample_disabled_by_default():
    records = build("key", "two_part")
    result, stats = oversample(records, target=0)
    assert stats["enabled"] is False
    assert len(result) == len(records)


def test_adjust_ood_raises_ratio_to_floor():
    records = build("key", "two_part")  # 3 条里 1 条 chitchat = 33%
    result, stats = adjust_ood(records, "chitchat", min_ratio=0.5, max_ratio=0.0, seed=1)
    ratio = stats["after"]
    assert stats["added"] > 0
    assert ratio >= 0.5
    assert stats["before"] == 0.3333


def test_adjust_ood_caps_ratio_when_max_given():
    records = [make_row(index, "chitchat" if index < 10 else "invoice") for index in range(20)]
    result, stats = adjust_ood(records, "chitchat", min_ratio=0.0, max_ratio=0.2, seed=1)
    assert stats["dropped"] > 0
    assert stats["after"] <= 0.2
    assert len(result) < len(records)


def test_adjust_ood_skips_cap_when_other_classes_are_too_few():
    records = build("key", "split")  # 4 条里只有 3 条非域外，硬裁会把域外删空
    result, stats = adjust_ood(records, "chitchat", min_ratio=0.0, max_ratio=0.2, seed=1)
    assert stats["dropped"] == 0
    assert len(result) == len(records)
    assert "note" in stats
