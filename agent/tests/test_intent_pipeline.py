"""意图语料流水线测试：种子生成、去重、dry-run 标注、结果校验。"""

from __future__ import annotations

from agent.intent.taxonomy import load_taxonomy
from scripts.build_intent_corpus import dedupe as corpus_dedupe
from scripts.build_intent_corpus import normalize
from scripts.gen_intent_seeds import build_rows
from scripts.gen_intent_seeds import dedupe as seeds_dedupe
from scripts.label_intent import dry_run_label, parse_label_response
from scripts.validate_intent_dataset import validate_rows


def test_seed_generator_covers_every_label():
    rows = seeds_dedupe(build_rows())
    taxonomy = load_taxonomy()
    covered = {label for row in rows for label in row["hint_labels"]}
    assert set(taxonomy.keys) <= covered
    assert len(rows) >= 500


def test_seed_dedupe_removes_exact_duplicates():
    rows = [
        {"id": "1", "text": "车厘子怎么保存", "hint_labels": []},
        {"id": "2", "text": "车厘子怎么保存", "hint_labels": []},
    ]
    assert len(seeds_dedupe(rows)) == 1


def test_corpus_normalize_strips_punctuation_and_spaces():
    assert normalize("  我的订单到哪了。 ") == "我的订单到哪了"


def test_corpus_dedupe_removes_exact_and_near_duplicates():
    rows = [
        {"text": "车厘子坏了一半", "context": None, "hint_labels": [], "source": "t"},
        {"text": "车厘子坏了一半", "context": None, "hint_labels": [], "source": "t"},
        {"text": "车厘子坏了一半啊", "context": None, "hint_labels": [], "source": "t"},
        {"text": "怎么开发票", "context": None, "hint_labels": [], "source": "t"},
    ]
    kept, removed = corpus_dedupe(rows, similarity=0.7)
    assert removed >= 2
    assert any(row["text"] == "怎么开发票" for row in kept)


def test_dry_run_label_prefers_hint_then_falls_back_to_rule():
    valid = set(load_taxonomy().keys)
    labels, confidence, _ = dry_run_label(
        {"text": "车厘子坏了一半", "hint_labels": ["bad_fruit_claim"]}, valid
    )
    assert labels == ["bad_fruit_claim"]
    assert confidence > 0.9

    fallback_labels, _, _ = dry_run_label({"text": "我的订单到哪了", "hint_labels": []}, valid)
    assert fallback_labels
    assert fallback_labels[0] in valid


def test_parse_label_response_handles_code_fence_and_unknown_label():
    valid = {"query_order", "bad_fruit_claim"}
    labels, confidence, reason = parse_label_response(
        '```json\n{"labels": ["bad_fruit_claim", "不存在"], "confidence": 0.9, "reason": "坏果"}\n```',
        valid,
    )
    assert labels == ["bad_fruit_claim"]
    assert confidence == 0.9
    assert reason == "坏果"


def test_validate_rows_flags_unknown_label_and_duplicates():
    valid = set(load_taxonomy().keys)
    rows = [
        {"id": "a", "text": "怎么开发票", "labels": ["invoice"], "confidence": 0.9},
        {"id": "b", "text": "怎么开发票", "labels": ["invoice"], "confidence": 0.8},
        {"id": "c", "text": "测试", "labels": ["不存在的标签"], "confidence": 0.5},
    ]
    problems, valid_rows = validate_rows(rows, valid)
    assert any("重复" in problem for problem in problems)
    assert any("体系外" in problem for problem in problems)
    assert len(valid_rows) == 1
