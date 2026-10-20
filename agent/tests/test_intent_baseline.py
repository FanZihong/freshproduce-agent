"""few-shot 基线测试：提示词渲染、输出解析、时延/成本记录、失败回落、缓存、后端开关。

全部用注入式假模型（complete_fn），不依赖网络与 API key。
"""

from __future__ import annotations

import json

import pytest

from agent.intent import classifier
from agent.intent.baseline import (
    FewShotIntentClassifier,
    LLMIntentConfig,
    build_messages,
    build_system_prompt,
    parse_intent_response,
)
from agent.intent.taxonomy import load_taxonomy


def make_config(**overrides) -> LLMIntentConfig:
    payload = {
        "api_key": "sk-test-key",
        "model": "test-model",
        "price_in_per_m": 2.0,
        "price_out_per_m": 8.0,
    }
    payload.update(overrides)
    return LLMIntentConfig(**payload)


def make_completer(payload: dict | str, prompt_tokens: int = 1000, completion_tokens: int = 20):
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)

    def complete(messages):
        return text, {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

    return complete


def test_system_prompt_covers_every_label_and_output_contract():
    taxonomy = load_taxonomy()
    prompt = build_system_prompt(taxonomy)
    for key in taxonomy.keys:
        assert key in prompt, key
    assert '"labels"' in prompt and "confidence" in prompt
    assert str(taxonomy.multi_intent_policy["max_labels"]) in prompt
    # 边界提示也进了提示词：否则模型只能靠类别名猜
    assert taxonomy.labels[0].confusions[0].tip in prompt


def test_system_prompt_contains_taxonomy_examples_and_hard_cases():
    prompt = build_system_prompt(load_taxonomy(), shots_per_class=1)
    assert "我的订单到哪了" in prompt          # 体系正例
    assert "车厘子坏了而且到现在还没发货" in prompt  # 人工难例（多意图）


def test_messages_include_context_when_available():
    messages = build_messages(
        "那它呢",
        {"last_question": "草莓怎么保存", "product": "草莓"},
        taxonomy=load_taxonomy(),
    )
    assert messages[0]["role"] == "system"
    assert "草莓怎么保存" in messages[1]["content"]
    assert "那它呢" in messages[1]["content"]


def test_parse_handles_code_fence_and_legacy_alias():
    valid = set(load_taxonomy().keys)
    raw = '```json\n{"labels": ["knowledge_qa"], "confidence": 0.8, "reason": "咨询"}\n```'
    labels, confidence, reason = parse_intent_response(raw, valid)
    assert labels == ("quality_qa",)
    assert confidence == 0.8
    assert reason == "咨询"


def test_parse_accepts_single_label_field_and_clamps_confidence():
    labels, confidence, _ = parse_intent_response(
        '{"label": "invoice", "confidence": 2, "reason": "开票"}', set(load_taxonomy().keys)
    )
    assert labels == ("invoice",)
    assert confidence == 1.0


def test_parse_caps_labels_by_policy_and_dedupes():
    raw = json.dumps(
        {"labels": ["bad_fruit_claim", "urge_logistics", "invoice", "bad_fruit_claim"]},
        ensure_ascii=False,
    )
    labels, _, _ = parse_intent_response(raw, set(load_taxonomy().keys), max_labels=2)
    assert labels == ("bad_fruit_claim", "urge_logistics")


def test_parse_rejects_unknown_labels():
    with pytest.raises(ValueError):
        parse_intent_response('{"labels": ["不存在的类别"]}', set(load_taxonomy().keys))
    with pytest.raises(ValueError):
        parse_intent_response("我觉得是理赔", set(load_taxonomy().keys))


def test_classify_with_stats_records_latency_tokens_and_cost():
    classifier_ = FewShotIntentClassifier(
        config=make_config(), complete_fn=make_completer({"labels": ["invoice"], "confidence": 0.9})
    )
    call = classifier_.classify_with_stats("怎么开发票")
    assert call.label == "invoice"
    assert call.labels == ("invoice",)
    assert call.latency_ms >= 0
    assert (call.prompt_tokens, call.completion_tokens) == (1000, 20)
    # 1000 * 2 + 20 * 8 每百万 token
    assert call.cost == 0.00216
    assert call.cached is False
    assert call.error == ""


def test_empty_input_short_circuits_without_calling_model():
    calls: list[list[dict]] = []

    def complete(messages):
        calls.append(messages)
        return "{}"

    classifier_ = FewShotIntentClassifier(config=make_config(), complete_fn=complete)
    call = classifier_.classify_with_stats("   ")
    assert call.strategy == "short-circuit"
    assert call.label == "chitchat"
    assert calls == []


def test_call_failure_is_reported_and_falls_back_to_rule():
    def boom(messages):
        raise RuntimeError("连接超时")

    classifier_ = FewShotIntentClassifier(config=make_config(), complete_fn=boom)
    call = classifier_.classify_with_stats("草莓坏了一半")
    assert call.labels == ()
    assert "RuntimeError" in call.error
    assert call.strategy == "fewshot-error"

    result = classifier_.classify("草莓坏了一半")
    assert result.label == "bad_fruit_claim"   # 规则版兜底
    assert "回落规则版" in result.reason


def test_parse_error_counted_as_wrong_but_cost_kept():
    classifier_ = FewShotIntentClassifier(
        config=make_config(), complete_fn=make_completer("这句不是 JSON")
    )
    call = classifier_.classify_with_stats("怎么开发票")
    assert call.labels == ()
    assert call.strategy == "fewshot-parse-error"
    assert call.cost == 0.00216  # 解析失败也真实消耗了 token


def test_response_cache_avoids_second_call_and_keeps_cost():
    calls: list[int] = []

    def complete(messages):
        calls.append(1)
        return (
            json.dumps({"labels": ["invoice"], "confidence": 0.9}, ensure_ascii=False),
            {"prompt_tokens": 1000, "completion_tokens": 20},
        )

    cache: dict = {}
    classifier_ = FewShotIntentClassifier(
        config=make_config(), complete_fn=complete, response_cache=cache
    )
    first = classifier_.classify_with_stats("怎么开发票")
    second = classifier_.classify_with_stats("怎么开发票")

    assert len(calls) == 1
    assert first.cached is False
    assert second.cached is True
    assert second.labels == first.labels
    assert second.cost == first.cost  # 缓存不清零成本口径
    assert len(cache) == 1


def test_available_requires_real_api_key():
    assert FewShotIntentClassifier(config=make_config(api_key="")).available is False
    assert FewShotIntentClassifier(config=make_config(api_key="sk-替换成你的-key")).available is False
    assert FewShotIntentClassifier(config=make_config()).available is True
    assert FewShotIntentClassifier(config=LLMIntentConfig(), complete_fn=make_completer("{}")).available


@pytest.fixture(autouse=True)
def _reset_backend_cache():
    classifier._classifiers.clear()
    classifier._warned.clear()
    yield
    classifier._classifiers.clear()
    classifier._warned.clear()


def test_backend_defaults_to_rule(monkeypatch):
    monkeypatch.delenv("INTENT_BACKEND", raising=False)
    assert classifier.get_backend() == "rule"
    assert classifier.classify("我的订单到哪了").label == "query_order"


def test_backend_switch_uses_fewshot_when_key_configured(monkeypatch):
    monkeypatch.setenv("INTENT_BACKEND", "fewshot")
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
    chosen = classifier.get_classifier()
    assert chosen.name == "fewshot-llm"


def test_backend_falls_back_to_rule_without_key(monkeypatch, capsys):
    monkeypatch.setenv("INTENT_BACKEND", "fewshot")
    monkeypatch.setenv("LLM_API_KEY", "sk-替换成你的-deepseek-key")
    assert classifier.get_classifier().name == "rule-based-v1"
    assert "回落 rule" in capsys.readouterr().err


def test_unknown_backend_falls_back_to_rule(monkeypatch):
    monkeypatch.setenv("INTENT_BACKEND", "gpt-5")
    assert classifier.get_backend() == "rule"
