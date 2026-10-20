"""LLM 改写测试：prompt 构造、输出清理、回落逻辑、注入假客户端。"""

from __future__ import annotations

from agent.rag.rewrite import (
    LLMRewriter,
    RewriteContext,
    build_rewrite_messages,
    postprocess_rewrite,
    reset_llm_rewriter,
    rewrite_detailed,
    rewrite_with_cache,
)


def test_build_rewrite_messages_includes_context():
    messages = build_rewrite_messages(
        "那它呢", RewriteContext(last_question="草莓怎么保存", product="草莓")
    )
    assert messages[0]["role"] == "system"
    user_content = messages[1]["content"]
    assert "草莓怎么保存" in user_content
    assert "草莓" in user_content
    assert "那它呢" in user_content


def test_postprocess_rewrite_strips_prefix_and_quotes():
    assert postprocess_rewrite("改写后：「草莓怎么保存」") == "草莓怎么保存"
    assert postprocess_rewrite("  草莓怎么保存\n多余的解释") == "草莓怎么保存"
    assert postprocess_rewrite("") == ""


def test_llm_rewriter_uses_injected_client():
    rewriter = LLMRewriter(complete_fn=lambda messages: "草莓怎么保存")
    result = rewriter.rewrite(
        "那它呢", RewriteContext(last_question="草莓怎么保存", product="草莓")
    )
    assert result is not None
    assert result.query == "草莓怎么保存"
    assert result.strategy == "llm"
    assert result.rewritten is True


def test_llm_rewriter_reports_error_on_failure():
    def boom(messages):
        raise RuntimeError("网络不可达")

    result = LLMRewriter(complete_fn=boom).rewrite("那它呢", RewriteContext())
    assert result is not None
    assert result.strategy == "llm-error"
    assert "网络不可达" in result.error
    assert result.query == "那它呢"


def test_llm_mode_falls_back_to_rule_without_real_key(monkeypatch):
    monkeypatch.setenv("LLM_REWRITE_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "sk-替换成你的-deepseek-key")
    monkeypatch.setenv("LLM_REWRITE_MODEL", "deepseek-flash")
    reset_llm_rewriter()
    try:
        result = rewrite_detailed(
            "那它呢",
            {"last_question": "草莓怎么保存", "product": "草莓"},
            mode="llm",
        )
        assert result.strategy == "rule-fallback"
        assert result.query == "草莓怎么保存"  # 规则版仍然给出正确改写
        assert "未配置" in result.error
    finally:
        reset_llm_rewriter()


def test_rule_mode_never_calls_llm():
    calls = {"count": 0}

    def spy(messages):
        calls["count"] += 1
        return "不应该被调用"

    result = rewrite_detailed(
        "那它呢",
        {"last_question": "草莓怎么保存", "product": "草莓"},
        mode="rule",
        rewriter=LLMRewriter(complete_fn=spy),
    )
    assert calls["count"] == 0
    assert result.strategy == "rule"
    assert result.query == "草莓怎么保存"


def test_auto_mode_uses_llm_when_available():
    rewriter = LLMRewriter(complete_fn=lambda messages: "车厘子能退吗")
    result = rewrite_detailed("这个能退吗", {"product": "车厘子"}, mode="auto", rewriter=rewriter)
    assert result.strategy == "llm"
    assert result.query == "车厘子能退吗"


def test_rewrite_cache_reuses_llm_result():
    calls = {"count": 0}

    def complete(messages):
        calls["count"] += 1
        return "草莓怎么保存"

    rewriter = LLMRewriter(complete_fn=complete)
    cache: dict = {}
    context = {"last_question": "草莓怎么保存", "product": "草莓"}

    first = rewrite_with_cache(
        "那它呢", context, mode="llm", cache=cache, model="deepseek-flash", rewriter=rewriter
    )
    second = rewrite_with_cache(
        "那它呢", context, mode="llm", cache=cache, model="deepseek-flash", rewriter=rewriter
    )

    assert calls["count"] == 1  # 第二次命中缓存，不再调用模型
    assert first.query == second.query == "草莓怎么保存"
    assert len(cache) == 1


def test_rewrite_cache_skips_fallback_results(monkeypatch):
    monkeypatch.setenv("LLM_REWRITE_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "sk-替换成你的-deepseek-key")
    reset_llm_rewriter()
    try:
        cache: dict = {}
        result = rewrite_with_cache(
            "那它呢",
            {"last_question": "草莓怎么保存", "product": "草莓"},
            mode="llm",
            cache=cache,
        )
        assert result.strategy == "rule-fallback"
        assert cache == {}  # 回落结果不能进缓存，否则配好 key 后永远命中兜底
    finally:
        reset_llm_rewriter()
