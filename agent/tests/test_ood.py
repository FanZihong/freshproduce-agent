"""域外判定测试：信号投票、护栏、相似度计算（用假 embedder，不依赖模型下载）。"""

from __future__ import annotations

import pytest

from agent.intent.ood import (
    OodConfig,
    build_class_centroids,
    collect_signals,
    cosine,
    domain_hits,
    is_out_of_domain,
    make_similarity_fn,
    max_class_similarity,
    required_votes,
)
from agent.intent.taxonomy import load_taxonomy


class FakeEmbedder:
    """把文本映射成 3 维向量的假 embedder：含"生鲜业务词"就往第一维靠。"""

    name = "fake-embedder"
    dim = 3

    def encode(self, texts, is_query: bool = False):
        rows = []
        for text in texts:
            business = any(word in text for word in ("订单", "赔付", "物流", "保存", "退款", "品质"))
            rows.append([1.0 if business else 0.0, 1.0 if not business else 0.0, 0.1])
        return rows


def test_domain_hits_matches_topics_and_products():
    assert "订单" in domain_hits("我的订单到哪了")
    assert "草莓" in domain_hits("草莓坏了一半怎么赔")      # 商品名也算领域词
    assert domain_hits("今天天气怎么样") == ()
    assert domain_hits("") == ()


def test_guard_blocks_false_kill_even_with_all_signals():
    """命中领域词时，哪怕两个信号都投 OOD，也强制判域内——这是误杀率的保险丝。"""
    signals = collect_signals(
        "我要投诉，退款什么时候到账",
        confidence=0.1,
        rule_label="fallback",
        config=OodConfig(),
    )
    verdict, reason = is_out_of_domain(signals)
    assert verdict is False
    assert "命中领域词" in reason


def test_model_chitchat_alone_triggers_ood():
    """模型自己判 chitchat 就是第一路（实测本地模型域外召回 0.94），单路即可判域外。"""
    signals = collect_signals("讲个笑话吧", model_label="chitchat", confidence=0.93)
    assert signals.model_says_ood is True
    assert "model_chitchat" in signals.votes
    verdict, reason = is_out_of_domain(signals)
    assert verdict is True
    assert "第一路信号" in reason


def test_guard_overrides_model_chitchat():
    """护栏优先：模型判错成 chitchat，但句子里有领域词 -> 仍然判域内。"""
    signals = collect_signals("我的订单到哪了", model_label="chitchat", confidence=0.9)
    verdict, reason = is_out_of_domain(signals)
    assert verdict is False
    assert "命中领域词" in reason


def test_rule_business_vetoes_model_chitchat():
    """规则版明确判成业务意图时，否决模型自己的 chitchat 判定（压误杀的主手段）。"""
    # 特意选一句不含领域词的文本，确保走的是"规则否决"而不是"领域词护栏"
    signals = collect_signals("这个怎么办啊", model_label="chitchat", rule_label="query_order")
    assert signals.rule_business is True
    assert signals.domain_hits == ()
    verdict, reason = is_out_of_domain(signals)
    assert verdict is False
    assert "否决" in reason

    # 关掉否决开关时，回到"模型说了算"
    verdict_no_veto, _ = is_out_of_domain(signals, use_rule_veto=False)
    assert verdict_no_veto is True


def test_rule_veto_does_not_fire_for_real_out_of_domain():
    """真正的域外问句上，规则版基本都判兜底 -> 否决不生效，召回不受影响。"""
    signals = collect_signals("今天天气怎么样", model_label="chitchat", rule_label="fallback")
    assert signals.rule_business is False
    verdict, reason = is_out_of_domain(signals)
    assert verdict is True
    assert "第一路信号" in reason


def test_model_label_counts_toward_available_signals():
    signals = collect_signals("随便聊聊", model_label="query_order", confidence=0.9)
    assert signals.available == 2  # 模型标签 + 置信度
    assert signals.votes == []
    assert is_out_of_domain(signals)[0] is False


def test_no_signal_means_in_domain():
    # 不传 rule_label：避开"规则否决"分支，专门验证"没有任何信号命中"时的兜底
    signals = collect_signals("今天天气怎么样", confidence=0.9)
    verdict, reason = is_out_of_domain(signals)
    assert verdict is False
    assert "没有信号命中" in reason


def test_two_signals_trigger_ood_without_embedder():
    signals = collect_signals("讲个笑话吧", confidence=0.3, rule_label="fallback")
    assert signals.available == 2
    verdict, reason = is_out_of_domain(signals)
    assert verdict is True
    assert "2/2 路信号" in reason


def test_single_signal_is_not_enough_when_two_available():
    signals = collect_signals("讲个笑话吧", confidence=0.3, rule_label="query_order")
    assert signals.votes == ["low_confidence"]
    verdict, _ = is_out_of_domain(signals)
    assert verdict is False


def test_similarity_signal_joins_the_vote():
    # 低置信度 + 低相似度 -> 两路命中
    signals = collect_signals("讲个笑话吧", confidence=0.4, max_similarity=0.1)
    assert signals.votes == ["low_confidence", "low_similarity"]
    assert is_out_of_domain(signals)[0] is True


def test_required_votes_drops_when_signals_missing():
    """没装 embedder 时只有 2 路可用，要求自动降到 2；只有 1 路可用时要求降到 1。"""
    two = collect_signals("x", confidence=0.9, rule_label="query_order")
    one = collect_signals("x", confidence=0.9)
    assert required_votes(two, OodConfig(min_votes=2)) == 2
    assert required_votes(one, OodConfig(min_votes=2)) == 1


def test_cosine_and_max_similarity_with_fake_embedder():
    embedder = FakeEmbedder()
    centroids = build_class_centroids(embedder, load_taxonomy())
    assert centroids                      # 每类都能算出质心
    assert len(centroids) >= 10

    # 假 embedder 只产出两个互斥方向，而余弦对缩放不敏感——同类向量之间恒为 1.0，
    # 所以这里断言"机制"（取值 = 与最像质心的余弦），而不是"业务句相似度更高"这种语义排序。
    business_query = [1.0, 0.0, 0.1]
    chitchat_query = [0.0, 1.0, 0.1]
    business_centroid = [1.0, 0.0, 0.0]
    assert cosine(business_query, business_centroid) > cosine(chitchat_query, business_centroid)

    manual = {"业务": [1.0, 0.0, 0.0], "无关": [0.0, 1.0, 0.0]}
    value = max_class_similarity(embedder, manual, "订单物流到哪了")
    assert value == pytest.approx(cosine(business_query, business_centroid), abs=1e-4)
    assert max_class_similarity(embedder, manual, "今天天气怎么样") == pytest.approx(
        cosine(chitchat_query, [0.0, 1.0, 0.0]), abs=1e-4
    )

    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0, abs=1e-6)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0, abs=1e-6)
    assert cosine([], [1]) == 0.0


def test_make_similarity_fn_returns_callable():
    similarity = make_similarity_fn(FakeEmbedder())
    assert 0.0 <= similarity("车厘子坏了一半") <= 1.0


def test_centroids_skip_when_embedder_returns_empty():
    class EmptyEmbedder:
        name = "empty"
        dim = 3

        def encode(self, texts, is_query: bool = False):
            return []

    assert build_class_centroids(EmptyEmbedder(), load_taxonomy()) == {}
    # 没有参照时返回 1.0（不投票），避免把一切都判成域外
    assert max_class_similarity(EmptyEmbedder(), {}, "任一句话") == 1.0
