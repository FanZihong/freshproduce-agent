"""意图体系测试：结构完整性、路由映射、与规则版的一致性下限。"""

from __future__ import annotations

from agent.intent.classifier import classify
from agent.intent.taxonomy import (
    VALID_ROUTES,
    load_taxonomy,
    route_of,
    validate,
)


def test_taxonomy_loads_and_validates():
    taxonomy = load_taxonomy()
    assert len(taxonomy.labels) >= 10
    assert validate(taxonomy) == []


def test_every_label_has_examples_and_route():
    taxonomy = load_taxonomy()
    for spec in taxonomy.labels:
        assert len(spec.positives) >= 3, spec.key
        assert len(spec.negatives) >= 2, spec.key
        assert spec.route in VALID_ROUTES, spec.key


def test_routes_cover_all_five_branches():
    taxonomy = load_taxonomy()
    routes = {spec.route for spec in taxonomy.labels}
    assert routes == VALID_ROUTES


def test_route_mapping_handles_alias_and_unknown_labels():
    # 代码里的旧标签要能映射到体系里的分支
    assert route_of("knowledge_qa") == "faq"
    assert route_of("fallback") == "fallback"
    assert route_of("quality_qa") == "faq"
    assert route_of("不存在的标签") == "fallback"


def test_routing_priority_covers_every_label():
    taxonomy = load_taxonomy()
    assert set(taxonomy.keys) <= set(taxonomy.routing_priority)


def test_rule_classifier_branch_accuracy_is_not_zero():
    """规则版只是兜底，这里只保证它在体系正例上有基本的可用性。"""
    taxonomy = load_taxonomy()
    total = 0
    hit = 0
    for spec in taxonomy.labels:
        for text in spec.positives:
            total += 1
            if taxonomy.route_of(classify(text).label) == spec.route:
                hit += 1
    assert hit / total >= 0.5
