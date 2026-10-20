"""reranker/检索配置迭代脚本的测试：错例集合、救回计数、排序与结论分支。"""

from __future__ import annotations

from pathlib import Path

from scripts.iterate_rerank import (
    build_advice,
    config_metrics,
    rank_configs,
    rescue_count,
    write_report,
)


def make_payload() -> dict:
    return {
        "configs": [
            {
                "config": "hybrid+rw+score",
                "label": "混合（+改写+分数重排）",
                "metrics": {"recall@5": 0.8333, "recall@1": 0.4167, "mrr": 0.6178},
                "misses": [
                    {"id": "q014", "question": "分量少了 10% 能赔多少",
                     "relevant": ["赔付标准表#rule#014"], "retrieved": ["a", "b"]},
                    {"id": "q018", "question": "买的水果不想要了能退吗",
                     "relevant": ["售后政策总则#rule#005"], "retrieved": ["c", "d"]},
                ],
            },
            {
                "config": "vector+rw",
                "label": "向量（+规则改写）",
                "metrics": {"recall@5": 0.9667, "recall@1": 0.6667, "mrr": 0.7233},
                "misses": [
                    {"id": "q018", "question": "买的水果不想要了能退吗",
                     "relevant": ["售后政策总则#rule#005"], "retrieved": ["c", "d"]},
                ],
            },
        ]
    }


def test_config_metrics_reads_metrics_and_miss_ids():
    payload = make_payload()
    row = config_metrics(payload, "hybrid+rw+score")
    assert row["available"] is True
    assert row["recall@5"] == 0.8333 and row["recall@1"] == 0.4167
    assert row["miss_ids"] == ["q014", "q018"]
    assert row["misses"][0]["question"].startswith("分量少了")

    empty = config_metrics(payload, "不存在")
    assert empty["available"] is False and empty["recall@5"] == 0.0 and empty["miss_ids"] == []


def test_rescue_count_counts_misses_recovered():
    assert rescue_count(["q014", "q018"], ["q018"]) == 1
    assert rescue_count(["q014", "q018"], []) == 2
    assert rescue_count([], ["q014"]) == 0


def test_rank_configs_orders_by_recall_then_mrr():
    rows = [
        config_metrics(make_payload(), "hybrid+rw+score"),
        config_metrics(make_payload(), "vector+rw"),
    ]
    ranked = rank_configs(rows)
    assert [row["config"] for row in ranked] == ["vector+rw", "hybrid+rw+score"]
    # 没数据的排最后
    ranked_with_missing = rank_configs(rows + [config_metrics(make_payload(), "没有")])
    assert ranked_with_missing[-1]["config"] == "没有"


def test_build_advice_recommends_switchable_config():
    payload = make_payload()
    current = config_metrics(payload, "hybrid+rw+score")
    rows = [current, config_metrics(payload, "vector+rw")]
    advice = build_advice(current, rows, with_cross_encoder=False)
    assert advice["ok"] is True
    assert advice["best"]["config"] == "vector+rw"
    assert advice["rescued"] == 1
    # 纯向量不是环境变量能切的档位，结论里必须说清
    assert "需要改代码" in advice["verdict"] or "不是环境变量" in advice["verdict"]


def test_build_advice_keeps_current_when_best_is_worse_on_recall():
    payload = make_payload()
    current = config_metrics(payload, "hybrid+rw+score")
    worse = {
        "config": "hybrid+rw+bge",
        "label": "混合+精排",
        "recall@5": 0.8,
        "recall@1": 0.4,
        "mrr": 0.9,
        "miss_ids": [],
        "misses": [],
        "available": True,
    }
    advice = build_advice(current, [current, worse], with_cross_encoder=True)
    # 排序把 recall@5 更高的当前配置排在前面
    assert advice["best"]["config"] == "hybrid+rw+score"
    assert "保持现状" in advice["verdict"]


def test_build_advice_reports_missing_data():
    current = config_metrics(make_payload(), "hybrid+rw+score")
    advice = build_advice(current, [config_metrics(make_payload(), "没有")], with_cross_encoder=False)
    assert advice["ok"] is False


def test_write_report_renders_table_and_targets(tmp_path):
    payload = make_payload()
    current = config_metrics(payload, "hybrid+rw+score")
    advice = build_advice(current, [current, config_metrics(payload, "vector+rw")], with_cross_encoder=False)
    path = tmp_path / "rerank.md"
    write_report(path, advice, "hybrid+rw+score")
    text = path.read_text(encoding="utf-8")
    assert "## 一、对比表" in text and "## 二、当前配置救不回来的错例" in text
    assert "q014" in text and "分量少了 10% 能赔多少" in text
    assert "要改代码" in text
    assert Path(tmp_path / "rerank.md").exists()
