"""数据集校准与划分测试：kappa、采纳率、分层划分、长尾告警。"""

from __future__ import annotations

from eval.metrics import cohens_kappa, confusion_matrix, label_agreement, macro_f1
from scripts.apply_calibration import merge_calibration
from scripts.split_intent_dataset import build_stats, split_dataset


def test_kappa_perfect_and_known_case():
    assert cohens_kappa(["a", "a"], ["a", "a"]) == 1.0
    # po = 0.75，pe = 0.5 -> kappa = 0.5
    assert cohens_kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"]) == 0.5
    assert label_agreement(["a", "b"], ["a", "b"]) == 1.0


def test_macro_f1_and_confusion_matrix():
    gold = ["a", "a", "b"]
    pred = ["a", "b", "b"]
    assert macro_f1(gold, pred) == 0.6667
    matrix = confusion_matrix(gold, pred, labels=["a", "b"])
    assert matrix["a"]["a"] == 1
    assert matrix["a"]["b"] == 1
    assert matrix["b"]["b"] == 1


def test_split_keeps_sessions_together_and_covers_all_rows():
    rows = [
        {
            "id": f"s{session}-t{turn}",
            "session_id": f"s{session}",
            "text": "测试问句",
            "labels": ["query_order"],
        }
        for session in range(20)
        for turn in range(3)
    ]
    splits = split_dataset(rows, seed=1)
    assert sum(len(split) for split in splits.values()) == len(rows)

    seen: dict[str, str] = {}
    for name, split_rows in splits.items():
        for row in split_rows:
            session = row["session_id"]
            assert session not in seen or seen[session] == name
            seen[session] = name


def test_merge_calibration_applies_reviewed_labels():
    prelabel = [{"id": "a", "text": "怎么开票", "labels": ["invoice"], "confidence": 0.9}]
    review = [
        {"id": "a", "reviewed_labels": ["refund_consult"], "note": "应为退款咨询"},
        {"id": "b", "reviewed_labels": []},
    ]
    merged, stats = merge_calibration(prelabel, review)

    assert merged[0]["labels"] == ["refund_consult"]
    assert merged[0]["review_status"] == "corrected"
    assert stats["corrected"] == 1
    assert stats["adoption_rate"] == 0.0
    assert stats["pending_review"] == 1


def test_merge_calibration_counts_adoption():
    prelabel = [{"id": "a", "text": "怎么开票", "labels": ["invoice"], "confidence": 0.9}]
    review = [{"id": "a", "reviewed_labels": ["invoice"]}]
    _, stats = merge_calibration(prelabel, review)
    assert stats["confirmed"] == 1
    assert stats["adoption_rate"] == 1.0


def test_build_stats_flags_long_tail():
    splits = {
        "train": [{"id": "1", "text": "怎么开票", "labels": ["invoice"]}],
        "dev": [],
        "test": [],
    }
    stats = build_stats(splits, min_per_class=50)
    assert any("invoice" in warning for warning in stats["warnings"])
