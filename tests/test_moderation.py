import pytest
from vision.moderation import (
    evaluate_moderation,
    detect_primary_deity,
    calculate_category_type_scores,
    ModerationAction,
    CategoryType,
)


def test_detect_primary_deity_shiva():
    clusters = {"shiva_iconography": 0.45, "puja_and_aarti_rituals": 0.30}
    assert detect_primary_deity(clusters) == "shiva"


def test_detect_primary_deity_none():
    clusters = {"modern_vehicles": 0.80}
    assert detect_primary_deity(clusters) is None


def test_category_type_scores_aggregation():
    clusters = {
        "shiva_iconography": 0.50,
        "spam_and_promotions": 0.10,
        "violence_and_harm": 0.15,
        "modern_vehicles": 0.20,
        "modern_nightlife_and_parties": 0.45,
    }
    type_scores = calculate_category_type_scores(clusters)
    assert type_scores[CategoryType.SACRED_DEVOTIONAL.value] == 0.50
    assert type_scores[CategoryType.COMMERCIAL_SPAM.value] == 0.10
    assert type_scores[CategoryType.SAFETY_VIOLATIONS.value] == 0.15
    assert type_scores[CategoryType.NEUTRAL_CONTENT.value] == 0.20
    assert type_scores[CategoryType.DISRUPTIVE_CONTENT.value] == 0.45


def test_strict_override_nudity_block():
    clusters = {"nudity_and_sexual_content": 0.22}
    res = evaluate_moderation(clusters)
    assert res.flagged is True
    assert res.action == ModerationAction.BLOCK.value
    assert "nudity_and_sexual_content: 0.22" in res.reasons


def test_strict_override_non_veg_meat_block():
    clusters = {"non_veg_and_meat": 0.28}
    res = evaluate_moderation(clusters)
    assert res.flagged is True
    assert res.action == ModerationAction.BLOCK.value
    assert "non_veg_and_meat: 0.28" in res.reasons


def test_disruptive_content_flag():
    # Modern nightlife >= 0.40 triggers FLAG
    clusters = {"modern_nightlife_and_parties": 0.45}
    res = evaluate_moderation(clusters)
    assert res.flagged is True
    assert res.action == ModerationAction.FLAG.value
    assert "modern_nightlife_and_parties: 0.45" in res.reasons


def test_disruptive_content_block():
    # Modern nightlife >= 0.60 triggers BLOCK
    clusters = {"modern_nightlife_and_parties": 0.65}
    res = evaluate_moderation(clusters)
    assert res.flagged is True
    assert res.action == ModerationAction.BLOCK.value
    assert "modern_nightlife_and_parties: 0.65" in res.reasons
