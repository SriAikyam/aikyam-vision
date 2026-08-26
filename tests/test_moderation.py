import pytest
from vision.moderation import (
    evaluate_moderation,
    detect_primary_deity,
    calculate_category_type_scores,
    ModerationAction,
    CategoryType,
)


def test_detect_primary_deity_shiva():
    clusters = {"shiva_iconography": 0.45, "puja_rituals_and_temples": 0.30}
    assert detect_primary_deity(clusters) == "shiva"


def test_detect_primary_deity_none():
    clusters = {"modern_vehicles_and_tech": 0.80}
    assert detect_primary_deity(clusters) is None


def test_category_type_scores_aggregation():
    clusters = {
        "shiva_iconography": 0.50,
        "commercial_spam_and_flyers": 0.10,
        "physical_violence_and_harm": 0.15,
        "modern_vehicles_and_tech": 0.20,
    }
    type_scores = calculate_category_type_scores(clusters)
    assert type_scores[CategoryType.SACRED_DEVOTIONAL.value] == 0.50
    assert type_scores[CategoryType.COMMERCIAL_SPAM.value] == 0.10
    assert type_scores[CategoryType.SAFETY_VIOLATIONS.value] == 0.15
    assert type_scores[CategoryType.NEUTRAL_CONTENT.value] == 0.20


def test_strict_override_nudity_block():
    clusters = {"nudity_and_sexual_content": 0.22}
    res = evaluate_moderation(clusters)
    assert res.flagged is True
    assert res.action == ModerationAction.BLOCK.value
    assert "nudity_and_sexual_content: 0.22" in res.reasons


def test_strict_override_violence_block():
    clusters = {"physical_violence_and_harm": 0.28}
    res = evaluate_moderation(clusters)
    assert res.flagged is True
    assert res.action == ModerationAction.BLOCK.value
    assert "physical_violence_and_harm: 0.28" in res.reasons


def test_spam_flag():
    clusters = {"commercial_spam_and_flyers": 0.45}
    res = evaluate_moderation(clusters)
    assert res.flagged is True
    assert res.action == ModerationAction.FLAG.value
    assert "commercial_spam_and_flyers: 0.45" in res.reasons
