import pytest
from vision.moderation import (
    evaluate_moderation,
    detect_primary_deity,
    calculate_category_type_scores,
    dominant_pillar,
    pillar_margin,
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


def test_spam_flag_escalates_to_block_since_spam_is_not_devotional():
    # The STRICT rule alone would only FLAG this, but the devotional gate
    # escalates every non-devotional result to BLOCK regardless.
    clusters = {"commercial_spam_and_flyers": 0.45}
    res = evaluate_moderation(clusters)
    assert res.flagged is True
    assert res.action == ModerationAction.BLOCK.value
    assert "commercial_spam_and_flyers: 0.45" in res.reasons
    assert "NOT_DEVOTIONAL_BLOCK" in res.blocked_reasons


def test_dominant_pillar_recombines_split_deity_evidence():
    # No single cluster wins outright, but summed devotional evidence beats
    # the lone non-devotional cluster -- this is the case top-1-cluster gets wrong.
    clusters = {
        "rama_iconography": 0.15,
        "krishna_iconography": 0.12,
        "vishnu_iconography": 0.10,
        "shiva_iconography": 0.08,
        "modern_vehicles_and_tech": 0.20,
    }
    type_scores = calculate_category_type_scores(clusters)
    assert dominant_pillar(type_scores) == CategoryType.SACRED_DEVOTIONAL.value


def test_dominant_pillar_empty_defaults_neutral():
    assert dominant_pillar({}) == CategoryType.NEUTRAL_CONTENT.value


def test_evaluate_moderation_sets_is_devotional():
    clusters = {"shiva_iconography": 0.50, "puja_rituals_and_temples": 0.20}
    res = evaluate_moderation(clusters)
    assert res.is_devotional is True
    assert res.primary_pillar == CategoryType.SACRED_DEVOTIONAL.value


def test_evaluate_moderation_not_devotional():
    clusters = {"modern_vehicles_and_tech": 0.60}
    res = evaluate_moderation(clusters)
    assert res.is_devotional is False
    assert res.primary_pillar == CategoryType.NEUTRAL_CONTENT.value


def test_devotional_gate_blocks_harmless_non_devotional_content():
    # Platform policy: only Hindu devotional content is allowed through.
    # An ordinary, harmless photo (no STRICT rule, no cumulative safety score)
    # still gets BLOCKed for being non-devotional.
    clusters = {"modern_vehicles_and_tech": 0.60}
    res = evaluate_moderation(clusters)
    assert res.flagged is True
    assert res.action == ModerationAction.BLOCK.value
    assert "NOT_DEVOTIONAL_BLOCK" in res.blocked_reasons


def test_devotional_gate_allows_clean_devotional_content():
    clusters = {"shiva_iconography": 0.50, "puja_rituals_and_temples": 0.20}
    res = evaluate_moderation(clusters)
    assert res.flagged is False
    assert res.action == ModerationAction.ALLOW.value


def test_pillar_margin_basic():
    assert pillar_margin({"SACRED_DEVOTIONAL": 0.70, "NEUTRAL_CONTENT": 0.20}) == pytest.approx(0.50)
    assert pillar_margin({}) == 0.0
    assert pillar_margin({"NEUTRAL_CONTENT": 0.30}) == 0.30


def test_manual_review_for_ambiguous_low_margin_call():
    # Neutral and devotional pillars are nearly tied (margin 0.005) -- too
    # close to auto-resolve, regardless of which one nominally wins.
    clusters = {"safe_everyday_life": 0.10, "puja_rituals_and_temples": 0.095}
    res = evaluate_moderation(clusters)
    assert res.flagged is True
    assert res.action == ModerationAction.MANUAL_REVIEW.value
    assert any("low_confidence_pillar_margin" in r for r in res.reasons)


def test_manual_review_does_not_override_a_strict_block():
    # weapons_and_firearms crosses its own STRICT threshold (0.25) independent
    # of the pillar margin -- a real, evidence-based BLOCK must stand even
    # though the pillar margin here (0.30 vs 0.29 = 0.01) is thin.
    clusters = {"weapons_and_firearms": 0.30, "puja_rituals_and_temples": 0.29}
    res = evaluate_moderation(clusters)
    assert res.action == ModerationAction.BLOCK.value
    assert "weapons_and_firearms: 0.30" in res.reasons


def test_manual_review_not_triggered_when_margin_is_wide():
    clusters = {"modern_vehicles_and_tech": 0.60}
    res = evaluate_moderation(clusters)
    assert res.action == ModerationAction.BLOCK.value
    assert res.action != ModerationAction.MANUAL_REVIEW.value
