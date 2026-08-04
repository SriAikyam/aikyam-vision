from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CategoryType(str, Enum):
    SAFETY_VIOLATIONS = "SAFETY_VIOLATIONS"
    COMMERCIAL_SPAM = "COMMERCIAL_SPAM"
    DISRUPTIVE_CONTENT = "DISRUPTIVE_CONTENT"
    NEUTRAL_CONTENT = "NEUTRAL_CONTENT"
    SACRED_DEVOTIONAL = "SACRED_DEVOTIONAL"


class ModerationAction(str, Enum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    FLAG = "FLAG"
    BLOCK = "BLOCK"


@dataclass
class ModerationRule:
    category_key: str
    name: str
    description: str
    threshold: float
    action: ModerationAction
    reason_code: str


@dataclass
class ModerationResult:
    flagged: bool = False
    action: str = ModerationAction.ALLOW.value
    primary_deity: str | None = None
    category_type_scores: dict[str, float] = field(default_factory=dict)
    total_prohibited_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)


# Category Types Mapping Registry
CLUSTER_TO_CATEGORY_TYPE: dict[str, CategoryType] = {
    # 1. SAFETY_VIOLATIONS
    "nudity_and_sexual_content": CategoryType.SAFETY_VIOLATIONS,
    "child_safety_risk": CategoryType.SAFETY_VIOLATIONS,
    "non_veg_and_meat": CategoryType.SAFETY_VIOLATIONS,
    "scam_and_phishing": CategoryType.SAFETY_VIOLATIONS,
    "fake_babas_and_occult_scams": CategoryType.SAFETY_VIOLATIONS,
    "disrespectful_temple_acts": CategoryType.SAFETY_VIOLATIONS,
    "footwear_in_sacred_space": CategoryType.SAFETY_VIOLATIONS,
    "substance_alcohol_drugs": CategoryType.SAFETY_VIOLATIONS,
    "violence_and_harm": CategoryType.SAFETY_VIOLATIONS,
    "hate_speech_and_abuse": CategoryType.SAFETY_VIOLATIONS,
    "animal_abuse": CategoryType.SAFETY_VIOLATIONS,
    "ocr_offensive_text": CategoryType.SAFETY_VIOLATIONS,

    # 2. COMMERCIAL_SPAM
    "spam_and_promotions": CategoryType.COMMERCIAL_SPAM,

    # 3. DISRUPTIVE_CONTENT
    "modern_nightlife_and_parties": CategoryType.DISRUPTIVE_CONTENT,

    # 4. NEUTRAL_CONTENT
    "modern_vehicles": CategoryType.NEUTRAL_CONTENT,
    "medical_and_healthcare": CategoryType.NEUTRAL_CONTENT,
    "everyday_safe_life": CategoryType.NEUTRAL_CONTENT,
    "safe": CategoryType.NEUTRAL_CONTENT,

    # 5. SACRED_DEVOTIONAL
    "shiva_iconography": CategoryType.SACRED_DEVOTIONAL,
    "ganesha_iconography": CategoryType.SACRED_DEVOTIONAL,
    "vishnu_krishna_iconography": CategoryType.SACRED_DEVOTIONAL,
    "devi_durga_iconography": CategoryType.SACRED_DEVOTIONAL,
    "hanuman_iconography": CategoryType.SACRED_DEVOTIONAL,
    "ram_sita_iconography": CategoryType.SACRED_DEVOTIONAL,
    "temple_architecture_and_deities": CategoryType.SACRED_DEVOTIONAL,
    "puja_and_aarti_rituals": CategoryType.SACRED_DEVOTIONAL,
    "bhajan_and_kirtan": CategoryType.SACRED_DEVOTIONAL,
    "festival_celebrations": CategoryType.SACRED_DEVOTIONAL,
    "sacred_yatra_and_pilgrimage": CategoryType.SACRED_DEVOTIONAL,
    "vedic_wisdom_and_astrology": CategoryType.SACRED_DEVOTIONAL,
    "prasadam_and_sattvic_food": CategoryType.SACRED_DEVOTIONAL,
    "spiritual_gurus_and_discourses": CategoryType.SACRED_DEVOTIONAL,
    "vedic_education_and_gurukul": CategoryType.SACRED_DEVOTIONAL,
    "classical_temple_arts": CategoryType.SACRED_DEVOTIONAL,
    "sacred_art_and_rangoli": CategoryType.SACRED_DEVOTIONAL,
    "temple_announcements_and_seva": CategoryType.SACRED_DEVOTIONAL,
}

# Deity Iconography Tag Map
DEITY_CLUSTER_MAP: dict[str, str] = {
    "shiva_iconography": "shiva",
    "ganesha_iconography": "ganesha",
    "vishnu_krishna_iconography": "krishna",
    "devi_durga_iconography": "durga",
    "hanuman_iconography": "hanuman",
    "ram_sita_iconography": "ram",
}

# Custom Moderation Rules Registry
STRICT_MODERATION_RULES: list[ModerationRule] = [
    ModerationRule(
        category_key="nudity_and_sexual_content",
        name="Nudity & Sexual Content",
        description="Explicit nudity, intimate poses, or suggestive exposure",
        threshold=0.20,
        action=ModerationAction.BLOCK,
        reason_code="EXPLICIT_NUDITY_BLOCK",
    ),
    ModerationRule(
        category_key="child_safety_risk",
        name="Child Safety Risk",
        description="Unsupervised minor in hazardous environment",
        threshold=0.15,
        action=ModerationAction.BLOCK,
        reason_code="CHILD_SAFETY_BLOCK",
    ),
    ModerationRule(
        category_key="non_veg_and_meat",
        name="Meat & Non-Vegetarian Content",
        description="Raw meat, butchery, poultry, or non-veg food (Sattvic prohibition)",
        threshold=0.25,
        action=ModerationAction.BLOCK,
        reason_code="NON_VEG_MEAT_BLOCK",
    ),
    ModerationRule(
        category_key="scam_and_phishing",
        name="Scams & Phishing Offers",
        description="Fake prize banners, QR codes, or unverified money requests",
        threshold=0.30,
        action=ModerationAction.BLOCK,
        reason_code="SCAM_PHISHING_BLOCK",
    ),
    ModerationRule(
        category_key="fake_babas_and_occult_scams",
        name="Occult & Black Magic Scams",
        description="Superstitious scams, Vashikaran, or black magic claims",
        threshold=0.25,
        action=ModerationAction.BLOCK,
        reason_code="OCCULT_SCAM_BLOCK",
    ),
    ModerationRule(
        category_key="disrespectful_temple_acts",
        name="Disrespectful Temple Behavior",
        description="Inappropriate conduct or mocking inside sacred temple sanctums",
        threshold=0.25,
        action=ModerationAction.BLOCK,
        reason_code="DISRESPECTFUL_TEMPLE_ACT",
    ),
    ModerationRule(
        category_key="modern_nightlife_and_parties",
        name="Nightlife & Party Content",
        description="Clubs, pub drinking, or loud nightlife party dancing",
        threshold=0.60,
        action=ModerationAction.BLOCK,
        reason_code="NIGHTLIFE_PARTY_BLOCK",
    ),
    ModerationRule(
        category_key="modern_nightlife_and_parties",
        name="Nightlife & Party Content",
        description="Clubs, pub drinking, or loud nightlife party dancing",
        threshold=0.40,
        action=ModerationAction.FLAG,
        reason_code="NIGHTLIFE_PARTY_FLAGGED",
    ),
    ModerationRule(
        category_key="footwear_in_sacred_space",
        name="Footwear in Sacred Space",
        description="Wearing shoes or sandals inside temple sanctums or near idols",
        threshold=0.30,
        action=ModerationAction.FLAG,
        reason_code="FOOTWEAR_IN_SACRED_SPACE",
    ),
    ModerationRule(
        category_key="substance_alcohol_drugs",
        name="Alcohol, Drugs & Smoking",
        description="Liquor bottles, smoking, pub drinking, or illicit substances",
        threshold=0.30,
        action=ModerationAction.FLAG,
        reason_code="SUBSTANCE_FLAGGED",
    ),
    ModerationRule(
        category_key="spam_and_promotions",
        name="Spam & Commercial Ads",
        description="Marketing flyers, coupon codes, and commercial advertisements",
        threshold=0.40,
        action=ModerationAction.FLAG,
        reason_code="SPAM_FLAGGED",
    ),
    ModerationRule(
        category_key="violence_and_harm",
        name="Violence & Weapons",
        description="Weapons, blood, violent assault, or bodily harm",
        threshold=0.40,
        action=ModerationAction.FLAG,
        reason_code="VIOLENCE_FLAGGED",
    ),
    ModerationRule(
        category_key="hate_speech_and_abuse",
        name="Hate Speech & Vulgar Slurs",
        description="Discriminatory text, vulgar slurs, or hate speech",
        threshold=0.35,
        action=ModerationAction.FLAG,
        reason_code="HATE_FLAGGED",
    ),
    ModerationRule(
        category_key="animal_abuse",
        name="Animal Cruelty & Harm",
        description="Confined, wounded, or mistreated animals",
        threshold=0.30,
        action=ModerationAction.FLAG,
        reason_code="ANIMAL_ABUSE_FLAGGED",
    ),
]


def detect_primary_deity(clusters: dict[str, float], threshold: float = 0.15) -> str | None:
    """Detects the primary deity based on deity iconography scores."""
    best_deity = None
    highest_score = 0.0

    for cluster_key, deity_tag in DEITY_CLUSTER_MAP.items():
        score = clusters.get(cluster_key, 0.0)
        if score >= threshold and score > highest_score:
            highest_score = score
            best_deity = deity_tag

    return best_deity


def calculate_category_type_scores(clusters: dict[str, float]) -> dict[str, float]:
    """Aggregates cluster scores into the 5 high-level Category Types."""
    type_sums: dict[str, float] = {
        CategoryType.SAFETY_VIOLATIONS.value: 0.0,
        CategoryType.COMMERCIAL_SPAM.value: 0.0,
        CategoryType.DISRUPTIVE_CONTENT.value: 0.0,
        CategoryType.NEUTRAL_CONTENT.value: 0.0,
        CategoryType.SACRED_DEVOTIONAL.value: 0.0,
    }

    for cluster_key, score in clusters.items():
        cat_type = CLUSTER_TO_CATEGORY_TYPE.get(cluster_key, CategoryType.NEUTRAL_CONTENT)
        type_sums[cat_type.value] = type_sums.get(cat_type.value, 0.0) + float(score)

    return {k: round(v, 4) for k, v in type_sums.items()}


def evaluate_moderation(clusters: dict[str, float]) -> ModerationResult:
    """
    Evaluates safety rules, strict category overrides, cumulative scores, and primary deity.
    """
    result = ModerationResult()
    result.category_type_scores = calculate_category_type_scores(clusters)
    result.primary_deity = detect_primary_deity(clusters)

    # 1. Check Custom Moderation Rules
    for rule in STRICT_MODERATION_RULES:
        score = clusters.get(rule.category_key, 0.0)
        if score >= rule.threshold:
            result.flagged = True
            reason_entry = f"{rule.category_key}: {score:.2f}"
            if reason_entry not in result.reasons:
                result.reasons.append(reason_entry)

            if rule.action == ModerationAction.BLOCK:
                result.action = ModerationAction.BLOCK.value
                if rule.reason_code not in result.blocked_reasons:
                    result.blocked_reasons.append(rule.reason_code)
            elif result.action != ModerationAction.BLOCK.value and rule.action == ModerationAction.FLAG:
                result.action = ModerationAction.FLAG.value

    # 2. Check Cumulative Safety Score
    safety_sum = result.category_type_scores.get(CategoryType.SAFETY_VIOLATIONS.value, 0.0)
    result.total_prohibited_score = safety_sum

    if safety_sum >= 0.70:
        result.flagged = True
        result.action = ModerationAction.BLOCK.value
        reason_entry = f"safety_violations_cumulative: {safety_sum:.2f}"
        if reason_entry not in result.reasons:
            result.reasons.append(reason_entry)
    elif safety_sum >= 0.40 and result.action != ModerationAction.BLOCK.value:
        result.flagged = True
        result.action = ModerationAction.FLAG.value
        reason_entry = f"safety_violations_cumulative: {safety_sum:.2f}"
        if reason_entry not in result.reasons:
            result.reasons.append(reason_entry)
    elif safety_sum >= 0.10 and result.action == ModerationAction.ALLOW.value:
        result.action = ModerationAction.WARN.value

    return result
