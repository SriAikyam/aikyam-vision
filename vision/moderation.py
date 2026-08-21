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


# Category Types Mapping Registry across 3 Pillars
CLUSTER_TO_CATEGORY_TYPE: dict[str, CategoryType] = {
    # 1. PILLAR 1: SACRED_DEVOTIONAL
    "shiva_iconography": CategoryType.SACRED_DEVOTIONAL,
    "krishna_iconography": CategoryType.SACRED_DEVOTIONAL,
    "vishnu_iconography": CategoryType.SACRED_DEVOTIONAL,
    "venkateswara_iconography": CategoryType.SACRED_DEVOTIONAL,
    "rama_iconography": CategoryType.SACRED_DEVOTIONAL,
    "hanuman_iconography": CategoryType.SACRED_DEVOTIONAL,
    "ganesha_iconography": CategoryType.SACRED_DEVOTIONAL,
    "devi_iconography": CategoryType.SACRED_DEVOTIONAL,
    "puja_rituals_and_temples": CategoryType.SACRED_DEVOTIONAL,
    "festivals_and_celebrations": CategoryType.SACRED_DEVOTIONAL,
    "prasad_and_sacred_food": CategoryType.SACRED_DEVOTIONAL,

    # 2. PILLAR 2: SAFETY_VIOLATIONS & COMMERCIAL_SPAM
    "nudity_and_sexual_content": CategoryType.SAFETY_VIOLATIONS,
    "physical_violence_and_harm": CategoryType.SAFETY_VIOLATIONS,
    "weapons_and_firearms": CategoryType.SAFETY_VIOLATIONS,
    "drugs_and_alcohol": CategoryType.SAFETY_VIOLATIONS,
    "non_vegetarian_food": CategoryType.SAFETY_VIOLATIONS,
    "commercial_spam_and_flyers": CategoryType.COMMERCIAL_SPAM,
    "modern_nightlife_and_parties": CategoryType.DISRUPTIVE_CONTENT,

    # 3. PILLAR 3: NEUTRAL_CONTENT
    "safe_everyday_life": CategoryType.NEUTRAL_CONTENT,
    "modern_vehicles_and_tech": CategoryType.NEUTRAL_CONTENT,
    "sports_and_recreation": CategoryType.NEUTRAL_CONTENT,
    "medical_and_healthcare": CategoryType.NEUTRAL_CONTENT,
    "christianity_iconography": CategoryType.NEUTRAL_CONTENT,
    "islam_iconography": CategoryType.NEUTRAL_CONTENT,
}

# Deity Iconography Tag Map
DEITY_CLUSTER_MAP: dict[str, str] = {
    "shiva_iconography": "shiva",
    "krishna_iconography": "krishna",
    "vishnu_iconography": "vishnu",
    "venkateswara_iconography": "venkateswara",
    "rama_iconography": "rama",
    "hanuman_iconography": "hanuman",
    "ganesha_iconography": "ganesha",
    "devi_iconography": "devi",
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
        category_key="physical_violence_and_harm",
        name="Physical Violence & Harm",
        description="Physical fights, assault, blood, or severe trauma",
        threshold=0.25,
        action=ModerationAction.BLOCK,
        reason_code="VIOLENCE_HARM_BLOCK",
    ),
    ModerationRule(
        category_key="weapons_and_firearms",
        name="Weapons & Firearms",
        description="Handguns, rifles, knives, or dangerous weapons",
        threshold=0.25,
        action=ModerationAction.BLOCK,
        reason_code="WEAPONS_FIREARMS_BLOCK",
    ),
    ModerationRule(
        category_key="drugs_and_alcohol",
        name="Drugs, Alcohol & Substance Abuse",
        description="Liquor bottles, pub drinking, smoking, or illicit narcotics",
        threshold=0.30,
        action=ModerationAction.FLAG,
        reason_code="SUBSTANCE_FLAGGED",
    ),
    ModerationRule(
        category_key="commercial_spam_and_flyers",
        name="Commercial Spam & Marketing Flyers",
        description="Marketing flyers, coupon codes, discount banners, and commercial ads",
        threshold=0.40,
        action=ModerationAction.FLAG,
        reason_code="SPAM_FLAGGED",
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
        category_key="non_vegetarian_food",
        name="Non-Vegetarian Food",
        description="Raw meat, butchery, poultry, seafood, or non-veg food (Sattvic prohibition)",
        threshold=0.25,
        action=ModerationAction.BLOCK,
        reason_code="NON_VEG_MEAT_BLOCK",
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
