from typing import Dict, Literal, TypedDict, Optional
import json
from pathlib import Path

data_path = Path(__file__).parent / "vastu_rules.json"

with open(data_path, "r") as f:
    VASTU_RULES = json.load(f)

Direction = Literal[
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW"
]


class VastuRule(TypedDict):
    score: int
    grade: Literal["A", "B", "C"]


ObjectRules = Dict[str, Dict[Direction, VastuRule]]


# -----------------------------
# Object name normalization map
# -----------------------------
OBJECT_ALIAS_MAP = {
    "STOVE": "KITCHEN",
    "BED": "MASTER BEDROOM",
    "DINING": "DINING ROOM",
    "SOFA": "FAMILY LOUNGE",
    "WARDROBE": "MASTER BEDROOM",
    "POOJA": "POOJA",
    "STAIRS": "STAIRCASE",
    "OVERHEADTANK": "OVERHEAD TANK",
    "UNDERGROUNDTANK": "UNDERGROUND TANK"
}

# The Vastu rules translated from lib/vastuRules.ts


# -----------------------------
# Normalize object name
# -----------------------------
def normalize_object_type(object_type: str) -> str:
    name = object_type.upper().strip()
    return OBJECT_ALIAS_MAP.get(name, name)


# -----------------------------
# Get rule
# -----------------------------
def get_vastu_rule(
    object_type: str,
    direction: Direction
) -> Optional[VastuRule]:

    normalized_object = normalize_object_type(object_type)

    object_rules = VASTU_RULES.get(normalized_object)

    if not object_rules:
        return None

    return object_rules.get(direction)


# -----------------------------
# Score impact
# -----------------------------
def get_vastu_score_impact(
    object_type: str,
    direction: Direction
) -> int:

    rule = get_vastu_rule(object_type, direction)

    if rule:
        return rule["score"]

    return 0


# -----------------------------
# Grade lookup
# -----------------------------
def get_vastu_grade(
    object_type: str,
    direction: Direction
) -> Optional[str]:

    rule = get_vastu_rule(object_type, direction)

    if rule:
        return rule["grade"]

    return None


# -----------------------------
# Remedy logic
# -----------------------------
def get_vastu_remedy(
    object_type: str,
    direction: Direction
) -> str:

    rule = get_vastu_rule(object_type, direction)

    if not rule:
        return "No rule available."

    grade = rule["grade"]

    if grade == "C":
        return f"CRITICAL: {object_type} in {direction} is a major Vastu Dosha. Immediate remedy or relocation required."
    elif grade == "B":
        return f"BAD: {object_type} in {direction} creates negative effects. Use Magic Rod or Magic Box as a remedy."
    elif grade == "A":
        return f"GOOD: {object_type} in {direction} is well-placed and beneficial."

    return "No remedy required."


# -----------------------------
# Score verdict
# -----------------------------
def get_vastu_verdict(score: int) -> Literal[
    "EXCELLENT", "GOOD", "BAD", "CRITICAL"
]:

    if score >= 10:
        return "EXCELLENT"

    if score > 0:
        return "GOOD"

    if score == 0:
        return "GOOD"

    if score >= -10:
        return "BAD"

    return "CRITICAL"