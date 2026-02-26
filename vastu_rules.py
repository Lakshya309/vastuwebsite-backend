# devta_microservice/vastu_rules.py

from typing import Dict, Literal, List

# Define general Vastu suitability for different object types in different regions
# This is a direct translation of lib/vastuRules.ts

# Type for direction (matching the frontend Direction type)
Direction = Literal[
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
    "ANY"
    
]

# Type for object rules (matching the frontend ObjectRules type)
ObjectRules = Dict[str, Dict[Direction, int]]

# The Vastu rules translated from lib/vastuRules.ts
VASTU_RULES: ObjectRules = {
  "TOILET": {
    "NE": -50,
    "NNE": -10,
    "E": -20,
    "ESE": 20,
    "SE": -20,
    "SSE": -20,
    "S": -20,
    "SSW": 20,
    "WSW": -10,
    "WNW": -20,
    "NW": -10,
    "NNW": -10,
    "N": -10
  },
  "DINING ROOM": {
    "NE": 10,
    "NNE": 10,
    "E": 10,
    "ESE": -10,
    "SE": 10,
    "SSE": 10,
    "S": 10,
    "SSW": 10,
    "SW": 10,
    "WSW": 10,
    "WNW": 10,
    "NW": 10,
    "NNW": 10,
    "N": 10
  },
  "SEPTIC TANK": {
    "NE": -20,
    "NNE": -20,
    "E": -20,
    "ESE": 20,
    "SE": -20,
    "SSE": -20,
    "S": -20,
    "SSW": 20,
    "SW": -20,
    "WSW": -20,
    "WNW": 20,
    "NW": -20,
    "NNW": -20,
    "N": -20
  },
  "FAMILY LOUNGE": {
    "NE": 10,
    "NNE": 10,
    "E": 10,
    "ESE": 10,
    "SE": -10,
    "SSE": 10,
    "S": 10,
    "SSW": -10,
    "SW": 10,
    "WSW": 10,
    "WNW": -10,
    "NW": 10,
    "NNW": 10,
    "N": 10
  },
  "STUDY TABLE": {
    "NE": 10,
    "NNE": 10,
    "E": 10,
    "ESE": -10,
    "SE": 10,
    "SSE": 10,
    "S": -10,
    "SSW": -10,
    "SW": 10,
    "WSW": 10,
    "WNW": -10,
    "NW": 10,
    "NNW": -10,
    "N": 10
  },
  "STORE ROOM": {
    "NE": -10,
    "NNE": -10,
    "E": -10,
    "ESE": 10,
    "SE": -10,
    "SSE": 10,
    "S": 10,
    "SSW": 10,
    "SW": -10,
    "WSW": 10,
    "WNW": 10,
    "NW": 10,
    "NNW": -10,
    "N": -10
  },
  "SERVENT ROOM": {
    "NE": 10,
    "NNE": 10,
    "E": 10,
    "ESE": 10,
    "SE": -10,
    "SSE": -10,
    "S": -10,
    "SSW": 10,
    "SW": -10,
    "WSW": -10,
    "WNW": 10,
    "NW": 10,
    "NNW": 10,
    "N": -10
  },
  "GUEST ROOM": {
    "NE": 10,
    "NNE": 10,
    "E": 10,
    "ESE": -10,
    "SE": -10,
    "SSE": 10,
    "S": -10,
    "SSW": -10,
    "SW": -10,
    "WSW": -10,
    "WNW": 10,
    "NW": 10,
    "NNW": 10,
    "N": -10
  },
  "BAR": {
    "NE": -10,
    "NNE": -10,
    "E": -10,
    "ESE": -10,
    "SE": -10,
    "SSE": -10,
    "S": -10,
    "SSW": 10,
    "SW": -10,
    "WSW": 10,
    "WNW": 10,
    "NW": 10,
    "NNW": 10,
    "N": -10
  },
  "STAIRCASE": {
    "NE": -20,
    "NNE": -20,
    "E": -20,
    "ESE": 10,
    "SE": 10,
    "SSE": -20,
    "S": 10,
    "SSW": 10,
    "SW": 10,
    "WSW": -20,
    "WNW": 10,
    "NW": 10,
    "NNW": -20,
    "N": -20
  },
  "PUJA": {
    "NE": 10,
    "NNE": 10,
    "E": 10,
    "ESE": -10,
    "SE": -10,
    "SSE": -10,
    "S": -10,
    "SSW": -10,
    "SW": -10,
    "WSW": 10,
    "WNW": -10,
    "NW": -10,
    "NNW": -10,
    "N": 10
  },
  "KITCHEN": {
    "NE": -20,
    "NNE": -10,
    "E": -20,
    "ESE": -20,
    "SE": 30,
    "SSE": 20,
    "S": 10,
    "SSW": -10,
    "SW": -10,
    "WSW": -10,
    "WNW": -10,
    "NW": -10,
    "NNW": -10,
    "N": -10
  },
  "MUSIC SYSTEM": {
    "NE": -5,
    "NNE": 5,
    "E": 5,
    "ESE": -5,
    "SE": 5,
    "SSE": 5,
    "S": 5,
    "SSW": 5,
    "SW": 5,
    "WSW": 5,
    "WNW": -5,
    "NW": 5,
    "NNW": -5,
    "N": 5
  },
  "INVETER": {
    "NE": -5,
    "NNE": -5,
    "E": -5,
    "ESE": 5,
    "SE": 5,
    "SSE": -5,
    "S": -5,
    "SSW": -5,
    "SW": -5,
    "WSW": -5,
    "WNW": 5,
    "NW": 5,
    "NNW": -5,
    "N": -5
  },
  "WATER HEATER": {
    "NE": -10,
    "NNE": -10,
    "E": 10,
    "ESE": -10,
    "SE": 10,
    "SSE": -10,
    "S": 10,
    "SSW": -10,
    "SW": -10,
    "WSW": 10,
    "WNW": 10,
    "NW": -10,
    "NNW": -10,
    "N": -10
  },
  "AIR CONDITIONER": {
    "NE": -10,
    "NNE": 10,
    "E": -10,
    "ESE": 10,
    "SE": -10,
    "SSE": 10,
    "S": 10,
    "SSW": 10,
    "SW": -10,
    "WSW": 10,
    "WNW": 10,
    "NW": 10,
    "NNW": -10,
    "N": 10
  },
  "MASTER BEDROOM": {
    "NE": -10,
    "NNE": -10,
    "E": 10,
    "ESE": -10,
    "SE": -10,
    "SSE": -10,
    "S": 20,
    "SSW": -20,
    "SW": 20,
    "WSW": 10,
    "WNW": -10,
    "NW": -10,
    "NNW": -10,
    "N": 10
  },
  "OVERHEAD TANK": {
    "NE": -30, "NNE": -20, "NE": -20, "ENE": -10,
    "E": -10, "ESE": 0, "SE": 10, "SSE": 20,
    "S": 30, "SSW": 40, "SW": 50, "WSW": 40,
    "W": 30, "WNW": 20, "NW": 10, "NNW": 0,
    "N": -10
  },
  "UNDERGROUND TANK": {
    "NE": 50, "NNE": 40, "NE": 30, "ENE": 20,
    "E": 10, "ESE": 0, "SE": -10, "SSE": -20,
    "S": -30, "SSW": -40, "SW": -50, "WSW": -40,
    "W": -30, "WNW": -20, "NW": -10, "NNW": 0,
    "N": 10
  }
}


def get_vastu_score_impact(object_type: str, direction: Direction) -> int:
    """
    Retrieves the Vastu score impact for a given object type and direction.
    Returns 0 if the object type or direction is not found.
    """
    # Normalize object_type to match the keys in VASTU_RULES
    normalized_object_type = object_type.upper() # Convert "Toilet" to "TOILET"

    # Handle cases where object_type from frontend might not directly match the rule keys
    # For example, "Toilet" from frontend matches "TOILET" in rules
    # I will need to refine this mapping if frontend object_type names are inconsistent
    if normalized_object_type == "TOILET":
        normalized_object_type = "TOILET"
    elif normalized_object_type == "STOVE":
        normalized_object_type = "KITCHEN"
    elif normalized_object_type == "BED":
        # For 'Bed', we might need to map it to 'MASTER BEDROOM' or create a specific 'BED' entry if rules vary
        # For now, let's assume 'BED' can map to 'MASTER BEDROOM' for scoring if no other rule exists.
        # This needs clarification with the user.
        normalized_object_type = "MASTER BEDROOM"
    elif normalized_object_type == "DINING":
        normalized_object_type = "DINING ROOM"
    elif normalized_object_type == "SOFA": # Assuming sofa is part of Family Lounge
        normalized_object_type = "FAMILY LOUNGE"
    elif normalized_object_type == "WARDROBE": # Wardrobe could be part of a bedroom, so let's use Master Bedroom for now
        normalized_object_type = "MASTER BEDROOM"
    elif normalized_object_type == "POOJA":
        normalized_object_type = "PUJA"
    elif normalized_object_type == "STAIRS":
        normalized_object_type = "STAIRCASE"
    elif normalized_object_type == "OVERHEADTANK":
        normalized_object_type = "OVERHEAD TANK"
    elif normalized_object_type == "UNDERGROUNDTANK":
        normalized_object_type = "UNDERGROUND TANK"

    object_rules = VASTU_RULES.get(normalized_object_type)
    if object_rules:
        return object_rules.get(direction, 0) # Default to 0 if direction not found
    return 0 # Default to 0 if object type not found

def get_vastu_verdict(score: int) -> Literal["EXCELLENT", "GOOD", "BAD", "CRITICAL"]:
    """
    Converts a score into a Vastu verdict.
    This can be refined with specific thresholds.
    """
    if score >= 10:
        return "EXCELLENT"
    elif score > 0:
        return "GOOD"
    elif score == 0:
        return "GOOD" # Neutral is also considered good in absence of negative impact
    elif score >= -10:
        return "BAD"
    else: # score < -10
        return "CRITICAL"
