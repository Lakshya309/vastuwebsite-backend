"""
Vastu Spatial Engine – Production Grade v5.2

Includes:
• 45 Devta Mandala (Traditional + Hybrid)
• 16 Direction Zones
• 8 Direction Zones

Coordinate system fix:
• Canvas uses Y-down; Shapely uses Y-up.
• All input polygons have Y negated before processing (canvas → math coords).
• All output polygons have Y negated back (math → canvas coords).
• north_direction from the frontend is a clockwise canvas angle.
  In Y-up math coords (CCW-positive), clockwise rotation = adding the angle,
  so we ADD north_base_rotation instead of subtracting it.
"""

import math
import uvicorn
from typing import List, Optional, Literal, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shapely.geometry import Polygon, Point, MultiPolygon, box
from shapely.affinity import scale as shapely_scale

from vastu_rules import get_vastu_score_impact, get_vastu_verdict, Direction, VASTU_RULES

# ======================================================
# FASTAPI
# ======================================================

app = FastAPI(title="Vastu Spatial Engine", version="5.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# MODELS
# ======================================================

class PointModel(BaseModel):
    x: float
    y: float

class Region(BaseModel):
    id: str
    name: str
    polygon: List[PointModel]
    ring: str
    startAngle: Optional[float] = None
    endAngle: Optional[float] = None
    source: str

class PlacedObject(BaseModel):
    id: str
    object_type: str
    boundary_normalized: List[PointModel]
    centroid: PointModel
    rotation: Optional[float] = None

class AnalyzedObjectResult(BaseModel):
    object_id: str
    object_type: str
    devta_region: Optional[str] = None
    zone16_direction: Optional[Direction] = None
    score_impact: int
    verdict: Literal["EXCELLENT", "GOOD", "BAD", "CRITICAL"]
    message: str

class DevtaArea(BaseModel):
    name: str
    area: float
    percentage: float

class VastuAnalysisResult(BaseModel):
    analyzed_objects: List[AnalyzedObjectResult]
    total_score: int
    overall_percentage: float
    overall_verdict: Literal["EXCELLENT", "GOOD", "BAD", "CRITICAL"]
    devta_areas_32: List[DevtaArea] = []
    devta_areas_45: List[DevtaArea] = []
    zone_areas_16: List[DevtaArea] = []
    zone_boundary_16: List[DevtaArea] = []
    zones16: List[Region] = [] # Added for frontend canvas support

class AnalysisRequest(BaseModel):
    boundary_normalized: List[PointModel]
    north_direction: float = Field(default=0.0)

class ObjectAnalysisRequest(AnalysisRequest):
    placed_objects: List[PlacedObject]

class AnalysisResponse(BaseModel):
    devtas45: List[Region]
    zones16: List[Region]
    zones8: List[Region]
    plot_centroid: Optional[PointModel] = None

# ======================================================
# CONSTANTS
# ======================================================

CENTER_DEVTA = "Brahma"

MIDDLE_DEVTAS = [
    "Bhudhar", "Aap", "Aapvatsa",
    "Aryama", "Savita", "Savitra",
    "Vivasvan", "Indra", "Indrajaya",
    "Mitra", "Rudra", "Rudrajaya"
]

OUTER_DEVTAS = [
    "Soma", "Bhujag", "Aditi", "Diti",
    "Shikhi", "Parjanya", "Jayanta", "Mahendra",
    "Surya", "Satya", "Bhrisha", "Antariksha",
    "Agni", "Pusha", "Vitatha", "Gruhakshata",
    "Yama", "Gandharva", "Bhringaraja", "Mriga",
    "Pitru", "Dauvarika", "Sugriva", "Pushpadanta",
    "Varuna", "Asura", "Shosha", "Papyakshama",
    "Roga", "Naga", "Mukhya", "Bhallata"
]

ZONE_NAMES_16: List[Direction] = [
    "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S",
    "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N"
]

ZONE_NAMES_8 = ["NE", "E", "SE", "S", "SW", "W", "NW", "N"]

# ======================================================
# COORDINATE HELPERS
# ======================================================
#
# Canvas:  +Y is DOWN  (screen/canvas convention)
# Shapely: +Y is UP    (standard math convention)
#
# Fix: negate Y on input so geometry is in math coords,
#      negate Y again on output to restore canvas coords.
#
# Rotation fix:
#   north_direction is the clockwise angle (on canvas) to True North.
#   In angular_wedge(), Vastu angles are converted to math angles via (90 - angle).
#   A Vastu angle increase (clockwise) means a math angle decrease (CCW).
#   To rotate the whole mandala clockwise by north_direction degrees,
#   we ADD north_direction to the base Vastu angles — this shifts each
#   wedge's Vastu angle forward, which (90 - angle) then maps correctly
#   to a CCW math shift, netting a CW rotation on the canvas.

def to_polygon(pts: List[PointModel]) -> Polygon:
    """Canvas coords (Y-down) → Shapely polygon in math coords (Y-up)."""
    if not pts:
        return Polygon()
    return Polygon([(p.x, -p.y) for p in pts]).buffer(0)


def to_points(poly) -> List[PointModel]:
    """Shapely polygon (math coords, Y-up) → canvas coords (Y-down)."""
    if poly.is_empty:
        return []
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.exterior:
        return [PointModel(x=x, y=-y) for x, y in list(poly.exterior.coords)[:-1]]
    return []

# ======================================================
# HELPERS
# ======================================================

def normalize_angle(a: float) -> float:
    return a % 360


def visual_center(poly: Polygon) -> Point:
    """Visual center in math coords (Y-up)."""
    c = poly.centroid
    return c if poly.contains(c) else poly.representative_point()


def get_angle_from_point(center: Point, p: PointModel) -> float:
    """
    Vastu angle (North=0, clockwise) from a math-coord center to a canvas-coord point.

    center : math coords (Y-up), from visual_center().
    p      : canvas coords (Y-down) — negate p.y to bring into math coords.
    """
    dx = p.x - center.x
    dy = (-p.y) - center.y  # convert canvas Y → math Y

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360

    # (0=East, CCW) → (0=North, CW)
    vastu_angle = (90 - angle_deg + 360) % 360
    return vastu_angle


def get_zone_from_angle(
    angle: float,
    north_base_rotation: float,
    zones_names: List[Direction]
) -> Optional[Direction]:
    """
    Returns the direction zone for a Vastu angle, consistent with
    the ADD-rotation convention used in generate_zones / generate_45_devtas.
    """
    num_zones = len(zones_names)
    if num_zones == 0:
        return None

    absolute_start = 0.0
    if zones_names == ZONE_NAMES_16:
        absolute_start = 11.25
    elif zones_names == ZONE_NAMES_8:
        absolute_start = 22.5

    step = 360 / num_zones

    # Invert the ADD rotation applied during generation
    angle_in_unrotated_mandala = (angle - north_base_rotation + 360) % 360

    for i, name in enumerate(zones_names):
        base_start = absolute_start + i * step
        base_end = absolute_start + (i + 1) * step

        if base_end >= 360 and base_start < 360:
            if (base_start <= angle_in_unrotated_mandala < 360) or \
               (0 <= angle_in_unrotated_mandala < (base_end % 360)):
                return name
        elif base_start <= angle_in_unrotated_mandala < base_end:
            return name

    return None

# ======================================================
# GEOMETRY CORE
# ======================================================

def angular_wedge(boundary: Polygon, center: Point, a1: float, a2: float, r: float = 2000):
    """
    Clips a wedge from `boundary` spanning Vastu angles a1 → a2.
    All coords are math (Y-up); (90 - angle) maps Vastu → standard math angle correctly.
    """
    a1r = math.radians(90 - a1)
    a2r = math.radians(90 - a2)

    p1 = (center.x + r * math.cos(a1r), center.y + r * math.sin(a1r))
    p2 = (center.x + r * math.cos(a2r), center.y + r * math.sin(a2r))

    wedge = Polygon([(center.x, center.y), p1, p2])
    clipped = wedge.intersection(boundary)

    if clipped.is_empty:
        return None
    if isinstance(clipped, MultiPolygon):
        return max(clipped.geoms, key=lambda g: g.area)
    return clipped


def largest_inner_rectangle(poly: Polygon) -> Polygon:
    minx, miny, maxx, maxy = poly.bounds
    core = box(minx, miny, maxx, maxy).intersection(poly)
    if core.area < poly.area * 0.4:
        return shapely_scale(poly, 0.7, 0.7, origin=visual_center(poly))
    return core

# ======================================================
# 45 DEVTA ENGINE
# ======================================================

def generate_45_devtas(poly: Polygon, north_base_rotation: float, tag: str = "traditional"):
    """
    poly is in math coords (Y-up).

    Rotation rule:
      ADD north_base_rotation to each base Vastu angle.
      This rotates the whole mandala clockwise on the canvas by north_base_rotation degrees,
      so that the mandala's North aligns with the plot's True North direction.
    """
    devtas = []
    center = visual_center(poly)
    did = 1

    inner = shapely_scale(poly, 0.33, 0.33, origin=center)
    middle = shapely_scale(poly, 0.66, 0.66, origin=center)

    devtas.append(Region(
        id=f"d-{did}", name=CENTER_DEVTA,
        polygon=to_points(inner),
        ring="center", source=tag
    ))
    did += 1

    # --- Middle ring: 12 devtas ---
    step_mid = 360 / 12
    absolute_start_middle = 345.0  # Bhudhar centered at North (0°)

    for i, name in enumerate(MIDDLE_DEVTAS):
        base_start = absolute_start_middle + i * step_mid
        base_end = absolute_start_middle + (i + 1) * step_mid

        start_angle = (base_start + north_base_rotation) % 360
        end_angle = (base_end + north_base_rotation) % 360

        w = angular_wedge(poly, center, start_angle, end_angle)
        if w:
            w = w.intersection(middle).difference(inner)
            if not w.is_empty:
                devtas.append(Region(
                    id=f"d-{did}", name=name,
                    polygon=to_points(w),
                    ring="middle",
                    startAngle=normalize_angle(start_angle),
                    endAngle=normalize_angle(end_angle),
                    source=tag
                ))
                did += 1

    # --- Outer ring: 32 devtas ---
    step_out = 360 / 32
    absolute_start_outer = 354.375  # Soma centered at North (0°)

    for i, name in enumerate(OUTER_DEVTAS):
        base_start = absolute_start_outer + i * step_out
        base_end = absolute_start_outer + (i + 1) * step_out

        start_angle = (base_start + north_base_rotation) % 360
        end_angle = (base_end + north_base_rotation) % 360

        w = angular_wedge(poly, center, start_angle, end_angle)
        if w:
            w = w.difference(middle)
            if not w.is_empty:
                devtas.append(Region(
                    id=f"d-{did}", name=name,
                    polygon=to_points(w),
                    ring="outer",
                    startAngle=normalize_angle(start_angle),
                    endAngle=normalize_angle(end_angle),
                    source=tag
                ))
                did += 1

    return devtas

# ======================================================
# 8 / 16 ZONE ENGINE
# ======================================================

def generate_zones(
    poly: Polygon,
    north_base_rotation: float,
    names: List[Direction],
    label: str
):
    """
    poly is in math coords (Y-up).
    ADD north_base_rotation to rotate zones to match the plot's True North.
    """
    zones = []
    center = visual_center(poly)
    step = 360 / len(names)

    absolute_start = 0.0
    if label == "zone16":
        absolute_start = 11.25
    elif label == "zone8":
        absolute_start = 22.5

    for i, name in enumerate(names):
        base_start = absolute_start + i * step
        base_end = absolute_start + (i + 1) * step

        start_angle = (base_start + north_base_rotation) % 360
        end_angle = (base_end + north_base_rotation) % 360

        w = angular_wedge(poly, center, start_angle, end_angle)
        if w:
            zones.append(Region(
                id=f"{label}-{i+1}",
                name=name,
                polygon=to_points(w),
                ring=label,
                startAngle=normalize_angle(start_angle),
                endAngle=normalize_angle(end_angle),
                source="directional"
            ))
    return zones

# ======================================================
# OBJECT ANALYSIS
# ======================================================

def analyze_objects(req: ObjectAnalysisRequest) -> VastuAnalysisResult:
    outer_polygon = to_polygon(req.boundary_normalized)  # math coords

    if outer_polygon.is_empty:
        return VastuAnalysisResult(
            analyzed_objects=[],
            total_score=0,
            overall_percentage=100.0,
            overall_verdict="GOOD",
            zones16=[]
        )

    if len(req.boundary_normalized) <= 4:
        devtas45_regions = generate_45_devtas(outer_polygon, req.north_direction)
    else:
        core_polygon = largest_inner_rectangle(outer_polygon)
        devtas45_regions = generate_45_devtas(core_polygon, req.north_direction, "traditional-core")

    zones16_regions = generate_zones(outer_polygon, req.north_direction, ZONE_NAMES_16, "zone16")

    analyzed_objects: List[AnalyzedObjectResult] = []
    total_score = 0

    max_rule_score = 0
    min_rule_score = 0
    has_rules = False
    for obj_type_rules in VASTU_RULES.values():
        for score in obj_type_rules.values():
            if score > max_rule_score:
                max_rule_score = score
            if score < min_rule_score:
                min_rule_score = score
            has_rules = True

    if not has_rules:
        max_rule_score = 100
        min_rule_score = -100

    if req.placed_objects:
        for obj in req.placed_objects:
            obj_polygon = to_polygon(obj.boundary_normalized)  # math coords

            if obj_polygon.is_empty:
                analyzed_objects.append(AnalyzedObjectResult(
                    object_id=obj.id,
                    object_type=obj.object_type,
                    devta_region=None,
                    zone16_direction=None,
                    score_impact=0,
                    verdict="CRITICAL",
                    message="Object has no valid geometry or is outside plot boundary."
                ))
                continue

            # Devta region: largest intersection
            # devtas45_regions polygons are stored in canvas coords (to_points flipped them),
            # so to_polygon() on them flips back to math coords for correct intersection.
            devta_name: Optional[str] = None
            max_devta_area = 0.0
            for devta_region in devtas45_regions:
                region_poly = to_polygon(devta_region.polygon)  # canvas → math
                if not region_poly.is_empty and obj_polygon.intersects(region_poly):
                    intersection = obj_polygon.intersection(region_poly)
                    if not intersection.is_empty and intersection.area > max_devta_area:
                        max_devta_area = intersection.area
                        devta_name = devta_region.name

            # 16-zone direction: largest intersection
            zone16_direction: Optional[Direction] = None
            max_zone16_area = 0.0
            for zone_region in zones16_regions:
                region_poly = to_polygon(zone_region.polygon)  # canvas → math
                if not region_poly.is_empty and obj_polygon.intersects(region_poly):
                    intersection = obj_polygon.intersection(region_poly)
                    if not intersection.is_empty and intersection.area > max_zone16_area:
                        max_zone16_area = intersection.area
                        zone16_direction = zone_region.name

            score_impact = 0
            verdict: Literal["EXCELLENT", "GOOD", "BAD", "CRITICAL"] = "GOOD"
            message = "Placement analyzed."

            if zone16_direction:
                score_impact = get_vastu_score_impact(obj.object_type, zone16_direction)
                verdict = get_vastu_verdict(score_impact)
                if score_impact > 0:
                    message = f"Good placement for {obj.object_type} in {zone16_direction}."
                elif score_impact < 0:
                    message = f"Problematic placement for {obj.object_type} in {zone16_direction}."
                else:
                    message = f"Neutral placement for {obj.object_type} in {zone16_direction}."
            else:
                # Centroid fallback
                plot_center_point = visual_center(outer_polygon)  # math coords
                obj_angle = get_angle_from_point(plot_center_point, obj.centroid)
                fallback_direction = get_zone_from_angle(obj_angle, req.north_direction, ZONE_NAMES_16)

                if fallback_direction:
                    zone16_direction = fallback_direction
                    score_impact = get_vastu_score_impact(obj.object_type, zone16_direction)
                    verdict = get_vastu_verdict(score_impact)
                    message = f"Zone by centroid fallback for {obj.object_type} in {zone16_direction}."
                else:
                    message = "Could not determine zone. Defaulting to neutral."
                    verdict = "BAD"
                    score_impact = 0

            analyzed_objects.append(AnalyzedObjectResult(
                object_id=obj.id,
                object_type=obj.object_type,
                devta_region=devta_name,
                zone16_direction=zone16_direction,
                score_impact=score_impact,
                verdict=verdict,
                message=message
            ))
            total_score += score_impact

    num_objects = len(req.placed_objects)
    overall_percentage = 0.0

    if num_objects > 0:
        min_overall = min_rule_score * num_objects
        max_overall = max_rule_score * num_objects
        if (max_overall - min_overall) > 0:
            overall_percentage = ((total_score - min_overall) / (max_overall - min_overall)) * 100.0
        else:
            overall_percentage = 100.0 if total_score >= 0 else 0.0
    else:
        overall_percentage = 100.0

    overall_percentage = max(0.0, min(100.0, overall_percentage))

    overall_verdict: Literal["EXCELLENT", "GOOD", "BAD", "CRITICAL"] = "GOOD"
    if overall_percentage >= 90:
        overall_verdict = "EXCELLENT"
    elif overall_percentage >= 60:
        overall_verdict = "GOOD"
    elif overall_percentage >= 30:
        overall_verdict = "BAD"
    else:
        overall_verdict = "CRITICAL"

    # Calculate Devta Areas for Graphs
    devta_areas_45: List[DevtaArea] = []
    devta_areas_32: List[DevtaArea] = []
    zone_areas_16: List[DevtaArea] = []
    zone_boundary_16: List[DevtaArea] = []
    total_area = outer_polygon.area

    if total_area > 0:
        # Calculate for all 45 devtas
        for devta in devtas45_regions:
            poly = to_polygon(devta.polygon)
            area = poly.area
            devta_areas_45.append(DevtaArea(
                name=devta.name,
                area=round(area, 4),
                percentage=round((area / total_area) * 100, 2)
            ))
        
        # Calculate specifically for outer 32 devtas
        for devta in devtas45_regions:
            if devta.ring == "outer":
                poly = to_polygon(devta.polygon)
                area = poly.area
                devta_areas_32.append(DevtaArea(
                    name=devta.name,
                    area=round(area, 4),
                    percentage=round((area / total_area) * 100, 2)
                ))

        # Calculate for 16 Zones Areas
        for zone in zones16_regions:
            poly = to_polygon(zone.polygon)
            area = poly.area
            zone_areas_16.append(DevtaArea(
                name=zone.name,
                area=round(area, 4),
                percentage=round((area / total_area) * 100, 2)
            ))

        # Calculate for 16 Zones Boundary Distribution
        boundary_line = outer_polygon.exterior
        total_perimeter = boundary_line.length
        if total_perimeter > 0:
            for zone in zones16_regions:
                # We can't use the zone polygon directly because it's clipped to the boundary.
                # Instead, we need the original wedge to see which part of the boundary_line it covers.
                # Actually, the zone polygon's intersection with the boundary exterior should give us the boundary segment.
                zone_poly = to_polygon(zone.polygon)
                boundary_in_zone = boundary_line.intersection(zone_poly)
                segment_length = boundary_in_zone.length
                zone_boundary_16.append(DevtaArea(
                    name=zone.name,
                    area=round(segment_length, 4), # using 'area' field for length here for model compatibility
                    percentage=round((segment_length / total_perimeter) * 100, 2)
                ))

    return VastuAnalysisResult(
        analyzed_objects=analyzed_objects,
        total_score=total_score,
        overall_percentage=round(overall_percentage, 2),
        overall_verdict=overall_verdict,
        devta_areas_45=devta_areas_45,
        devta_areas_32=devta_areas_32,
        zone_areas_16=zone_areas_16,
        zone_boundary_16=zone_boundary_16,
        zones16=zones16_regions
    )

# ======================================================
# MAIN PIPELINE
# ======================================================

def analyze_plot(req: AnalysisRequest) -> AnalysisResponse:
    outer = to_polygon(req.boundary_normalized)  # math coords (Y-up)
    center = visual_center(outer)                 # math coords (Y-up)

    if len(req.boundary_normalized) <= 4:
        devtas = generate_45_devtas(outer, req.north_direction)
    else:
        core = largest_inner_rectangle(outer)
        devtas = generate_45_devtas(core, req.north_direction, "traditional-core")

    return AnalysisResponse(
        devtas45=devtas,
        zones16=generate_zones(outer, req.north_direction, ZONE_NAMES_16, "zone16"),
        zones8=generate_zones(outer, req.north_direction, ZONE_NAMES_8, "zone8"),
        plot_centroid=PointModel(x=center.x, y=-center.y),  # math → canvas coords
    )

# ======================================================
# API
# ======================================================

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalysisRequest):
    return analyze_plot(req)

@app.post("/analyze_objects", response_model=VastuAnalysisResult)
async def analyze_objects_endpoint(req: ObjectAnalysisRequest):
    return analyze_objects(req)

@app.get("/health")
def health():
    return {"status": "ok", "engine": "vastu-spatial-v5.2"}

# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)