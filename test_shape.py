from shapely.geometry import Polygon
import math
from shapely.affinity import rotate as shapely_rotate

def is_rectangular(poly: Polygon, north_base_rotation: float) -> bool:
    if poly.is_empty:
        return False
    min_rect = poly.minimum_rotated_rectangle
    if min_rect.area <= 0:
        return False
    ratio = poly.area / min_rect.area
    simplified = poly.simplify(0.5)
    vertex_count = len(simplified.exterior.coords) - 1
    
    # Old logic
    center = poly.centroid
    aligned = shapely_rotate(poly, -north_base_rotation, origin=center)
    minx, miny, maxx, maxy = aligned.bounds
    old_bbox_area = (maxx - minx) * (maxy - miny)
    old_ratio = poly.area / old_bbox_area
    old_simplified = poly.simplify(0.01)
    old_vertex_count = len(old_simplified.exterior.coords) - 1
    
    print(f'area: {poly.area}')
    print(f'NEW: min_rect_area: {min_rect.area}, ratio: {ratio}, vertex_count: {vertex_count}, result: {ratio > 0.95 and vertex_count == 4}')
    print(f'OLD: bbox_area: {old_bbox_area}, ratio: {old_ratio}, vertex_count: {old_vertex_count}, result: {old_ratio > 0.95 and old_vertex_count == 4}')
    return ratio > 0.95 and vertex_count == 4

p1 = Polygon([(400, 300), (500, 301), (501, 400), (400, 400)])
print('\nSlightly off square (N=0):')
is_rectangular(p1, 0)

p2 = Polygon([(400, 300), (450, 250), (500, 300), (450, 350)])
print('\nDiamond (Rotated square) (N=0):')
is_rectangular(p2, 0)
