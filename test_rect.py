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
    print(f'area: {poly.area}, min_rect_area: {min_rect.area}, ratio: {ratio}, vertex_count: {vertex_count}, coords: {len(poly.exterior.coords)}')
    return ratio > 0.95 and vertex_count == 4

p = Polygon([(400, -300), (500, -300), (500, -400), (400, -400)])
print('Square aligned (N=0):', is_rectangular(p, 0))
