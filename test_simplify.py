from shapely.geometry import Polygon

p = Polygon([(400, -300), (500, -300), (500, -400), (400, -400)])
print('Before buffer:', len(p.exterior.coords))
b = p.buffer(0)
print('After buffer:', len(b.exterior.coords))
s = b.simplify(0.5)
print('After simplify:', len(s.exterior.coords))
