import urllib.request
import json

payload = {
  "boundary_normalized": [
    {"x": -25.0, "y": -25.0},
    {"x": 25.0, "y": -25.0},
    {"x": 25.0, "y": 25.0},
    {"x": -25.0, "y": 25.0}
  ],
  "north_direction": 0,
  "grid_type": "81",
  "placed_objects": []
}

req = urllib.request.Request(
    'http://localhost:5000/analyze_objects',
    data=json.dumps(payload).encode('utf-8'),
    headers={'content-type': 'application/json'}
)

try:
    with urllib.request.urlopen(req) as f:
        resp = json.loads(f.read().decode('utf-8'))
        print(len(resp.get('devta_areas_45', [])))
        # check if devtas have outer ring
        print([d['name'] for d in resp.get('devta_areas_45', [])][:6])
except Exception as e:
    print(e)
