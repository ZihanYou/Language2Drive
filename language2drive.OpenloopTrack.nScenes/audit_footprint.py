#!/usr/bin/env python3
"""Approximate vehicle-footprint coverage for witness review.

The ego-pose origin's exact axle offset is not calibrated here. Results are
diagnostic under the declared rectangle assumptions, not official safety.
"""

from paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, RELEASE_ROOT

import json
import math
from pathlib import Path

from shapely.geometry import Polygon
from build_lane_pcus import MapGeometry


ROOT = ARTIFACT_ROOT
SHAPES = {'assumed_rearward_origin':(-1.5,3.5,-1.0,1.0),
          'assumed_center_origin':(-2.5,2.5,-1.0,1.0)}


def body_polygon(xy,heading,shape):
    xmin,xmax,ymin,ymax = shape
    c,s = math.cos(heading),math.sin(heading)
    return Polygon([(xy[0]+c*x-s*y,xy[1]+s*x+c*y)
                    for x,y in ((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))])


def covered(geometry,polygon):
    # Nine-point approximation avoids expensive whole-map polygon unions.
    ring = polygon.exterior
    samples = [polygon.centroid]
    samples.extend(ring.interpolate(i*ring.length/8) for i in range(8))
    return all(geometry.drivable_at((p.x,p.y)) for p in samples)


def evaluate(geometry,points,anchor_yaw):
    headings = [anchor_yaw]
    for a,b in zip(points,points[1:]):
        headings.append(math.atan2(b[1]-a[1],b[0]-a[0]))
    result = {}
    for label,shape in SHAPES.items():
        flags = [covered(geometry,body_polygon(xy,heading,shape))
                 for xy,heading in zip(points,headings)]
        result[label] = {'covered_points':sum(flags),'total_points':len(flags),
                         'all_covered':all(flags)}
    return result


def main():
    maps = {}
    output = ROOT/'audit/footprint_diagnostics.jsonl'
    with (ROOT/'candidates/all_review_queue.jsonl').open() as source, output.open('w') as dest:
        for line in source:
            pcu = json.loads(line)
            location = pcu['public_input']['location']
            if location not in maps:
                maps[location] = MapGeometry(location)
            pose = pcu['public_input']['ego_pose']
            sidecar = pcu['private_sidecar']
            first,second = (('witness_change','witness_keep') if pcu['family']=='lane_change_keep'
                            else ('witness_low','witness_high'))
            result = {'pcu_id':pcu['pcu_id'],'status':'assumption_sensitive_diagnostic',
                      'dimensions_m':'length 5.0, width 2.0; 9 sampled body points',
                      'arm_0':evaluate(maps[location],sidecar[first]['global_xy_m'],pose['yaw_rad']),
                      'arm_1':evaluate(maps[location],sidecar[second]['global_xy_m'],pose['yaw_rad'])}
            dest.write(json.dumps(result)+'\n')
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
