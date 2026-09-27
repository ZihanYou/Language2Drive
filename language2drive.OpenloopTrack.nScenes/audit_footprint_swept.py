#!/usr/bin/env python3
"""Check full swept vehicle rectangles against local nuScenes drivable polygons.

Two declared ego-pose-origin assumptions are tested. A 0.2 m map tolerance is
applied. This still does not prove dynamic collision freedom.
"""

from paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, RELEASE_ROOT

import json
import math
from pathlib import Path

from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from audit_footprint import SHAPES, body_polygon
from build_lane_pcus import MapGeometry


ROOT = ARTIFACT_ROOT


def witness_keys(pcu):
    return (('witness_change','witness_keep') if pcu['family']=='lane_change_keep'
            else ('witness_low','witness_high'))


def local_drivable(geom, witnesses, anchor):
    paths = [LineString([anchor]+w['global_xy_m']) for w in witnesses]
    corridor = paths[0].union(paths[1]).buffer(8.0)
    nearby = geom.drivable_tree.query(corridor)
    if not nearby:
        return None
    return unary_union(nearby).buffer(0.2)


def evaluate(drivable, witness, anchor, anchor_yaw):
    points = [anchor]+witness['global_xy_m']
    headings = [anchor_yaw]
    headings.extend(math.atan2(b[1]-a[1],b[0]-a[0]) for a,b in zip(points,points[1:]))
    result = {}
    for name,shape in SHAPES.items():
        bodies = [body_polygon(point,heading,shape)
                  for point,heading in zip(points,headings)]
        flags = []
        for before,after in zip(bodies,bodies[1:]):
            swept = before.union(after).convex_hull
            flags.append(bool(drivable is not None and drivable.covers(swept)))
        result[name] = {'covered_segments':sum(flags),'total_segments':len(flags),
                        'all_covered':all(flags)}
    return result


def main():
    maps = {}
    output = ROOT/'audit/footprint_swept_diagnostics.jsonl'
    counts = {'all_both_origins':0,'any_origin_each_arm':0,'failed_both_origins':0}
    with (ROOT/'candidates/all_review_queue.jsonl').open() as source, output.open('w') as dest:
        for line in source:
            pcu = json.loads(line)
            location = pcu['public_input']['location']
            if location not in maps:
                maps[location] = MapGeometry(location)
            pose = pcu['public_input']['ego_pose']
            anchor = pose['xy_m']
            first,second = witness_keys(pcu)
            sidecar = pcu['private_sidecar']
            witnesses = [sidecar[first],sidecar[second]]
            drivable = local_drivable(maps[location],witnesses,anchor)
            record = {'pcu_id':pcu['pcu_id'],'status':'assumption_sensitive_diagnostic',
                      'method':'convex hull of adjacent 5 x 2 m rectangles, 0.2 m map tolerance',
                      'arm_0':evaluate(drivable,witnesses[0],anchor,pose['yaw_rad']),
                      'arm_1':evaluate(drivable,witnesses[1],anchor,pose['yaw_rad'])}
            dest.write(json.dumps(record)+'\n')
            both = all(record[arm][origin]['all_covered'] for arm in ('arm_0','arm_1') for origin in SHAPES)
            any_each = all(any(record[arm][origin]['all_covered'] for origin in SHAPES)
                           for arm in ('arm_0','arm_1'))
            counts['all_both_origins'] += both
            counts['any_origin_each_arm'] += any_each
            counts['failed_both_origins'] += not any_each
    print(json.dumps(counts,indent=2))


if __name__=='__main__':
    main()
