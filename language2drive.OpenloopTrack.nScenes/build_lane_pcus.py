#!/usr/bin/env python3
"""Discover map-bound lane-change/keep PCUs in the nuScenes test catalog.

Only a PCU with two verifier-accepted, dynamically plausible witnesses is saved.
The result is a review queue, not a frozen benchmark release.
"""

from paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, RELEASE_ROOT

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree


ROOT = PROJECT_ROOT
DATA = DATA_ROOT
CATALOG = ARTIFACT_ROOT / 'scene_catalog/anchors.jsonl'
OUT = ARTIFACT_ROOT / 'candidates'
TIMES = [round(i * 0.25, 2) for i in range(1, 13)]


def angle_diff(a, b):
    return math.atan2(math.sin(a-b), math.cos(a-b))


def yaw(a, b):
    return math.atan2(b[1]-a[1], b[0]-a[0])


def make_polygon(record, nodes):
    exterior = [nodes[t] for t in record['exterior_node_tokens']]
    holes = [[nodes[t] for t in h['node_tokens']] for h in record.get('holes', [])]
    p = Polygon(exterior, holes)
    return p if p.is_valid else p.buffer(0)


class MapGeometry:
    def __init__(self, location):
        self.location = location
        self.data = json.loads((DATA / 'maps/expansion' / f'{location}.json').read_text())
        d = self.data
        nodes = {n['token']: (n['x'], n['y']) for n in d['node']}
        polygons = {p['token']: p for p in d['polygon']}
        self.lanes = {x['token']: x for x in d['lane'] if x.get('lane_type') == 'CAR'}
        self.lane_polygons = {t: make_polygon(polygons[x['polygon_token']], nodes)
                              for t, x in self.lanes.items()}
        self.lane_tolerances = {t: geom.buffer(0.15) for t, geom in self.lane_polygons.items()}
        self.geometries = list(self.lane_polygons.values())
        self.geometry_tokens = {id(g): t for t, g in self.lane_polygons.items()}
        self.tree = STRtree(self.geometries)
        self.paths = {}
        for token, parts in d['arcline_path_3'].items():
            coords = [tuple(parts[0]['start_pose'][:2])]
            coords.extend(tuple(part['end_pose'][:2]) for part in parts)
            if len(coords) > 1 and len(set(coords)) > 1:
                self.paths[token] = LineString(coords)
        self.connectivity = d['connectivity']
        # Spatially indexed source polygons are used for pointwise drivable checks.
        self.drivable = [make_polygon(polygons[t], nodes)
                         for rec in d['drivable_area'] for t in rec['polygon_tokens']]
        self.drivable_tree = STRtree(self.drivable)

    def nearest_driving_lane(self, xy, heading):
        pt = Point(xy)
        eligible = []
        for geom in self.tree.query(pt.buffer(2.0)):
            token = self.geometry_tokens[id(geom)]
            line = self.paths.get(token)
            if line is None:
                continue
            s = line.project(pt)
            before = line.interpolate(max(0, s-0.5))
            after = line.interpolate(min(line.length, s+0.5))
            direction = yaw(before.coords[0], after.coords[0])
            if abs(angle_diff(heading, direction)) > math.radians(30):
                continue
            distance = geom.distance(pt)
            if distance < 1.0:
                eligible.append((distance, line.distance(pt), token, direction))
        return min(eligible) if eligible else None

    def adjacent_lane(self, current_token, xy, heading, side):
        pt = Point(xy)
        current = self.lanes[current_token]
        divider = current[f'{side}_lane_divider_segments']
        if not divider or not all('DASHED' in seg['segment_type'] for seg in divider):
            return None
        candidates = []
        for geom in self.tree.query(pt.buffer(7.0)):
            token = self.geometry_tokens[id(geom)]
            if token == current_token or geom.distance(self.lane_polygons[current_token]) > 0.8:
                continue
            line = self.paths.get(token)
            if line is None:
                continue
            s = line.project(pt)
            before = line.interpolate(max(0, s-0.5))
            after = line.interpolate(min(line.length, s+0.5))
            direction = yaw(before.coords[0], after.coords[0])
            if abs(angle_diff(heading, direction)) > math.radians(20):
                continue
            nearest = line.interpolate(s)
            dx, dy = nearest.x-xy[0], nearest.y-xy[1]
            lateral = -math.sin(heading)*dx + math.cos(heading)*dy
            forward = math.cos(heading)*dx + math.sin(heading)*dy
            signed_lateral = lateral if side == 'left' else -lateral
            if 2.4 <= signed_lateral <= 5.0 and abs(forward) <= 5.0:
                candidates.append((abs(signed_lateral-3.5)+abs(forward)*0.1, token))
        return min(candidates)[1] if candidates else None

    def extended_path(self, start_token, min_length=80):
        coords = list(self.paths[start_token].coords)
        tokens = [start_token]
        visited = {start_token}
        while LineString(coords).length < min_length:
            current = tokens[-1]
            last_heading = yaw(coords[-2], coords[-1])
            choices = []
            for token in self.connectivity.get(current, {}).get('outgoing', []):
                if token in visited or token not in self.paths:
                    continue
                candidate = self.paths[token]
                points = list(candidate.coords)
                gap = math.dist(coords[-1], points[0])
                if gap > 3.0:
                    continue
                turn = abs(angle_diff(last_heading, yaw(points[0], points[1])))
                if turn < math.radians(35):
                    choices.append((turn + gap*0.02, token, points))
            if not choices:
                break
            _, token, points = min(choices)
            coords.extend(points[1:] if math.dist(coords[-1], points[0]) < 0.1 else points)
            tokens.append(token)
            visited.add(token)
        return LineString(coords), tokens

    def drivable_at(self, xy):
        pt = Point(xy)
        return any(g.covers(pt) or g.distance(pt) <= 0.2
                   for g in self.drivable_tree.query(pt.buffer(0.2)))


def trajectory(path, anchor_xy, anchor_yaw, speed, blend_with=None):
    origin = Point(anchor_xy)
    start = path.project(origin)
    initial_center = path.interpolate(start)
    initial_offset = (anchor_xy[0]-initial_center.x, anchor_xy[1]-initial_center.y)
    target_start = blend_with.project(origin) if blend_with is not None else None
    xy_global, xy_ego = [], []
    for t in TIMES:
        s = start + speed*t
        if s > path.length-0.5:
            return None
        base = path.interpolate(s)
        x, y = base.x, base.y
        # Start exactly at the recorded ego pose; fade any map-centerline offset.
        offset_weight = max(0.0, 1.0-t/1.0)
        x += initial_offset[0]*offset_weight
        y += initial_offset[1]*offset_weight
        if blend_with is not None:
            target_s = target_start + speed*t
            if target_s > blend_with.length-0.5:
                return None
            target = blend_with.interpolate(target_s)
            u = max(0.0, min(1.0, (t-0.4)/2.0))
            weight = u*u*(3-2*u)
            x += (target.x-x)*weight
            y += (target.y-y)*weight
        xy_global.append([x, y])
        dx, dy = x-anchor_xy[0], y-anchor_xy[1]
        xy_ego.append([math.cos(anchor_yaw)*dx+math.sin(anchor_yaw)*dy,
                       -math.sin(anchor_yaw)*dx+math.cos(anchor_yaw)*dy])
    return {'timestamps_s': TIMES, 'xy_m': xy_ego, 'global_xy_m': xy_global,
            'coordinate_frame': 'ego_at_anchor', 'reference_point': 'ego_pose_origin'}


def occupies(geom, point_xy, tokens):
    point = Point(point_xy)
    return any(geom.lane_tolerances[t].covers(point)
               for t in tokens if t in geom.lane_tolerances)


def contract_pass(geom, witness, tokens, hold_from):
    if witness is None:
        return False
    return all(occupies(geom, p, tokens) for p,t in zip(witness['global_xy_m'], TIMES)
               if t >= hold_from)


def verify_witness(geom, witness, anchor_xy, speed, current_tokens, target_tokens, change):
    if witness is None:
        return False, 'short_path'
    points = witness['global_xy_m']
    if not all(geom.drivable_at(p) for p in points):
        return False, 'outside_drivable'
    displacements = [math.dist(a, b)/0.25 for a, b in zip([anchor_xy]+points, points)]
    headings = [yaw(a,b) for a,b in zip([anchor_xy]+points,points)]
    if (not displacements or max(displacements) > 17 or
            abs(displacements[0]-speed) > 3 or
            max(abs(a-b) for a,b in zip(displacements,displacements[1:]))/0.25 > 5 or
            max((abs(angle_diff(a,b))/0.25 for a,b in zip(headings,headings[1:])),default=0) > 1.0):
        return False, 'motion_limit'
    if math.dist(points[0], points[-1]) < 8:
        return False, 'insufficient_progress'
    if change:
        if not contract_pass(geom, witness, target_tokens, 2.25):
            return False, 'target_not_held'
        if contract_pass(geom, witness, current_tokens, 0):
            return False, 'opposite_contract_true'
    else:
        if not contract_pass(geom, witness, current_tokens, 0):
            return False, 'current_not_held'
        if contract_pass(geom, witness, target_tokens, 2.25):
            return False, 'opposite_contract_true'
    return True, 'ok'


def build(catalog, output, limit):
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in catalog.open()]
    by_token = {r['sample_token']: r for r in rows}
    maps = {}
    counts = Counter()
    accepted_scene = set()
    with (output / 'review_queue.jsonl').open('w') as out:
        for row in rows:
            counts['anchors_seen'] += 1
            if row['scene_token'] in accepted_scene:
                continue
            previous = row['history_sample_tokens'][-1:]
            if not previous:
                counts['missing_history'] += 1
                continue
            prev = by_token[previous[0]]
            gap = (row['timestamp_us'] - prev['timestamp_us']) / 1e6
            if not 0.4 <= gap <= 0.6:
                continue
            xy = row['ego_pose']['xy_m']
            old_xy = prev['ego_pose']['xy_m']
            speed = math.dist(xy, old_xy) / gap
            if not 4.5 <= speed <= 9.0:
                counts['speed_rejected'] += 1
                continue
            location = row['location']
            if location not in maps:
                maps[location] = MapGeometry(location)
            geom = maps[location]
            heading = row['ego_pose']['yaw_rad']
            current = geom.nearest_driving_lane(xy, heading)
            if not current:
                counts['no_current_lane'] += 1
                continue
            current_token = current[2]
            current_path, current_tokens = geom.extended_path(current_token)
            for side in ('left','right'):
                target_token = geom.adjacent_lane(current_token, xy, heading, side)
                if not target_token:
                    counts[f'no_{side}_lane'] += 1
                    continue
                target_path, target_tokens = geom.extended_path(target_token)
                keep = trajectory(current_path, xy, heading, speed)
                change = trajectory(current_path, xy, heading, speed, target_path)
                keep_ok, keep_reason = verify_witness(geom, keep, xy, speed, current_tokens, target_tokens, False)
                change_ok, change_reason = verify_witness(geom, change, xy, speed, current_tokens, target_tokens, True)
                if not keep_ok or not change_ok:
                    counts[f'witness_rejected:{keep_reason if not keep_ok else change_reason}'] += 1
                    continue
                pcu_id = f"nuscenes-test:{row['scene_name']}:{row['sample_token'][:12]}:{side}-change-v1"
                shared = {'sample_token': row['sample_token'], 'scene_token': row['scene_token'],
                          'location': location, 'public_input_hash': row['public_input_hash'],
                          'sensor_refs': row['sensor_refs'],
                          'history_sample_tokens': row['history_sample_tokens'],
                          'history_refs': row['history_refs'],
                          'ego_pose': row['ego_pose']}
                private = {'current_lane_tokens': current_tokens, 'target_lane_tokens': target_tokens,
                           'current_lane_token': current_token, 'target_lane_token': target_token,
                           'target_side': side, 'speed_mps': speed,
                           'witness_keep': keep, 'witness_change': change,
                           'witness_verifier': {'keep': keep_reason, 'change': change_reason}}
                record = {'pcu_id': pcu_id, 'dataset_version': 'v1.0-test',
                          'status': 'needs_human_review', 'family': 'lane_change_keep',
                          'public_input': shared,
                          'arms': {'arm_0': {'instruction': f'Change to the lane on the {side} and stay there.',
                                            'contract': 'target_lane_held_0.75s'},
                                   'arm_1': {'instruction': 'Stay in the current lane.',
                                            'contract': 'current_lane_held_3s'}},
                          'private_sidecar': private}
                out.write(json.dumps(record, separators=(',', ':'))+'\n')
                counts['candidate_pcus'] += 1
                accepted_scene.add(row['scene_token'])
                break
            if limit and counts['candidate_pcus'] >= limit:
                break
    summary = {'source': str(catalog), 'dataset_version': 'v1.0-test',
               'counts': dict(counts), 'maps_loaded': sorted(maps),
               'status': 'review_queue_not_frozen_benchmark'}
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=CATALOG)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--limit', type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(build(args.catalog, args.output, args.limit), indent=2))


if __name__ == '__main__':
    main()
