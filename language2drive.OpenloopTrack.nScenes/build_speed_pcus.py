#!/usr/bin/env python3
"""Build map-bound scoped-speed PCU review candidates from nuScenes test."""

from collections import Counter
import json
import math
from pathlib import Path
from paths import ARTIFACT_ROOT

from shapely.geometry import Point

from build_lane_pcus import CATALOG, MapGeometry, ROOT, TIMES, angle_diff, yaw


OUT = ARTIFACT_ROOT/'candidates/speed_review_queue.jsonl'
LANE = ARTIFACT_ROOT/'candidates/review_queue.jsonl'
LOW_SPEED = 4.0
HIGH_SPEED = 8.0
BANDS = {'low':[3.2,4.8], 'high':[7.2,8.8]}


def distance_at(t,initial,target):
    if t <= 1.0:
        return initial*t + 0.5*(target-initial)*t*t
    return 0.5*(initial+target)+target*(t-1.0)


def witness(path,anchor_xy,anchor_yaw,initial_speed,target_speed):
    point = Point(anchor_xy)
    start = path.project(point)
    center = path.interpolate(start)
    offset = (anchor_xy[0]-center.x,anchor_xy[1]-center.y)
    xy_global,xy_ego = [],[]
    for t in TIMES:
        s = start+distance_at(t,initial_speed,target_speed)
        if s > path.length-0.5:
            return None
        on_path = path.interpolate(s)
        fade = max(0,1-t)
        x,y = on_path.x+offset[0]*fade,on_path.y+offset[1]*fade
        xy_global.append([x,y])
        dx,dy = x-anchor_xy[0],y-anchor_xy[1]
        xy_ego.append([math.cos(anchor_yaw)*dx+math.sin(anchor_yaw)*dy,
                       -math.sin(anchor_yaw)*dx+math.cos(anchor_yaw)*dy])
    return {'timestamps_s':TIMES,'xy_m':xy_ego,'global_xy_m':xy_global,
            'coordinate_frame':'ego_at_anchor','reference_point':'ego_pose_origin'}


def contract(traj,band):
    if traj is None:
        return False
    points = traj['global_xy_m']
    positions = [[0,0]]+traj['xy_m']
    speeds = [math.dist(a,b)/0.25 for a,b in zip(positions,positions[1:])]
    lo,hi = band
    return (all(lo<=speed<=hi for speed,t in zip(speeds,TIMES) if t>=1.25)
            and math.dist(positions[0],positions[-1])>=8)


def quality(geom,traj,anchor_xy,initial_speed):
    if traj is None:
        return False
    points = traj['global_xy_m']
    if not all(geom.drivable_at(p) for p in points):
        return False
    positions = [anchor_xy]+points
    speeds = [math.dist(a,b)/0.25 for a,b in zip(positions,positions[1:])]
    headings = [yaw(a,b) for a,b in zip(positions,positions[1:])]
    return (abs(speeds[0]-initial_speed)<=3 and max(speeds)<=17 and
            max((abs(a-b)/0.25 for a,b in zip(speeds,speeds[1:])),default=0)<=5 and
            max((abs(angle_diff(a,b))/0.25 for a,b in zip(headings,headings[1:])),default=0)<=1.0)


def main():
    lane_scenes = {json.loads(line)['public_input']['scene_token'] for line in LANE.open()}
    rows = [json.loads(line) for line in CATALOG.open()]
    by_token = {row['sample_token']:row for row in rows}
    maps = {}
    accepted = set()
    counts = Counter()
    with OUT.open('w') as output:
        for row in rows:
            counts['anchors_seen'] += 1
            if row['scene_token'] in lane_scenes or row['scene_token'] in accepted:
                continue
            history = row['history_sample_tokens']
            if not history:
                continue
            old = by_token[history[-1]]
            dt = (row['timestamp_us']-old['timestamp_us'])/1e6
            if not 0.4<=dt<=0.6:
                continue
            anchor = row['ego_pose']['xy_m']
            speed = math.dist(anchor,old['ego_pose']['xy_m'])/dt
            if not 5.5<=speed<=6.5:
                continue
            location = row['location']
            if location not in maps:
                maps[location] = MapGeometry(location)
            geom = maps[location]
            current = geom.nearest_driving_lane(anchor,row['ego_pose']['yaw_rad'])
            if not current:
                continue
            path,tokens = geom.extended_path(current[2])
            low = witness(path,anchor,row['ego_pose']['yaw_rad'],speed,LOW_SPEED)
            high = witness(path,anchor,row['ego_pose']['yaw_rad'],speed,HIGH_SPEED)
            if not quality(geom,low,anchor,speed) or not quality(geom,high,anchor,speed):
                counts['quality_rejected'] += 1
                continue
            if not (contract(low,BANDS['low']) and contract(high,BANDS['high'])):
                counts['contract_rejected'] += 1
                continue
            if contract(low,BANDS['high']) or contract(high,BANDS['low']):
                raise AssertionError('Speed bands are not mutually exclusive')
            shared = {'sample_token':row['sample_token'],'scene_token':row['scene_token'],
                      'location':location,'public_input_hash':row['public_input_hash'],
                      'sensor_refs':row['sensor_refs'],
                      'history_sample_tokens':row['history_sample_tokens'],
                      'history_refs':row['history_refs'],'ego_pose':row['ego_pose']}
            record = {
                'pcu_id':f"nuscenes-test:{row['scene_name']}:{row['sample_token'][:12]}:speed-v1",
                'dataset_version':'v1.0-test','status':'needs_human_review','family':'scoped_speed',
                'public_input':shared,
                'arms':{
                    'arm_0':{'instruction':'Maintain about 15 km/h through this stretch.',
                             'contract':'speed_3.2_to_4.8_mps_t1.25_to_3'},
                    'arm_1':{'instruction':'Maintain about 30 km/h through this stretch.',
                             'contract':'speed_7.2_to_8.8_mps_t1.25_to_3'}},
                'private_sidecar':{'current_lane_tokens':tokens,'speed_mps':speed,
                                   'speed_bands_mps':BANDS,'scope_s':[1.25,3.0],
                                   'witness_low':low,'witness_high':high}}
            output.write(json.dumps(record,separators=(',',':'))+'\n')
            counts['candidate_pcus'] += 1
            accepted.add(row['scene_token'])
    (OUT.parent/'speed_summary.json').write_text(json.dumps({'counts':dict(counts),
        'dataset_version':'v1.0-test','status':'review_queue_not_frozen_benchmark'},indent=2))
    print(json.dumps(dict(counts),indent=2))


if __name__ == '__main__':
    main()
