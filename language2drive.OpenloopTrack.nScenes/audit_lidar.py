#!/usr/bin/env python3
"""Flag present-time LiDAR returns near candidate witness corridors.

This is an audit aid, not object detection or proof of future clearance.
"""

from paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, RELEASE_ROOT

import json
from pathlib import Path

import numpy as np


ROOT = ARTIFACT_ROOT
DATA = DATA_ROOT


def rotation_matrix(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],dtype=np.float32)


def corridor_count(points, trajectory, width=1.7):
    path = np.asarray(trajectory['xy_m'],dtype=np.float32)
    path = path[3:]  # From t=1.0 s onward; the first transition is near the ego body.
    closest = np.full(len(points),np.inf,dtype=np.float32)
    for a,b in zip(path[:-1],path[1:]):
        direction = b-a
        length2 = float(direction @ direction)
        if length2 < 1e-6:
            continue
        u = np.clip(((points[:,:2]-a) @ direction)/length2,0,1)
        projection = a + u[:,None]*direction
        closest = np.minimum(closest,np.sum((points[:,:2]-projection)**2,axis=1))
    return int(np.count_nonzero(closest < width*width))


def main():
    metadata = DATA/'v1.0-test'
    calibration = {x['token']:x for x in json.loads((metadata/'calibrated_sensor.json').read_text())}
    output = ROOT/'audit/lidar_diagnostics.jsonl'
    counts = []
    with (ROOT/'candidates/all_review_queue.jsonl').open() as source, output.open('w') as dest:
        for line in source:
            pcu = json.loads(line)
            ref = pcu['public_input']['sensor_refs']['LIDAR_TOP']
            cal = calibration[ref['calibrated_sensor_token']]
            raw = np.fromfile(DATA/ref['filename'],dtype=np.float32)
            if len(raw)%5:
                raise ValueError(f'Unexpected LiDAR file shape: {ref["filename"]}')
            xyz = raw.reshape(-1,5)[:,:3]
            ego = xyz @ rotation_matrix(cal['rotation']).T + np.asarray(cal['translation'],dtype=np.float32)
            mask = ((ego[:,2]>=0.6)&(ego[:,2]<=2.5)&
                    (ego[:,0]>=-5)&(ego[:,0]<=40)&(np.abs(ego[:,1])<=12))
            points = ego[mask]
            sidecar = pcu['private_sidecar']
            first,second = (('witness_change','witness_keep') if pcu['family']=='lane_change_keep'
                            else ('witness_low','witness_high'))
            count_first = corridor_count(points,sidecar[first])
            count_second = corridor_count(points,sidecar[second])
            record = {'pcu_id':pcu['pcu_id'],'lidar_points_in_arm_0_corridor':count_first,
                      'lidar_points_in_arm_1_corridor':count_second,
                      'threshold_description':'z in [0.6,2.5] m, lateral radius 1.7 m, t in [1,3] s',
                      'status':'diagnostic_only'}
            dest.write(json.dumps(record)+'\n')
            counts.append((count_first,count_second))
    print(f'Audited {len(counts)} PCUs. Arm-0 corridor count >20: {sum(x[0]>20 for x in counts)}')


if __name__ == '__main__':
    main()
