"""Language-blind public-input baseline for protocol checks only."""

import math


def predict(public_input, instruction, seed):
    current = public_input['ego_pose']
    history = public_input['history_refs']
    if history:
        previous = history[-1]
        dt = (public_input['timestamp_us']-previous['timestamp_us'])/1e6
        distance = math.dist(current['xy_m'],previous['ego_pose']['xy_m'])
        speed = max(0.0,min(15.0,distance/dt)) if dt>0 else 0.0
    else:
        speed = 0.0
    times = [round(i*0.25,2) for i in range(1,13)]
    return {'timestamps_s':times,'xy_m':[[speed*t,0.0] for t in times],
            'coordinate_frame':'ego_at_anchor','reference_point':'ego_pose_origin'}
