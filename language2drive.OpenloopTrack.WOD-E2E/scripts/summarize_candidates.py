"""Normalize rated arrays using official RFS prefix convention; summarize candidates.
These geometry/score hints are NOT semantic labels or safety certification.
"""
import collections
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
POLICY='official_rfs_prefix20_v1'


def features(t):
    pts=list(zip(t['x'][:20],t['y'][:20])); previous=(0.,0.)
    speeds=[]
    for point in pts:
        speeds.append(math.dist(point,previous)*4); previous=point
    accelerations=[(b-a)*4 for a,b in zip(speeds,speeds[1:])]
    return {'x3_m':pts[11][0],'y3_m':pts[11][1],'x5_m':pts[19][0],'y5_m':pts[19][1],
            'path_length_m':sum(speeds)/4,'mean_speed_mps':sum(speeds)/20,
            'mean_last_second_speed_mps':sum(speeds[-4:])/4,'first_interval_speed_mps':speeds[0],
            'max_abs_interval_acceleration_mps2':max(map(abs,accelerations)),
            'speeds_mps':speeds,'status':'motion_features_only'}


def main():
    a=ROOT/'artifacts'; summary=json.loads((a/'local_inventory.json').read_text())
    rows=[json.loads(s) for s in (a/'rated_frames.jsonl').read_text().splitlines()]
    pairs=[json.loads(s) for s in (a/'raw_pairs.jsonl').read_text().splitlines()]
    by_frame={r['frame_name']:r for r in rows}; normalized=[]; feature_rows=[]
    for r in rows:
        nr={'frame_name':r['frame_name'],'sequence_id':r['sequence_id'],'normalization_policy':POLICY,
            'coordinate_frame':'ego_at_anchor','reference_point':'rear_axle',
            'timestamps_s':[(i+1)/4 for i in range(20)],'preferences':[],'approved':False}
        initial=math.hypot(r['past']['vx'][-1],r['past']['vy'][-1])
        for i in r['valid_preference_indices']:
            t=r['preferences'][i]
            if len(t['x'])<20: continue
            ft=features(t)
            ft.update(frame_name=r['frame_name'],reference_index=i,preference_score=t['score'],
                      raw_point_count=len(t['x']),initial_speed_mps=initial,
                      first_interval_speed_minus_initial_mps=ft['first_interval_speed_mps']-initial)
            feature_rows.append(ft)
            nr['preferences'].append({'reference_index':i,'score':t['score'],'raw_point_count':len(t['x']),
                                      'xy_m':list(map(list,zip(t['x'][:20],t['y'][:20])))})
        normalized.append(nr)
    candidates=[]
    for p in pairs:
        r=by_frame[p['frame_name']]; ta,tb=[r['preferences'][i] for i in p['indices']]
        if min(len(ta['x']),len(tb['x']))<20: continue
        p=dict(p)
        p.update(normalization_policy=POLICY,time_axis_status='official_rfs_index_convention',
                 dx3=abs(ta['x'][11]-tb['x'][11]),dy3=abs(ta['y'][11]-tb['y'][11]),
                 dx5=abs(ta['x'][19]-tb['x'][19]),dy5=abs(ta['y'][19]-tb['y'][19]))
        p['geometric_retrieval_hint']=max(p['dx3'],p['dx5'])>=3 or max(p['dy3'],p['dy5'])>=0.75
        p['both_scores_ge6_hint']=min(p['scores'])>=6
        p['approved']=False
        candidates.append(p)
    for name,values in [('normalized_rated_frames.jsonl',normalized),('trajectory_features.jsonl',feature_rows),('candidate_pairs.jsonl',candidates)]:
        with (a/name).open('w') as f:
            for r in values:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    strong=[p for p in candidates if p['geometric_retrieval_hint'] and p['both_scores_ge6_hint']]
    stats={'normalization_policy':POLICY,'normalized_pairs':len(candidates),
           'excluded_short_reference_pairs':len(pairs)-len(candidates),
           'full_length_pair_sequences':len({p['sequence_id'] for p in candidates}),
           'geometry_hint_pairs':sum(p['geometric_retrieval_hint'] for p in candidates),
           'both_ge6_and_geometry_hint_pairs':len(strong),
           'both_ge6_and_geometry_hint_sequences':len({p['sequence_id'] for p in strong}),
           'approved_core_pcus':0,'approved_scus':0}
    (a/'candidate_summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2))

if __name__=='__main__': main()
