#!/usr/bin/env python3
"""Export display-only geometry for four CARLA PCU reference-path explorers."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

CASES={'A':'A-001','B':'B-009','C':'C-025','D':'D-001'}

def simplify(ps,tolerance=.12):
    if len(ps)<3:return ps
    a,b=ps[0],ps[-1];dx,dy=b[0]-a[0],b[1]-a[1];length=dx*dx+dy*dy
    def distance(p):
        t=max(0,min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length)) if length else 0
        return math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)
    ds=[distance(p) for p in ps[1:-1]];maximum=max(ds)
    if maximum<=tolerance:return [a,b]
    index=ds.index(maximum)+1
    return simplify(ps[:index+1],tolerance)[:-1]+simplify(ps[index:],tolerance)

def build(archive):
    cases=[]
    with zipfile.ZipFile(archive) as z:
        for scope,key in CASES.items():
            member=f'pcu_video_evidence_20261002/cases/formal_{key}/data.json';raw=z.read(member);data=json.loads(raw)
            origin=data['arms'][0]['oracle_path'][0];angle=math.radians(origin['yaw_deg']);co,si=math.cos(angle),math.sin(angle)
            def xy(p):
                if isinstance(p,dict):p=[p['x'],p['y']]
                dx,dy=p[0]-origin['x'],p[1]-origin['y']
                return [round(dx*co+dy*si,3),round(-dx*si+dy*co,3)]
            paths=[simplify([xy(p) for p in a['oracle_path']],.025) for a in data['arms']]
            points=[p for ps in paths for p in ps];pad=6
            bounds=[min(p[0] for p in points)-pad,min(p[1] for p in points)-pad,max(p[0] for p in points)+pad,max(p[1] for p in points)+pad]
            lanes=[]
            for lane in data['lanes']:
                ps=simplify([xy(p) for p in lane['xy']])
                if any(bounds[0]-10<=p[0]<=bounds[2]+10 and bounds[1]-10<=p[1]<=bounds[3]+10 for p in ps):lanes.append({'points':ps,'width':lane['lane_width_m']})
            goals=[]
            for a,arm in enumerate(data['arms']):
                target=arm['target'];goal={'point':paths[a][-1],'label':target.get('slot_number',target.get('semantic_name','Goal '+ 'AB'[a]))}
                if 'polygon' in target:goal['polygon']=[xy(p) for p in target['polygon']]
                goals.append(goal)
            if scope=='D':
                # Each arm's last stage is the other arm's first named destination.
                for a,arm in enumerate(data['arms']):goals[a]['label']=arm['target']['stages'][-1].split(':')[-1]
            cases.append({'scope':scope,'pcu':key,'case_id':'carla-formal-'+key.lower(),'scene':data['town'],'instructions':[a['instruction'] for a in data['arms']],'paths':paths,'lanes':lanes,'goals':goals,'bounds':bounds,'evidence_role':'benchmark-target','source_member':member,'source_sha256':hashlib.sha256(raw).hexdigest()})
    return {'schema':'language2drive.scope-geometry.v1','coordinates':'Rigid transform of native CARLA XY; common initial pose at the origin, initial forward direction to the right. Distances in meters.','animation':'Normalized progression along reference paths.','cases':cases}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',type=Path,required=True);args=parser.parse_args()
    dest=Path(__file__).resolve().parents[1]/'docs/data/scope-cases.json';dest.write_text(json.dumps(build(args.archive),separators=(',',':'))+'\n');print(dest)
