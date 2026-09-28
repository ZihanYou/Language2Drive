"""Build private visual review material. Never approves or freezes a candidate."""
import csv
import io
import json
import math
from pathlib import Path
import struct
from PIL import Image
from audit_local_data import fields

ROOT=Path(__file__).resolve().parents[1]
CAMERAS={1:'FRONT',2:'FRONT_LEFT',3:'FRONT_RIGHT',4:'SIDE_LEFT',5:'SIDE_RIGHT',6:'REAR_LEFT',7:'REAR',8:'REAR_RIGHT'}


def images(row):
    with open(row['file'],'rb') as f:
        f.seek(row['offset'])
        size=struct.unpack('<Q',f.read(8))[0]; f.read(4)
        for n,w,start,z in fields(f,f.tell()+size):
            if n==1 and w==2:
                for fn,fw,fs,fz in fields(f,start+z):
                    if fn==4 and fw==2:
                        camera=None; jpeg=None
                        for cn,cw,cs,cz in fields(f,fs+fz):
                            if cn==1 and cw==0: camera=cs
                            elif cn==2 and cw==2: jpeg=f.read(cz)
                        if camera is not None and jpeg is not None: yield camera,jpeg


def main():
    rows=[json.loads(s) for s in (ROOT/'artifacts/rated_frames.jsonl').read_text().splitlines()]
    pairs=[json.loads(s) for s in (ROOT/'artifacts/candidate_pairs.jsonl').read_text().splitlines()]
    out=ROOT/'review'; out.mkdir(exist_ok=True); (out/'assets').mkdir(exist_ok=True)
    cards=[]; errors=[]; decoded=0
    for i,row in enumerate(rows):
        name=row['frame_name']; assets=[]
        try:
            for cam,raw in images(row):
                pic=Image.open(io.BytesIO(raw)); pic.load(); original=pic.size
                pic.thumbnail((960,640))
                filename=f'{name}_{cam}.jpg'; pic.convert('RGB').save(out/'assets'/filename,quality=85)
                assets.append({'camera':CAMERAS.get(cam,str(cam)),'src':'assets/'+filename,'original_size':original})
                decoded+=1
        except Exception as e:
            errors.append({'frame_name':name,'error':str(e)})
        cards.append({'frame_name':name,'sequence_id':row['sequence_id'],'cluster':row['scenario_cluster'],
                      'intent':row.get('intent'),'past':row.get('past',{}),'preferences':row['preferences'],
                      'images':assets,'time_axis_status':'official_rfs_prefix20_v1'})
        if (i+1)%25==0: print(f'{i+1}/{len(rows)} anchors, {decoded} images',flush=True)
    pairs.sort(key=lambda p:(min(p['scores'])>=6, max(p['dy3']/0.75,p['dy5']/0.75,p['dx3']/3,p['dx5']/3)),reverse=True)
    columns=['candidate_id','frame_name','sequence_id','scenario_cluster','reference_a','reference_b','score_a','score_b',
             'point_count_a','point_count_b','dx_endpoint_m','dy_endpoint_m','dx3_m','dy3_m','dx5_m','dy5_m','time_axis_status','status',
             'reviewer','review_round','family','observable','both_feasible','passenger_selectable','mutually_exclusive',
             'horizon_decidable','route_policy','geometry_evidence','strong_interaction','decision','reason']
    with (out/'review_queue.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns); w.writeheader()
        for i,p in enumerate(pairs):
            w.writerow({'candidate_id':f'CAND-{i+1:04d}','frame_name':p['frame_name'],'sequence_id':p['sequence_id'],
                'scenario_cluster':p['scenario_cluster'],'reference_a':p['indices'][0],'reference_b':p['indices'][1],
                'score_a':p['scores'][0],'score_b':p['scores'][1],'point_count_a':p['point_counts'][0],
                'point_count_b':p['point_counts'][1],'dx_endpoint_m':p['dx_endpoint'],'dy_endpoint_m':p['dy_endpoint'],
                'dx3_m':p['dx3'],'dy3_m':p['dy3'],'dx5_m':p['dx5'],'dy5_m':p['dy5'],
                'time_axis_status':p['time_axis_status'],'status':'DISCOVERED'})
    template=(ROOT/'scripts/review_template.html').read_text()
    payload=json.dumps(cards,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    (out/'index.html').write_text(template.replace('__DATA__',payload))
    (out/'build_report.json').write_text(json.dumps({'anchors':len(rows),'decoded_images':decoded,'errors':errors,
                                                   'candidate_pairs':len(pairs),'approved_units':0},indent=2)+'\n')
    print('Review materials ready; no candidates approved.',flush=True)

if __name__=='__main__': main()
