#!/usr/bin/env python3
"""Export CARLA target pairs and the six-map atlas from the author archive.

Only viewing media, measured lane geometry and display metadata are exported.
Frame ZIPs, actor logs and private task contracts remain in the input archive.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import re
from pathlib import Path
import shutil
import subprocess
import zipfile
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs'
MAPS=[('American-city0428','American City','Urban grid','Named streets, branching intersections and scene-grounded route choice.'),
      ('yangluogang1','Yangluo Port','Port logistics','Loading zones, storage areas and destination-conditioned navigation.'),
      ('parkinglotfish-new3','Fishbone Parking','Night parking','Named bays and parking zones for precise destinations and instruction programs.'),
      ('hanghailou3','Hanghai District','Urban neighborhood','Compact roads and junctions for persistent driving behavior.'),
      ('hubeiwenli4','Hubei Campus','Campus roads','A connected campus road network for local maneuvers and lane choice.'),
      ('road1','Road One','Straight-road control','A straight corridor for exploring speed and lane behavior.')]
TITLES={'junction':'Intersection choice','lane':'Change lane or stay','speed_start':'Adjust the requested speed','waiting_parking':'Left or right parking space','speed_constraint':'Maintain the requested speed','lane_constraint':'Keep the requested lane','object_relation':'Choose the following distance','passing_order':'Pass or follow','road_name':'Choose a named street','parking_region':'Choose a parking zone','exact_parking':'Choose a named parking bay','port_target':'Choose a port destination','strict_order':'Same destinations, different order','conditional':'Act on a scene condition','memory_reference':'Remember the earlier reference','until_invariant':'Maintain a constraint until the marker'}

def digest(data):return hashlib.sha256(data).hexdigest()
def jread(z,p):return json.loads(z.read(p))
def csvread(z,p):return list(csv.DictReader(io.StringIO(z.read(p).decode('utf-8-sig'))))
def prefix(z):return z.namelist()[0].split('/')[0]+'/'

def collect_targets(archive):
    cases=[]
    with zipfile.ZipFile(archive) as z:
        r=prefix(z)
        for pair in csvread(z,r+'pair_index.csv'):
            for view,field in [('exterior','video'),('front','front_video')]:
                member,partner=r+pair[field+'_0'],r+pair[field+'_1']
                cases.append(dict(id='carla-target-'+pair['pcu_id'].lower()+'-'+view,pcu=pair['pcu_id'],track='carla',kind='target',
                    title=TITLES[pair['category']],family=pair['category'],scope=pair['layer'],scene=pair['town'],model='Benchmark target',
                    instructions=[pair['instruction_0'],pair['instruction_1']],archive=archive.name,member=member,partner_member=partner,
                    camera_view=view,playback_speed=float(pair['playback_speed']),target_kind='CARLA reference-path animation',
                    trajectory_source=[n.rsplit('/',1)[0]+'/target_trajectory_30hz.csv' for n in (member,partner)],
                    trajectory_panel=('Vehicle front view' if view=='front' else 'Exterior follow camera')+' · reference-path animation',
                    note='Benchmark target: CARLA renders the requested reference motion for each instruction. '+('Vehicle front view.' if view=='front' else 'Exterior follow camera.')+' Both arms share the initial scene and playback scale; blue/orange identify A/B. Full reference paths use identical bounds and equal XY scale.'))
    return cases

def render_target_pair(case,archive,src,video,work,refresh):
    from build_showcase import sha
    from render_case import layout
    # The original clips are independent arms; join them without changing timing.
    other=work/'source'/(case['id']+'-partner.mp4')
    identity=other.with_suffix('.source.json');expected={'archive':case['archive'],'member':case['partner_member']}
    if not other.exists() or not identity.exists() or json.loads(identity.read_text())!=expected:
        with zipfile.ZipFile(archive) as z,z.open(case['partner_member']) as stream,other.open('wb') as f:shutil.copyfileobj(stream,f)
        identity.write_text(json.dumps(expected)+'\n');refresh=True
    case['sources']=[{'member':m,'sha256':sha(p)} for m,p in [(case['member'],src),(case['partner_member'],other)]]
    if video.exists() and not refresh:return
    bg=work/(case['id']+'-layout.png');layout(case,archive,bg)
    graph='[1:v]scale=824:328:force_original_aspect_ratio=decrease,pad=824:328:(ow-iw)/2:(oh-ih)/2:color=0x173240,setsar=1[a];[2:v]scale=824:328:force_original_aspect_ratio=decrease,pad=824:328:(ow-iw)/2:(oh-ih)/2:color=0x173240,setsar=1[b];[0:v][a]overlay=88:381:shortest=1[v];[v][b]overlay=1008:381:shortest=1,format=yuv420p[out]'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-loop','1','-framerate','12','-i',str(bg),'-i',str(src),'-i',str(other),'-filter_complex_threads','1','-filter_complex',graph,'-map','[out]','-an','-r','12','-c:v','libx264','-preset','fast','-crf','23','-threads','2','-movflags','+faststart',str(video)],check=True)

def media_job(archive,member,dest):
    """Cache by source member and SHA; sources are removed after web encoding."""
    cache=Path('/tmp/l2d-simulator/cache');cache.mkdir(parents=True,exist_ok=True)
    side=cache/(digest(dest.as_posix().encode())+'.json')
    with zipfile.ZipFile(archive) as z:data=z.read(member)
    source_sha=digest(data)
    recipe='webp-1080-q87' if dest.suffix=='.webp' else 'h264-1080-crf25-cap3m-v2'
    if not dest.exists() or not side.exists() or jread_file(side).get('source_sha256')!=source_sha or jread_file(side).get('recipe')!=recipe:
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.suffix=='.webp':
            im=Image.open(io.BytesIO(data));im.thumbnail((1920,1080));im.save(dest,'WEBP',quality=87)
        else:
            src=Path('/tmp/l2d-simulator')/(dest.stem+'-'+dest.parent.name+'-source.mp4');src.parent.mkdir(exist_ok=True);src.write_bytes(data)
            tmp=dest.with_name(dest.stem+'-encode.mp4')
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(src),'-an','-c:v','libx264','-preset','fast','-crf','25','-maxrate','3000k','-bufsize','6000k','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],check=True)
            tmp.replace(dest)
            src.unlink()
        side.write_text(json.dumps({'member':member,'source_sha256':source_sha,'recipe':recipe})+'\n')
    from build_showcase import sha
    info={'path':dest.relative_to(DOCS).as_posix(),'source_member':member,'source_sha256':source_sha,'sha256':sha(dest),'bytes':dest.stat().st_size}
    if dest.suffix=='.mp4':
        poster=dest.with_suffix('.webp')
        if not poster.exists():subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss','8','-i',str(dest),'-frames:v','1','-vf','scale=960:540','-threads','1','-c:v','libwebp','-quality','85',str(poster)],check=True)
        info['poster']=poster.relative_to(DOCS).as_posix()
    return info

def jread_file(p):return json.loads(p.read_text())

def build_maps(archive,workers):
    maps=[];jobs=[]
    with zipfile.ZipFile(archive) as z:
        r=prefix(z)
        for town,title,tag,description in MAPS:
            base=r+'map_exploration/'+town+'/'
            graph_member=r+f'reference/maps/{town}/{town}.lane_graph.json'
            graph=jread(z,graph_member);key=jread(z,base+'key_locations.json');road=jread(z,base+'road_level/key_locations.json')
            lanes=[]
            for lane in graph['lanes']:
                samples=lane['samples']
                if len(samples)<2:continue
                # Preserve curved geometry while reducing browser payload; endpoints always retained.
                stride=max(1,len(samples)//150)
                ps=samples[::stride]
                if ps[-1]!=samples[-1]:ps.append(samples[-1])
                lanes.append({'points':[[round(p['x'],3),round(p['y'],3)] for p in ps], 'width_m':round(samples[0]['lane_width_m'],2),'junction':bool(lane['is_junction'])})
            done=jread(z,base+'exploration_done.json')
            item={'id':town,'title':title,'tag':tag,'description':description,'lighting':done['lighting']['mode'],'seed':done['seed'],
                'geometry_member':graph_member,'geometry_sha256':digest(z.read(graph_member)),'lanes':lanes,'locations':[],'tours':[]}
            for i,loc in enumerate(key):
                point=loc['point'];raw=point['name']
                label=('Landmark '+raw if raw.isdigit() else re.sub(r'(?<=[a-z])(?=[A-Z])',' ',raw.replace('_',' ')))
                if raw.startswith('junction_'):label='Junction '+raw.split('_')[1][1:]
                if raw.startswith('road_'):label='Road '+raw.split('_')[1][1:]+' · lane '+raw.split('_')[2][1:]
                location={'id':f'location-{i+1:02d}','label':label,'source_name':raw,'x':point['x'],'y':point['y'],'z':point['z'],'views':{}}
                for view,entry,folder in [('overview',loc,base),*[('reverse' if 'reverse' in a['image'] else 'forward',a,base+'road_level/') for a in road if a['point']['name']==raw]]:
                    dest=DOCS/'assets/maps'/town/f'{location["id"]}-{view}.webp'
                    jobs.append((location['views'],view,folder+entry['image'],dest))
                item['locations'].append(location)
            for vantage,folder in [('overview',base),('road',base+'road_level/')]:
                receipt=done if vantage=='overview' else jread(z,folder+'road_level_done.json')
                for n in (1,2):
                    clip=folder+f'tour_{n:02d}/';rows=csvread(z,clip+'camera_trajectory.csv')
                    samples=rows[::3]
                    if samples[-1]!=rows[-1]:samples.append(rows[-1])
                    route=[[round(float(a['video_time_s']),3),round(float(a['camera_x']),3),round(float(a['camera_y']),3),round(float(a['camera_yaw_deg']),2)] for a in samples]
                    tour={'id':f'{vantage}-{n}','label':('Map tour' if vantage=='overview' else 'Road-level drive')+f' {n}', 'view':vantage,'route':route,'route_member':clip+'camera_trajectory.csv','route_sha256':digest(z.read(clip+'camera_trajectory.csv')),'duration_s':len(rows)/15,'vehicles':receipt['spawned_vehicles']}
                    tour['cuts_s']=[s['video_frame']/15 for s in receipt['clips'][n-1].get('viewpoint_switches',[]) if s['video_frame']>0]
                    jobs.append((tour,'media',clip+'video.mp4',DOCS/'assets/maps'/town/f'{vantage}-{n}.mp4'))
                    item['tours'].append(tour)
            maps.append(item)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[(owner,key,pool.submit(media_job,archive,member,dest)) for owner,key,member,dest in jobs]
        for owner,key,f in futures:owner[key]=f.result()
    data={'schema':'language2drive.simulator.v1','archive':archive.name,'coordinates':'CARLA world XY in meters; +X right, +Y down in this atlas. Lane geometry and camera routes use the exported CARLA coordinate frame.','maps':maps}
    (DOCS/'data/maps.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
    # Build lightweight vector thumbnails from exactly the same measured lane geometry.
    for m in maps:
        ps=[p for l in m['lanes'] for p in l['points']];xs=[p[0] for p in ps];ys=[p[1] for p in ps]
        xmin,xmax,ymin,ymax=min(xs),max(xs),min(ys),max(ys);scale=min(248/max(xmax-xmin,1),120/max(ymax-ymin,1))
        xy=lambda p:f'{136+(p[0]-(xmin+xmax)/2)*scale:.2f},{72+(p[1]-(ymin+ymax)/2)*scale:.2f}'
        lines=''.join(f'<polyline points="{" ".join(xy(p) for p in l["points"])}" fill="none" stroke="#588f83" stroke-width="{max(.75,l["width_m"]*scale*.7):.2f}" stroke-linecap="round"/>' for l in m['lanes'])
        thumb=DOCS/'assets/maps'/m['id']/'map.svg';thumb.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 272 144"><rect width="272" height="144" fill="#edf2e9"/>{lines}</svg>\n')
        m['thumbnail']=thumb.relative_to(DOCS).as_posix()
    (DOCS/'data/maps.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
    print('MAPS READY',len(maps),'locations',sum(len(m['locations']) for m in maps),'tours',sum(len(m['tours']) for m in maps),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--work-dir',type=Path,default=Path('/tmp/l2d-showcase'));p.add_argument('--workers',type=int,default=4);p.add_argument('--maps-only',action='store_true');p.add_argument('--targets-only',action='store_true');args=p.parse_args()
    if not args.maps_only:
        from build_showcase import build_media
        old=jread_file(DOCS/'data/cases.json');prior={c['id']:c for c in old['cases']}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            new=list(pool.map(lambda c:build_media(c,args.archive.parent,args.work_dir,DOCS/'assets/cases',False,prior.get(c['id'],{}).get('partner_member')!=c['partner_member']),collect_targets(args.archive)))
        old['cases']=[c for c in old['cases'] if not c['id'].startswith('carla-target-')]+new
        (DOCS/'data/cases.json').write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n')
        print('TARGETS READY',len(new),flush=True)
    if not args.targets_only:build_maps(args.archive,args.workers)

if __name__=='__main__':main()
