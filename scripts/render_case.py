"""Consistent 1080p case presentation from archived cameras and native trajectories.

The full-trajectory panels are redrawn from numeric evidence at equal XY scale.
Camera crops preserve the source movie's timing. No trajectories are synthesized.
"""
import csv
from functools import lru_cache
import io
import json
import math
from pathlib import Path
import subprocess
import zipfile
from PIL import Image, ImageDraw, ImageFont

PAPER='#f5f4ef'; INK='#173240'; MUTED='#64757b'; LINE='#dce2df'
BLUE='#2876ce'; ORANGE='#d46b40'; GREEN='#267866'; GRAY='#879b99'
COLORS=[BLUE,ORANGE,'#8a66b5',GREEN]
TRACKS={'carla':'CARLA-L2D','realsim':'RealSim','nuscenes':'nuScenes','navsim':'NAVSIM','wod':'WOD-E2E'}

@lru_cache(maxsize=50)
def font(n,bold=False):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',n)

def label(im,xy,s,size=20,color=INK,bold=False,anchor=None):
    ImageDraw.Draw(im).text(xy,s,font=font(size,bold),fill=color,anchor=anchor,spacing=7)

def wrap(s,size,width):
    lines=[];line=''
    for word in s.split():
        trial=(line+' '+word).strip()
        if font(size).getlength(trial)>width and line:lines.append(line);line=word
        else:line=trial
    return '\n'.join(lines+[line])

def role(case):
    return 'benchmark-target' if case['kind'] in ('reference','oracle','target') else 'model-result'

def role_name(case):
    return 'BENCHMARK TARGET' if role(case)=='benchmark-target' else 'MODEL RESULT'

def mode(case):
    if case['kind']=='target':return case['trajectory_panel']
    if case['kind']=='oracle':return 'Reference-controller execution'
    if case['kind']=='reference':return 'Reference trajectory'
    if case['track']=='carla':return 'Closed-loop execution · rendered replay'
    if case['track']=='realsim':return 'Closed-loop execution · NuRec camera'
    return 'Open-loop model prediction'

def rows(z,path):
    return list(csv.DictReader(io.StringIO(z.read(path).decode('utf-8-sig'))))

def plot_data(case,archive):
    """Return paths in a common native coordinate frame for the two arms."""
    result=[[],[]];refs=[[],[]];details={};track=case['track']
    with zipfile.ZipFile(archive) as z:
        base=case['member'].rsplit('/',1)[0]
        if track=='carla':
            path='pcu_video_evidence_20261002/cases/'+case['member'].split('/')[-1].replace('.mp4','')+'/data.json'
            data=json.loads(z.read(path));origin=data['arms'][0]['series'][0]
            xy=lambda p:[float(p['x'])-origin['x'],float(p['y'])-origin['y']]
            for a,arm in enumerate(data['arms']):
                result[a]=[(case['model'],[xy(p) for p in arm['series']])]
                refs[a]=[xy(p) for p in arm['oracle_path']]
            details={'frame':'World displacement (m)','speedup':max(1,max(a['duration_s'] for a in data['arms'])/60),'source_data':path}
        elif track=='realsim':
            series=[rows(z,base+f'/R{a}/executed_rig_trajectory_for_plot.csv') for a in range(2)]
            origin=series[0][0];ox,oy=float(origin['x']),float(origin['y'])
            for a,rs in enumerate(series):result[a]=[(case['model'],[[float(r['x'])-ox,float(r['y'])-oy] for r in rs])]
            details={'frame':'Rig displacement (m)','source_data':base+'/R*/executed_rig_trajectory_for_plot.csv'}
        elif track=='nuscenes':
            rs=rows(z,base+'/trajectories.csv')
            for a in range(2):
                arm=[r for r in rs if int(r['arm'])==a]
                result[a]=[(case['model'],[[float(r['displayed_x_m']),float(r['displayed_y_m'])] for r in arm])]
                if case['kind']!='reference':refs[a]=[[float(r['reference_x_m']),float(r['reference_y_m'])] for r in arm]
            details={'frame':'Ego at anchor (m) · 3 s horizon','source_data':base+'/trajectories.csv'}
        elif track=='navsim':
            for a in range(2):result[a]=[('Reference trajectory',[[0,0],*[p[:2] for p in case['comparison']['reference'][a]]])]
            details={'frame':'Rear axle at anchor (m) · 4 s horizon','source_data':case['member'].split('/')[0]+'/cases/'+case['pcu']+'.json'}
        else:
            base=case['member'].split('/cases/')[0]+'/cases/'+case['pcu']
            data=json.loads(z.read(base+'/model_predictions.json'))
            for a,arm in enumerate(['a','b']):
                for name,pred in data[arm].items():
                    ps=pred.get('xy_m',pred.get('trajectory'))
                    if ps is None:raise ValueError('Missing native prediction: '+name)
                    result[a].append((name,[[0,0],*[p[:2] for p in ps]]))
            details={'frame':'Rear axle at anchor (m) · 5 s horizon','source_data':base+'/model_predictions.json'}
    for arm in result:
        assert arm and all(points for _,points in arm)
    return result,refs,details


def dashed(draw,points,color=GRAY,width=3):
    # Constant dash spacing across all segments (including dense trace samples).
    distance=0
    for p,q in zip(points,points[1:]):
        length=math.dist(p,q)
        if length==0:continue
        pos=0
        while pos<length:
            phase=distance%16;step=min(length-pos, (9-phase if phase<9 else 16-phase))
            if phase<9:
                draw.line((p[0]+(q[0]-p[0])*pos/length,p[1]+(q[1]-p[1])*pos/length,p[0]+(q[0]-p[0])*(pos+step)/length,p[1]+(q[1]-p[1])*(pos+step)/length),fill=color,width=width)
            pos+=max(step,1e-6);distance+=max(step,1e-6)


def layout(case,archive,path):
    result,refs,details=plot_data(case,archive)
    im=Image.new('RGB',(1920,1080),PAPER);d=ImageDraw.Draw(im)
    label(im,(64,35),'Language2Drive',28,INK,True)
    label(im,(380,43),TRACKS[case['track']].upper(),19,MUTED,True)
    target=role(case)=='benchmark-target';accent=GREEN if target else BLUE
    d.rounded_rectangle((1450,29,1856,78),radius=24,fill=accent)
    label(im,(1653,53),role_name(case),20,'white',True,'mm')
    d.line((64,98,1856,98),fill=LINE,width=2)
    label(im,(64,118),case['title'],41,INK,True)
    model='Reference controller' if case['kind']=='oracle' else 'Reference trajectory' if target else case['model']
    # Keep long PCU identifiers in metadata; the scene remains readable in-video.
    label(im,(65,177),f"{model}  /  {case['scene']}"+(f"  /  Scope {case['scope']}" if case.get('scope') else ''),21,MUTED)
    allpts=[p for arm in result for _,ps in arm for p in ps]+[p for ps in refs for p in ps]+[[0,0]]
    xmin,xmax=min(p[0] for p in allpts),max(p[0] for p in allpts)
    ymin,ymax=min(p[1] for p in allpts),max(p[1] for p in allpts)
    span=max(xmax-xmin,ymax-ymin,5);pad=span*.05
    xmin-=pad;xmax+=pad;ymin-=pad;ymax+=pad
    scale=min(746/max(1,xmax-xmin),163/max(1,ymax-ymin))
    for a,x in enumerate([64,984]):
        col=[BLUE,ORANGE][a]
        d.rounded_rectangle((x,226,x+872,996),radius=18,fill='white',outline=LINE,width=2)
        d.rounded_rectangle((x+24,247,x+60,283),radius=9,fill=col)
        label(im,(x+42,265),'AB'[a],20,'white',True,'mm')
        label(im,(x+74,254),'INSTRUCTION  /  REQUESTED GOAL',17,col,True)
        size=25;instruction=wrap(case['instructions'][a],size,810)
        while d.multiline_textbbox((x+24,297),instruction,font=font(size),spacing=7)[3]>369:
            size-=1;instruction=wrap(case['instructions'][a],size,810)
        label(im,(x+24,297),instruction,size,INK)
        # Camera region is inserted by ffmpeg. Mode remains explicit next to it.
        label(im,(x+24,721),mode(case),17,MUTED)
        label(im,(x+24,763),('TARGET' if target else 'MODEL RESULT')+'  /  '+('FULL EXECUTED PATH' if case['track'] in ('carla','realsim') else 'PREDICTED PATH' if not target else 'REFERENCE PATH'),17,col,True)
        cx,cy=x+451,884
        sx=lambda v:cx+(v-(xmin+xmax)/2)*scale
        sy=lambda v:cy-(v-(ymin+ymax)/2)*scale
        raw_step=span/5;power=10**math.floor(math.log10(raw_step));step=min([1,2,5,10],key=lambda n:abs(n*power-raw_step))*power
        for k in range(math.ceil(xmin/step),math.floor(xmax/step)+1):
            xx=sx(k*step);d.line((xx,801,xx,965),fill=LINE);label(im,(xx,967),f'{k*step:g}',12,MUTED,anchor='ma')
        # Grid uses a single metric scale for X and Y and identical bounds in both arms.
        for k in range(math.ceil(ymin/step),math.floor(ymax/step)+1):
            yy=sy(k*step);d.line((x+78,yy,x+824,yy),fill=LINE);label(im,(x+65,yy),f'{k*step:g}',12,MUTED,anchor='rm')
        if refs[a]:dashed(d,[(sx(p[0]),sy(p[1])) for p in refs[a]])
        for i,(name,ps) in enumerate(result[a]):
            color=COLORS[i] if len(result[a])>1 else col
            pts=[(sx(p[0]),sy(p[1])) for p in ps]
            if len(pts)>1:d.line(pts,fill=color,width=4,joint='curve')
            for point,r,fill in [(pts[0],4,'white'),(pts[-1],6,color)]:
                px,py=point;d.ellipse((px-r,py-r,px+r,py+r),fill=fill,outline=color,width=2)
        label(im,(x+813,967),'X (m)',12,MUTED,anchor='ra');label(im,(x+30,800),'Y (m)',12,MUTED)
    if case['track']=='wod':
        x=64
        for i,(name,_) in enumerate(result[0]):
            d.line((x,1032,x+30,1032),fill=COLORS[i],width=4);label(im,(x+42,1020),name,17,MUTED);x+=310
    elif any(refs):
        d.line((64,1032,94,1032),fill=BLUE,width=4);label(im,(106,1020),'Model result',18,MUTED)
        dashed(d,[(285,1032),(327,1032)]);label(im,(339,1020),'Benchmark target',18,MUTED)
    else:label(im,(64,1020),mode(case),18,MUTED)
    label(im,(1856,1021),details['frame'],17,MUTED,anchor='ra')
    if details.get('speedup',1)>1:label(im,(1080,1021),f"Replay {details['speedup']:.2f}×",17,MUTED)
    im.save(path)
    case.update(evidence_role=role(case),evidence_label=role_name(case).title(),display_mode=mode(case),trajectory_source=details['source_data'],trajectory_panel='Full paths; identical bounds and equal XY scale for both arms')


# Camera-only windows in the original producers' 1280px-wide coordinate systems.
# Annotation strips and the original plots are excluded; source timing is retained.
CROPS={'carla':[(60,177,540,276),(700,177,540,276)],
       'realsim':[(16,164,624,330),(656,164,624,330)],
       'nuscenes':[(20,156,608,295),(652,156,608,295)],
       'navsim':[(20,135,608,298),(660,135,608,298)],
       'wod':[(24,106,606,514),(24,106,606,514)]}


def render(case,archive,src,dest,work):
    bg=work/(case['id']+'-layout.png');layout(case,archive,bg)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(src)]))
    stream=next(s for s in probe['streams'] if s['codec_type']=='video');factor=stream['width']/1280
    filters=['[1:v]split=2[sa][sb]']
    for a,(x,y,w,h) in enumerate(CROPS[case['track']]):
        x,y,w,h=[int(v*factor)//2*2 for v in (x,y,w,h)]
        filters.append(f'[{"sa" if a==0 else "sb"}]crop={w}:{h}:{x}:{y},scale=824:328:force_original_aspect_ratio=decrease,pad=824:328:(ow-iw)/2:(oh-ih)/2:color=0x173240,setsar=1[c{a}]')
    filters+=['[0:v][c0]overlay=88:381:shortest=1[v0]','[v0][c1]overlay=1008:381:shortest=1,format=yuv420p[out]']
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-loop','1','-framerate','12','-i',str(bg),'-i',str(src),'-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[out]','-an','-r','12','-c:v','libx264','-preset','fast','-crf','22','-threads','2','-movflags','+faststart',str(dest)],check=True)
