#!/usr/bin/env python3
"""Render the 120s / 1080p / 24fps English-captioned project film.

Uses original framework artwork and the curated viewing copies. Pillow draws
native title/diagram frames; ffmpeg preserves source video timing within excerpts.
No model inference, synthesized driving footage, speech, or rescoring is involved.
"""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import shutil
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs'; W,H,FPS=1920,1080,24
INK='#173240'; PAPER='#f6f5f0'; MUTED='#63717a'; BLUE='#2469c9'; ORANGE='#ce6238'; LINE='#d9ddd6'
FONTS={'sans':'/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
       'serif':'/usr/share/fonts/opentype/urw-base35/P052-Roman.otf',
       'mono':'/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'}

@lru_cache(maxsize=80)
def font(size,kind='sans'):
    return ImageFont.truetype(FONTS[kind],size)

def text(im,xy,value,size=28,color=INK,kind='sans',anchor=None):
    ImageDraw.Draw(im).text(xy,value,font=font(size,kind),fill=color,anchor=anchor,spacing=10)

def wrap(value,size,width,kind='sans'):
    lines=[];current=''
    for word in value.split():
        trial=(current+' '+word).strip()
        if font(size,kind).getlength(trial)>width and current:lines.append(current);current=word
        else:current=trial
    if current:lines.append(current)
    return '\n'.join(lines)

@lru_cache(maxsize=30)
def asset(name,size):
    im=Image.open(DOCS/'assets/framework'/name).convert('RGBA');im.thumbnail(size,Image.Resampling.LANCZOS);return im

def paste(im,name,box):
    x,y,w,h=box;art=asset(name,(w,h));im.paste(art,(int(x+(w-art.width)/2),int(y+(h-art.height)/2)),art)

def base(chapter,dark=False):
    im=Image.new('RGB',(W,H),INK if dark else PAPER);d=ImageDraw.Draw(im)
    fg=PAPER if dark else INK;mut='#aac2ca' if dark else MUTED
    text(im,(80,42),'L2D  /  LANGUAGE2DRIVE',19,fg,'mono');text(im,(1840,42),chapter.upper(),17,mut,'mono','ra')
    d.line((80,91,1840,91),fill='#35505d' if dark else LINE,width=1)
    return im

def caption(im,value,dark=False):
    d=ImageDraw.Draw(im);d.rectangle((0,967,W,H),fill='#102834' if dark else '#ecefe8')
    lines=wrap(value,25,1700);n=lines.count('\n')+1
    text(im,(W/2,1014-(n-1)*17),lines,25,PAPER if dark else INK,anchor='mm')

def pill(im,xy,value,color=BLUE):
    x,y=xy;d=ImageDraw.Draw(im);width=font(20,'mono').getlength(value)+32
    d.rounded_rectangle((x,y,x+width,y+40),radius=20,fill=color);text(im,(x+16,y+8),value,20,'white','mono')

def car(im,x,y,color=BLUE,angle=0):
    sprite=Image.new('RGBA',(60,104));d=ImageDraw.Draw(sprite)
    d.rounded_rectangle((7,4,53,100),radius=13,fill=color,outline='white',width=3)
    d.rounded_rectangle((15,22,45,40),radius=4,fill='#b5d9e6');d.rounded_rectangle((15,70,45,85),radius=3,fill='#b5d9e6')
    sprite=sprite.rotate(angle,expand=True,resample=Image.Resampling.BICUBIC);im.paste(sprite,(int(x-sprite.width/2),int(y-sprite.height/2)),sprite)

def road(im,box,t,arm=None):
    x,y,w,h=box;d=ImageDraw.Draw(im);cx=x+w*.58;cy=y+h*.42
    for dx in range(int(x),int(x+w),42):d.line((dx,y,dx,y+h),fill='#e5e8e0',width=1)
    for dy in range(int(y),int(y+h),42):d.line((x,dy,x+w,dy),fill='#e5e8e0',width=1)
    d.line((cx,y,cx,y+h),fill='#d1d9d2',width=112);d.line((x,cy,x+w,cy),fill='#d1d9d2',width=112)
    for dy in range(int(y),int(y+h),36):d.line((cx-19,dy,cx-19,dy+17),fill=PAPER,width=3)
    p=min(1,max(0,t));origin=(cx+15,y+h-35)
    for a,col in [('a',BLUE),('b',ORANGE)]:
        if arm and a!=arm:continue
        points=[origin]
        for i in range(1,81):
            q=i/80
            if a=='b':pos=(origin[0],origin[1]-q*(h-55))
            elif q<.42:pos=(origin[0],origin[1]-(origin[1]-(cy+30))*q/.42)
            else:
                theta=(q-.42)/.58*math.pi/2;pos=(origin[0]-(w*.45)*(1-math.cos(theta)),cy+30-65*math.sin(theta))
            if q<=p:points.append(pos)
        if len(points)>1:d.line(points,fill=col,width=7,joint='curve')
        car(im,*points[-1],col,90*max(0,(p-.42)/.58) if a=='a' else 0)

def scope_map(im,c,arm,box,progress):
    """Native CARLA road geometry and normalized reference-path progression."""
    x,y,w,h=box;tile=Image.new('RGB',(w,h),'#f0f3eb');d=ImageDraw.Draw(tile)
    bounds=c['bounds'];scale=min((w-70)/(bounds[2]-bounds[0]),(h-65)/(bounds[3]-bounds[1]))
    project=lambda p:(w/2+(p[0]-(bounds[0]+bounds[2])/2)*scale,h/2+(p[1]-(bounds[1]+bounds[3])/2)*scale)
    for gx in range(0,w,24):d.line((gx,0,gx,h),fill='#e5ebe1')
    for gy in range(0,h,24):d.line((0,gy,w,gy),fill='#e5ebe1')
    for color,extra in [('#ccd8cd',2),('#dfe7dc',-1)]:
        for lane in c['lanes']:
            ps=[project(p) for p in lane['points']]
            if len(ps)>1:d.line(ps,fill=color,width=max(1,int(lane['width']*scale+extra)),joint='curve')
    col=[BLUE,ORANGE][arm];ps=[project(p) for p in c['paths'][arm]]
    d.line(ps,fill='white',width=8,joint='curve');d.line(ps,fill=['#aac4e9','#e6b7a1'][arm],width=5,joint='curve')
    lengths=[math.dist(a,b) for a,b in zip(ps,ps[1:])];remaining=sum(lengths)*progress;shown=[ps[0]];heading=0
    for a,b,length in zip(ps,ps[1:],lengths):
        if length==0:continue
        ratio=min(1,max(0,remaining/length));shown.append((a[0]+(b[0]-a[0])*ratio,a[1]+(b[1]-a[1])*ratio));heading=-math.degrees(math.atan2(b[1]-a[1],b[0]-a[0]))-90
        remaining-=length
        if remaining<=0:break
    if len(shown)>1:d.line(shown,fill=col,width=5,joint='curve')
    for i,g in enumerate(c['goals']):
        if c['scope']!='D' and i!=arm:continue
        gx,gy=project(g['point']);name=g['label'];name=(str(2 if i==arm else 1)+' · '+name) if c['scope']=='D' else name
        text(tile,(max(65,min(w-65,gx)),max(18,gy-21)),name,17,col,anchor='mm')
        if g.get('polygon'):d.polygon([project(p) for p in g['polygon']],outline=col)
    # A small directional marker follows native reference points, not model output.
    px,py=shown[-1];d.ellipse((px-7,py-7,px+7,py+7),fill=col,outline='white',width=2)
    im.paste(tile,(x,y))

def make_card(kind,duration,subtitle,**kwargs):
    dark=kind in ('opening','closing')
    b=base(kwargs.get('chapter','The benchmark'),dark);d=ImageDraw.Draw(b)
    if kind=='opening':
        text(b,(95,145),'Can a driving VLA',96,PAPER,'serif');text(b,(95,250),'follow your instructions?',96,PAPER,'serif')
        text(b,(100,410),'Same scene. Same seed.',31,'#bbced7');text(b,(100,460),'Different instructions.',31,'#edac85')
        text(b,(100,690),'5',106,PAPER,'serif');text(b,(185,747),'BACKENDS',20,'#bbced7','mono')
        text(b,(435,690),'4',106,PAPER,'serif');text(b,(520,747),'SCOPES',20,'#bbced7','mono')
        text(b,(100,875),'DO VISION-LANGUAGE-ACTION MODELS REALLY FOLLOW THE INSTRUCTIONS?',19,'#aac2ca','mono')
        paste(b,'ego-car.webp',(1320,355,350,400))
    elif kind in ('shared','paired','measure'):
        titles={'shared':'Hold the world fixed.','paired':'Change only the instruction.','measure':'Measure both requested behaviors.'}
        text(b,(90,125),titles[kind],65,INK,'serif')
        if kind=='shared':
            paste(b,'carla-parking.webp',(90,280,700,300));paste(b,'realsim.webp',(90,610,700,300))
            for i,label in enumerate(['Same initial state','Same non-language inputs','Two feasible alternatives']):pill(b,(920,270+i*70),label,INK)
        else:
            pill(b,(110,247),'A  TURN LEFT',BLUE);pill(b,(1020,247),'B  GO STRAIGHT',ORANGE)
            if kind=='measure':
                text(b,(100,835),'Requested contract',28,BLUE);text(b,(585,835),'Opposite rejected',28,ORANGE);text(b,(1085,835),'Both arms complete',28,INK)
    elif kind=='scope':
        scope=kwargs['scope'];name,description=kwargs['info'];colors={'A':BLUE,'B':'#377c6e','C':'#a07b25','D':'#89759a'}
        pill(b,(90,133),'SCOPE '+scope,colors[scope]);text(b,(90,205),name,75,INK,'serif')
        c=next(c for c in json.loads((DOCS/'data/scope-cases.json').read_text())['cases'] if c['scope']==scope)
        text(b,(95,307),f"CARLA {c['pcu']} / {c['scene']}",21,MUTED)
        pill(b,(1440,300),'BENCHMARK TARGET','#267866')
        for arm,x in [(0,95),(1,990)]:
            text(b,(x,367),'AB'[arm]+'  '+wrap(c['instructions'][arm],24,790),24,[BLUE,ORANGE][arm])
        text(b,(100,897),description,25,MUTED)
    elif kind=='adapter':
        text(b,(90,138),'One adapter. Five evaluation backends.',72,INK,'serif')
        d.rounded_rectangle((95,285,885,790),radius=18,fill=INK)
        text(b,(135,333),'YOUR MODEL',20,'#a5c1cd','mono')
        text(b,(135,410),'def predict(request):\n    # cameras + ego history\n    # active instruction\n    return {"waypoints": xy}',27,PAPER,'mono')
        for i,(name,detail) in enumerate([('CARLA','Closed-loop simulation'),('RealSim','Closed-loop reconstruction'),('Open-loop','nuScenes / NAVSIM / WOD-E2E')]):
            y=310+i*170;d.line((925,y+50,1010,y+50),fill=BLUE,width=3);text(b,(1050,y),name,43,INK,'serif');text(b,(1050,y+65),detail,24,MUTED)
        text(b,(100,865),'Public observations in. Ego-frame trajectories out.',30,MUTED)
    elif kind=='closing':
        text(b,(95,150),'Does your model',100,PAPER,'serif');text(b,(95,272),'follow both instructions?',100,PAPER,'serif')
        for i,label in enumerate(['01  Adapt your model','02  Run the paired evaluation','03  Share scores + native evidence']):text(b,(100,490+i*94),label,34,'#c4d8df')
        text(b,(100,855),'github.com/ZihanYou/Language2Drive',38,'#edac85','mono')
        paste(b,'coverage.webp',(1300,480,410,410))
    elif kind=='predictions':
        text(b,(90,130),'Inspect the model’s response.',68,INK,'serif')
        text(b,(95,228),'MODEL RESULT / NAVSIM / Qwen-Drive-1.0-4B / saved predictions',24,MUTED)
        text(b,(100,869),'Solid: model result     Dashed: benchmark target     Blue: A / Orange: B',23,MUTED)
    caption(b,subtitle,dark)
    def frame(t):
        im=b.copy();d=ImageDraw.Draw(im);p=min(1,t/3)
        if kind=='opening':
            # Native vector orbit suggests two alternatives, without simulated footage.
            for radius in (190,260,325):d.arc((1495-radius,555-radius,1495+radius,555+radius),-100,-100+min(300,t*60),fill='#466473',width=2)
            for col,angle in [(BLUE,t*.6),(ORANGE,t*.6+math.pi)]:
                x=1495+260*math.cos(angle);y=555+260*math.sin(angle);d.ellipse((x-9,y-9,x+9,y+9),fill=col)
        elif kind=='shared':road(im,(950,515,810,385),p,'a')
        elif kind in ('paired','measure'):
            road(im,(110,330,745,460),p,'a');road(im,(1020,330,745,460),p,'b')
        elif kind=='scope':
            c=next(c for c in json.loads((DOCS/'data/scope-cases.json').read_text())['cases'] if c['scope']==kwargs['scope'])
            scope_map(im,c,0,(95,458,825,400),p)
            scope_map(im,c,1,(990,458,825,400),p)
        elif kind=='predictions':
            data=kwargs['comparison'];paths=data['models']['Qwen-Drive-1.0-4B'];refs=data['reference'];allpts=[[0,0],*paths['0'],*paths['1'],*refs[0],*refs[1]]
            xmax=max(pt[0] for pt in allpts)+2;ymin=min(pt[1] for pt in allpts)-2;ymax=max(pt[1] for pt in allpts)+2
            scale=min(1450/xmax,480/(ymax-ymin));sx=lambda x:200+x*scale;sy=lambda y:730-(y-ymin)*scale
            for x in range(0,int(xmax)+1,5):d.line((sx(x),315,sx(x),805),fill=LINE);text(im,(sx(x),808),str(x),18,MUTED,anchor='ma')
            count=min(8,max(1,int(t/4*8)+1))
            for arm,col in [(0,BLUE),(1,ORANGE)]:
                for i in range(1,len(refs[arm])):
                    a,c=refs[arm][i-1],refs[arm][i]
                    for j in range(0,10,2):d.line((sx(a[0]+(c[0]-a[0])*j/10),sy(a[1]+(c[1]-a[1])*j/10),sx(a[0]+(c[0]-a[0])*(j+1)/10),sy(a[1]+(c[1]-a[1])*(j+1)/10)),fill=col,width=2)
                pts=[(sx(x),sy(y)) for x,y,*_ in [[0,0],*paths[str(arm)][:count]]];d.line(pts,fill=col,width=6);x,y=pts[-1];d.ellipse((x-8,y-8,x+8,y+8),fill=col)
            text(im,(1660,310),f'{count*.5:.1f} s',28,MUTED,'mono')
        # Restrained progress accent, continuous within every chapter.
        d.rectangle((80,91,80+1760*min(1,t/duration),94),fill=ORANGE if dark else BLUE)
        # Fade from the project paper/ink color, never cover source evidence with motion.
        if t<.3:im=Image.blend(Image.new('RGB',(W,H),INK if dark else PAPER),im,t/.3)
        return im
    return frame

def encode_frames(path,duration,fn):
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
         '-an','-c:v','libx264','-preset','fast','-crf','19','-threads','3','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for index in range(int(duration*FPS)):proc.stdin.write(fn(index/FPS).tobytes())
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('ffmpeg frame encoder failed')
    finally:
        if proc.poll() is None:proc.kill();proc.wait()

def encode_case(path,case,duration,subtitle,work,start=0,tag=None):
    # The gallery source already contains role, A/B goals, cameras and native paths.
    # Keep that entire composition; reserve a separate band for the film caption.
    b=Image.new('RGB',(W,H),PAPER)
    caption(b,subtitle);bg=work/(path.stem+'.png');b.save(bg)
    vf=f'[1:v]fps={FPS},scale=1712:964,setsar=1,tpad=stop_mode=clone:stop_duration={duration}[v];[0:v][v]overlay=104:0:shortest=1,format=yuv420p[out]'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-loop','1','-framerate',str(FPS),'-i',str(bg),'-ss',str(start),'-i',str(DOCS/case['video']),'-filter_complex_threads','1','-filter_complex',vf,'-map','[out]','-an','-t',str(duration),'-c:v','libx264','-crf','19','-preset','fast','-threads','3','-movflags','+faststart',str(path)],check=True)

def stamp(seconds):
    return f'{int(seconds)//3600:02}:{int(seconds)//60%60:02}:{int(seconds)%60:02}.000'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work-dir',type=Path,default=Path('/tmp/l2d-film'));p.add_argument('--force',action='store_true');p.add_argument('--refresh-cases',action='store_true');args=p.parse_args();work=args.work_dir;work.mkdir(parents=True,exist_ok=True)
    dest=DOCS/'assets/demo';dest.mkdir(parents=True,exist_ok=True)
    cases={c['id']:c for c in json.loads((DOCS/'data/cases.json').read_text())['cases']}
    previous = json.loads((dest/'storyboard.json').read_text()) if (dest/'storyboard.json').exists() else {'segments': []}
    previous_cases = {s['case_id']: s for s in previous['segments'] if s['kind']=='case'}
    segments=[];captions=[];story=[];clock=0
    def card(kind,duration,subtitle,**kwargs):
        nonlocal clock
        path=work/f'{len(segments):02}-{kind}.mp4'
        if args.force or not path.exists():encode_frames(path,duration,make_card(kind,duration,subtitle,**kwargs))
        record=dict(start_s=clock,duration_s=duration,kind=kind,caption=subtitle)
        if kind=='scope':
            geometry=next(c for c in json.loads((DOCS/'data/scope-cases.json').read_text())['cases'] if c['scope']==kwargs['scope'])
            record.update(scope=geometry['scope'],case_id=geometry['case_id'],evidence_role='benchmark-target',source_member=geometry['source_member'],source_sha256=geometry['source_sha256'],animation='Normalized reference-path progression')
        elif kind=='predictions':
            record.update(evidence_role='model-result',model='Qwen-Drive-1.0-4B',case_id=kwargs['case_id'])
        segments.append(path);captions.append((clock,clock+duration,subtitle));story.append(record);clock+=duration;print('READY',path.name,clock,flush=True)
    def clip(key,duration,subtitle,start=0,tag=None):
        nonlocal clock
        path=work/f'{len(segments):02}-{key}.mp4';case=cases[key]
        old = previous_cases.get(key, {})
        changed = old.get('source_sha256') != case['video_sha256'] or old.get('caption') != subtitle or old.get('source_offset_s') != start or old.get('duration_s') != duration
        if args.force or args.refresh_cases or changed or not path.exists():encode_case(path,case,duration,subtitle,work,start,tag)
        segments.append(path);captions.append((clock,clock+duration,subtitle));story.append(dict(start_s=clock,duration_s=duration,kind='case',case_id=key,evidence_role=case['evidence_role'],source_offset_s=start,source_sha256=case['video_sha256'],caption=subtitle));clock+=duration;print('READY',path.name,clock,flush=True)
    card('opening',12,'Language2Drive asks whether driving models follow language across two feasible alternatives.',chapter='The question')
    card('shared',8,'A Paired Counterfactual Unit keeps the scene, seed and initial state fixed.',chapter='The paired mechanism')
    card('paired',8,'Run the same model independently. Change only the mutually exclusive instruction.',chapter='The paired mechanism')
    card('measure',4,'Both alternatives must satisfy their execution requirements for strict paired success.',chapter='The paired mechanism')
    for s,name,description in [('A','Atomic action','Select the requested turn, lane change or parking maneuver.'),('B','Persistent behavior','Maintain a speed, lane or following constraint over time.'),('C','Scene-grounded navigation','Connect words to locations and objects in the observed scene.'),('D','Long-horizon instruction program','Remember the goal and execute the required stages in order.')]:
        card('scope',4,description,scope=s,info=(name,description),chapter='Four instruction scopes')
    clip('carla-formal-a-014',7,'Model result: CARLA execution replay under paired lane instructions.',start=4,tag='Execution replay')
    clip('carla-formal-d-009',7,'Conditional destinations bring scene understanding and multi-stage instructions into the same evaluation.',start=8,tag='Execution replay')
    clip('realsim-alpamayo15-p02-a2-s08',12,'Model result: Alpamayo 1.5 in reconstructed RealSim driving scenes.')
    clip('nuscenes-p01-scene-0077-speed-opendrivevla-0-5b',5,'Compare instruction-conditioned model forecasts in the same recorded camera context.',tag='Open-loop prediction / excerpt')
    clip('nuscenes-p01-scene-0077-speed-omnidrive',5,'Switch models on the same paired case to inspect their responses to the requested speed.',tag='Open-loop prediction / excerpt')
    nav='navsim-lane-left-c2e92aa83a599e79'
    clip(nav,8,'Benchmark targets: NAVSIM reference trajectories define the two requested outcomes.',tag='Reference trajectories')
    card('predictions',4,'Compare instruction-conditioned model predictions on a shared coordinate scale.',comparison=cases[nav]['comparison'],case_id=nav,chapter='The evidence / NAVSIM')
    clip('wod-cand-0040',4,'WOD-E2E benchmark targets pair stopping with continued forward progress in the same observed scene.',tag='Benchmark targets')
    clip('wod-cand-0038',4,'Paired speed requirements define the requested outcomes over shared real-world camera context.',tag='Benchmark targets')
    card('adapter',8,'Adapt once, configure the resources, and run the native evaluations across three runtime groups.',chapter='Evaluate and contribute')
    card('closing',8,'Evaluate your model. Share native scores, execution coverage and reproducible evidence.',chapter='Join the benchmark')
    assert clock==120
    concat=work/'segments.txt';concat.write_text(''.join("file '"+str(path).replace("'","'\\''")+"'\n" for path in segments))
    final=work/'language2drive-film.mp4';subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(final)],check=True);shutil.copyfile(final,dest/final.name)
    (dest/'captions.vtt').write_text(('WEBVTT\n\n'+''.join(f'{stamp(a)} --> {stamp(b)}\n{c}\n\n' for a,b,c in captions)).rstrip()+'\n')
    (dest/'storyboard.json').write_text(json.dumps({'duration_s':clock,'fps':FPS,'resolution':[W,H],'audio':'none; English captions are burned in','segments':story},indent=2)+'\n')
    poster=make_card('opening',12,captions[0][2],chapter='The question')(5);poster.save(dest/'poster.webp',quality=92)
    print('EXPORTED',dest/final.name,flush=True)

if __name__=='__main__':main()
