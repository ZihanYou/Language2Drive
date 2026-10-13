/* Native CARLA map geometry and benchmark reference paths; SVG remains editable. */
(() => {
  'use strict';
  const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
  const colors=['#2469c9','#ce6238'],ns='http://www.w3.org/2000/svg';
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  let catalog=[],current=null,raf=null,progress=1;
  const node=(tag,attrs={},text)=>{const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;return n;};
  const line=ps=>ps.map((p,i)=>`${i?'L':'M'}${p[0].toFixed(2)} ${p[1].toFixed(2)}`).join(' ');
  function diagram(c,arm,small=false){
    const w=600,h=350,pad=small?34:48,b=c.bounds;
    const scale=Math.min((w-2*pad)/(b[2]-b[0]),(h-2*pad)/(b[3]-b[1]));
    const project=p=>[w/2+(p[0]-(b[0]+b[2])/2)*scale,h/2+(p[1]-(b[1]+b[3])/2)*scale];
    const svg=node('svg',{viewBox:`0 0 ${w} ${h}`,role:'img','aria-label':`CARLA ${c.pcu}: benchmark target ${arm===null?'pair':'AB'[arm]}`});
    svg.append(node('rect',{width:w,height:h,rx:12,fill:'#f0f3eb'}));
    // Fine grid and native lane centerlines form a quiet, geometric atlas.
    for(let x=0;x<w;x+=24)svg.append(node('line',{x1:x,x2:x,y1:0,y2:h,stroke:'#e5ebe1','stroke-width':.6}));
    for(let y=0;y<h;y+=24)svg.append(node('line',{x1:0,x2:w,y1:y,y2:y,stroke:'#e5ebe1','stroke-width':.6}));
    const roads=node('g');
    c.lanes.forEach(l=>roads.append(node('path',{d:line(l.points.map(project)),fill:'none',stroke:'#cdd7ce','stroke-width':Math.max(2,l.width*scale+2),'stroke-linejoin':'round','stroke-linecap':'round'})));
    c.lanes.forEach(l=>roads.append(node('path',{d:line(l.points.map(project)),fill:'none',stroke:'#e0e7de','stroke-width':Math.max(1,l.width*scale-.7),'stroke-linejoin':'round','stroke-linecap':'round'})));
    c.lanes.forEach(l=>roads.append(node('path',{d:line(l.points.map(project)),fill:'none',stroke:'#ffffffb5','stroke-width':1,'stroke-dasharray':'7 6'})));
    svg.append(roads);
    c.goals.forEach((g,i)=>{
      if(g.polygon)svg.append(node('path',{d:line(g.polygon.map(project))+'Z',fill:colors[i]+'20',stroke:colors[i],'stroke-width':1.3}));
    });
    const arms=arm===null?[0,1]:[arm];
    arms.forEach(a=>{
      const d=line(c.paths[a].map(project));
      svg.append(node('path',{d,fill:'none',stroke:'white','stroke-width':7,'stroke-linecap':'round','stroke-linejoin':'round'}));
      svg.append(node('path',{d,fill:'none',stroke:colors[a],opacity:.22,'stroke-width':4,'stroke-linecap':'round','stroke-linejoin':'round'}));
      const path=node('path',{d,fill:'none',stroke:colors[a],'stroke-width':small?4:3.5,'stroke-linecap':'round','stroke-linejoin':'round',class:small?'':'scope-route'});
      svg.append(path);
      if(!small){
        const car=node('g',{class:'scope-car','data-arm':a});
        car.append(node('rect',{x:-10,y:-6,width:20,height:12,rx:4,fill:colors[a],stroke:'white','stroke-width':1.5}));
        car.append(node('rect',{x:2,y:-4,width:4,height:8,rx:1,fill:'#dfedf2'}));svg.append(car);
      }
    });
    if(!small){
      const [x,y]=project(c.paths[0][0]);svg.append(node('circle',{cx:x,cy:y,r:4,fill:'white',stroke:'#173240','stroke-width':1.5}));
      svg.append(node('text',{x:x-7,y:y+19,fill:'#4e645d','text-anchor':'end','font-size':10},'START'));
      const destinations=c.scope==='D'?c.goals:c.goals.filter((_,i)=>i===arm);
      destinations.forEach(g=>{
        const [gx,gy]=project(g.point),label=c.scope==='D'?`${g===c.goals[arm]?2:1} · ${g.label}`:g.label;
        const tx=Math.max(48,Math.min(552,gx)),ty=Math.max(20,Math.min(328,gy-19)),width=Math.max(51,label.length*6+18);
        svg.append(node('rect',{x:tx-width/2,y:ty-11,width,height:22,rx:5,fill:'white',stroke:colors[arm],'stroke-width':.8}));
        svg.append(node('text',{x:tx,y:ty+3.4,'text-anchor':'middle',fill:colors[arm],'font-size':10,'font-weight':600},label));
      });
      // A metric scale bar uses the same rigid transform as the plotted paths.
      const meters=[1,2,5,10,20].filter(n=>n*scale<72).pop()||1;
      svg.append(node('path',{d:`M24 ${h-30}v5h${meters*scale}v-5`,fill:'none',stroke:'#73877c','stroke-width':1}));
      svg.append(node('text',{x:24,y:h-11,fill:'#73877c','font-size':9},`${meters} m`));
    }
    return svg;
  }
  function seek(value){
    progress=value;$('#scope-progress').value=Math.round(value*100);$('#scope-progress-label').textContent=Math.round(value*100)+'%';
    $$('.scope-map').forEach(svg=>{
      const path=svg.querySelector('.scope-route'),car=svg.querySelector('.scope-car');if(!path)return;
      const length=path.getTotalLength(),at=length*value,p=path.getPointAtLength(at),q=path.getPointAtLength(Math.min(length,at+.8)),r=path.getPointAtLength(Math.max(0,at-.8));
      path.style.strokeDasharray=length;path.style.strokeDashoffset=length*(1-value);
      car.setAttribute('transform',`translate(${p.x},${p.y}) rotate(${Math.atan2(q.y-r.y,q.x-r.x)*180/Math.PI})`);
    });
  }
  function stop(){cancelAnimationFrame(raf);raf=null;$('#scope-play').textContent='▶ Animate routes';$('#scope-play').setAttribute('aria-pressed','false');}
  function play(){
    if(raf){stop();return;}
    if(reduced.matches){seek(1);return;}
    const offset=progress>=1?0:progress,start=performance.now();
    $('#scope-play').textContent='Ⅱ Pause';$('#scope-play').setAttribute('aria-pressed','true');
    function tick(now){const p=Math.min(1,offset+(now-start)/6500);seek(p);if(p<1)raf=requestAnimationFrame(tick);else stop();}raf=requestAnimationFrame(tick);
  }
  function select(scope){
    const c=catalog.find(x=>x.scope===scope);if(!c)return;stop();current=c;
    $('#scope-case-name').textContent=`Scope ${scope} · ${c.pcu}`;$('#scope-case-scene').textContent=c.scene;
    [0,1].forEach(a=>{
      const svg=diagram(c,a);svg.classList.add('scope-map');$(`#scope-map-${a}`).replaceChildren(svg);
      $(`#scope-command-${a}`).textContent=c.instructions[a];
    });
    $('#scope-watch').href=`?case=${c.case_id}#examples`;
    seek(1);
  }
  $('#scope-play').addEventListener('click',play);
  $('#scope-progress').addEventListener('input',e=>{stop();seek(Number(e.target.value)/100);});
  $('#scope-watch').addEventListener('click',e=>{e.preventDefault();if(current)window.L2DShowcase?.select(current.case_id);});
  $$('[data-scope]').forEach(b=>b.addEventListener('click',()=>select(b.dataset.scope)));
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)stop();}).observe($('#scope-explorer'));
  fetch('data/scope-cases.json').then(r=>{if(!r.ok)throw Error('Scope geometry');return r.json();}).then(data=>{
    catalog=data.cases;
    catalog.forEach(c=>{const b=$(`[data-scope="${c.scope}"]`);b.querySelector('img')?.replaceWith(diagram(c,null,true));});
    select('A');
  }).catch(()=>{$('#scope-case-name').textContent='CARLA paired trajectories';$('#scope-case-scene').textContent='Reload to load the reference paths.';});
})();
