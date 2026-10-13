/* Six-map atlas: source lane geometry, location photos and recorded camera routes. */
(() => {
  'use strict';
  const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
  const svg=$('#atlas-map'), video=$('#atlas-video'), dialog=$('#atlas-lightbox');
  if(!svg)return;
  const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
  const ns=(tag,attrs={},text)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;return n;};
  let maps=[],cases=[],map=null,place=null,view='overview',tour=null,mode='places',zoom=1,center=[380,230],projection=null,drag=null,routeGroup=null,cursor=null;
  const urlState=()=>{const u=new URL(location.href);u.searchParams.set('map',map.id);u.searchParams.set('place',place.id);u.searchParams.set('view',view);u.searchParams.set('tour',tour.id);history.replaceState(null,'',u);};
  const coords=p=>`X ${p.x.toFixed(1)} · Y ${p.y.toFixed(1)} · Z ${p.z.toFixed(1)} m`;
  const screenUnit=()=>{const r=svg.getBoundingClientRect();return 1/(Math.max(.001,Math.min(r.width/760,r.height/460))*zoom);};
  function cameraLabel(){return {overview:'Overview',forward:'Road forward',reverse:'Road reverse'}[view];}
  function setMode(value){
    mode=value;$$('[data-atlas-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.atlasMode===mode)));
    $('#atlas-photo-view').hidden=mode!=='places';$('#atlas-tour-view').hidden=mode!=='tours';
    if(mode==='places')video.pause();
    if(routeGroup)routeGroup.setAttribute('visibility',mode==='tours'?'visible':'hidden');
  }
  function showPhoto(){
    const asset=place.views[view]||place.views.overview;
    $('#atlas-photo').src=asset.path;$('#atlas-photo').alt=`${map.title} · ${place.label} · ${cameraLabel()}`;
    $('#atlas-place-title').textContent=place.label;$('#atlas-place-counter').textContent=`${map.locations.indexOf(place)+1} / ${map.locations.length}`;
    $('#atlas-place-coordinates').textContent=coords(place);$('#atlas-photo-download').href=asset.path;
    $('#atlas-photo-download').download=`${map.id}-${place.id}-${view}.webp`;
    $$('[data-atlas-view]').forEach(b=>{b.disabled=!place.views[b.dataset.atlasView];b.setAttribute('aria-pressed',String(b.dataset.atlasView===view));});
    showLarge();
  }
  function selectPlace(p,{focus=false,update=true}={}){
    place=p;setMode('places');showPhoto();
    $$('.atlas-pin, .atlas-places button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.place===p.id)));
    if(focus){center=projection([p.x,p.y]);zoom=Math.max(zoom,3);updateViewport();}
    if(update)urlState();
  }
  function showLarge(){
    $('#atlas-large-photo').src=(place.views[view]||place.views.overview).path;
    $('#atlas-large-photo').alt=`${map.title} · ${place.label} · ${cameraLabel()}`;
    $('#atlas-lightbox-title').textContent=`${map.title} / ${place.label}`;
    $('#atlas-large-caption').textContent=`${cameraLabel()} · ${coords(place)}`;
  }
  function movePhoto(delta){const i=map.locations.indexOf(place);selectPlace(map.locations[(i+delta+map.locations.length)%map.locations.length]);}
  function updateViewport(){
    const w=760/zoom,h=460/zoom;svg.setAttribute('viewBox',`${center[0]-w/2} ${center[1]-h/2} ${w} ${h}`);
    $('#atlas-zoom-label').textContent=`${zoom.toFixed(1)}× · drag to explore`;
    $('#atlas-scale').textContent=`${(70/projection.scale/zoom).toFixed(0)} m`;
    const placed=[],unit=screenUnit();
    const fit=p=>zoom===1?[Math.min(760-16*unit,Math.max(16*unit,p[0])),Math.min(460-16*unit,Math.max(16*unit,p[1]))]:p;
    $$('.atlas-pin').forEach(g=>{
      const anchor=[Number(g.dataset.x),Number(g.dataset.y)];let p=fit(anchor);
      if(placed.some(q=>Math.hypot(p[0]-q[0],p[1]-q[1])<30*unit)){
        outer:for(const radius of [27,42,58])for(let i=0;i<12;i++){
          const candidate=fit([anchor[0]+radius*unit*Math.cos(i*Math.PI/6),anchor[1]+radius*unit*Math.sin(i*Math.PI/6)]);
          if(!placed.some(q=>Math.hypot(candidate[0]-q[0],candidate[1]-q[1])<30*unit)){p=candidate;break outer;}
        }
      }
      placed.push(p);const circle=g.querySelector('circle'),text=g.querySelector('text'),leader=g.querySelector('line');
      circle.setAttribute('cx',p[0]);circle.setAttribute('cy',p[1]);circle.setAttribute('r',14*unit);circle.style.strokeWidth=1.6*unit+'px';
      text.style.fontSize=11*unit+'px';text.setAttribute('x',p[0]);text.setAttribute('y',p[1]+4*unit);
      leader.setAttribute('x2',p[0]);leader.setAttribute('y2',p[1]);leader.setAttribute('stroke-width',unit);
    });
    if(cursor){cursor.setAttribute('r',6*unit);cursor.setAttribute('stroke-width',2*unit);}
    if(routeGroup)routeGroup.querySelectorAll('polyline').forEach(n=>n.setAttribute('stroke-width',2*unit));
  }
  function zoomBy(f){zoom=Math.min(12,Math.max(1,zoom*f));if(zoom===1)center=[380,230];updateViewport();}
  function drawMap(){
    svg.replaceChildren();zoom=1;center=[380,230];
    const ps=map.lanes.flatMap(l=>l.points),xs=ps.map(p=>p[0]),ys=ps.map(p=>p[1]);
    const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),scale=Math.min(650/Math.max(1,xmax-xmin),350/Math.max(1,ymax-ymin));
    projection=p=>[380+(p[0]-(xmin+xmax)/2)*scale,230+(p[1]-(ymin+ymax)/2)*scale];projection.scale=scale;
    const defs=ns('defs'),pattern=ns('pattern',{id:'atlas-grid',width:40,height:40,patternUnits:'userSpaceOnUse'});pattern.append(ns('path',{d:'M 40 0 L 0 0 0 40',fill:'none',stroke:'#d7e2d1','stroke-width':.7}));defs.append(pattern);svg.append(defs,ns('rect',{x:-3000,y:-3000,width:6800,height:6500,fill:'url(#atlas-grid)'}));
    // Both lane surfaces and centerlines follow the source samples, in one metric scale.
    const road=ns('g'),line=ns('g');
    map.lanes.forEach(l=>{const points=l.points.map(p=>projection(p).join(',')).join(' ');road.append(ns('polyline',{points,fill:'none',stroke:l.junction?'#bbcdb5':'#a7bda0','stroke-width':Math.max(.8,l.width_m*scale+1.3),'stroke-linecap':'round','stroke-linejoin':'round'}));line.append(ns('polyline',{points,fill:'none',stroke:'#fffefa','stroke-width':Math.max(.6,l.width_m*scale*.64),'stroke-linecap':'round','stroke-linejoin':'round'}));});svg.append(road,line);
    const title=ns('text',{x:35,y:30,fill:'#62816c','font-family':'monospace','font-size':10,'letter-spacing':2},map.id.toUpperCase());svg.append(title);
    routeGroup=ns('g',{visibility:mode==='tours'?'visible':'hidden'});svg.append(routeGroup);
    map.locations.forEach((p,i)=>{const [x,y]=projection([p.x,p.y]);const pin=ns('g',{class:'atlas-pin',role:'button',tabindex:0,'aria-label':`${i+1}. ${p.label}: open location photos`,'aria-pressed':String(p.id===place.id),'data-place':p.id,'data-x':x,'data-y':y});pin.append(ns('title',{},p.label),ns('line',{x1:x,y1:y,x2:x,y2:y,stroke:'#527a68'}),ns('circle',{cx:x,cy:y,r:12}),ns('text',{x,y:y+4},String(i+1)));pin.addEventListener('click',()=>selectPlace(p));pin.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectPlace(p);}});svg.append(pin);});
    updateViewport();drawRoute();
  }
  function drawRoute(){
    if(!routeGroup||!tour)return;routeGroup.replaceChildren();
    // Cut road-level camera paths at recorded viewpoint switches.
    let segments=[[]];const cuts=tour.cuts_s||[];
    tour.route.forEach((p,i)=>{if(i&&cuts.some(t=>t>tour.route[i-1][0]&&t<=p[0]))segments.push([]);segments.at(-1).push(projection([p[1],p[2]]));});
    segments.filter(s=>s.length>1).forEach(s=>routeGroup.append(ns('polyline',{class:'atlas-route',points:s.map(p=>p.join(',')).join(' ')})));
    cursor=ns('circle',{class:'atlas-cursor',r:6});routeGroup.append(cursor);updateViewport();updateCursor();
  }
  function updateCursor(){
    if(!tour||!cursor)return;
    const r=tour.route;let i=0;while(i<r.length-2&&r[i+1][0]<=video.currentTime)i++;
    const p=r[i],q=r[Math.min(i+1,r.length-1)];
    const cut=(tour.cuts_s||[]).some(t=>t>p[0]&&t<=q[0]);
    const t=cut?0:Math.min(1,Math.max(0,(video.currentTime-p[0])/Math.max(.001,q[0]-p[0])));
    const xy=projection([p[1]+(q[1]-p[1])*t,p[2]+(q[2]-p[2])*t]);cursor.setAttribute('cx',xy[0]);cursor.setAttribute('cy',xy[1]);
  }
  function selectTour(t,{show=true,update=true}={}){
    tour=t;video.pause();video.src=t.media.path;video.poster=t.media.poster;video.load();$('#atlas-media-error').hidden=true;
    $('#atlas-tour-info').textContent=`${t.label} · ${Math.round(t.duration_s)} s · ${t.vehicles} traffic vehicles`;
    $('#atlas-tour-download').href=t.media.path;$('#atlas-tour-download').download=`${map.id}-${t.id}.mp4`;
    $$('#atlas-tours button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.tour===t.id)));
    drawRoute();if(show)setMode('tours');if(update)urlState();
  }
  function showCases(){
    const box=$('#atlas-case-links');box.replaceChildren(el('span','PAIRED TARGETS','mini'));
    const selected=cases.filter(c=>c.track==='carla'&&c.kind==='target'&&c.scene===map.id&&c.camera_view==='exterior');
    selected.forEach(c=>{const b=el('button',`${c.pcu} · ${c.title}`);b.type='button';b.onclick=()=>window.L2DShowcase?.select(c.id);box.append(b);});
    if(!selected.length)box.append(el('span','Explore this map’s control corridor and traffic scenes.'));
  }
  function selectMap(m,{initial=false}={}){
    map=m;video.pause();view='overview';mode='places';
    const u=new URL(location.href),initialPlace=initial&&m.locations.find(p=>p.id===u.searchParams.get('place'));
    place=initialPlace||m.locations[0];if(initial&&place.views[u.searchParams.get('view')])view=u.searchParams.get('view');
    tour=initial&&m.tours.find(t=>t.id===u.searchParams.get('tour'))||m.tours[0];
    $('#atlas-title').textContent=m.title;$('#atlas-description').textContent=m.description;$('#atlas-tag').textContent=`${m.tag.toUpperCase()} / ${m.id}`;
    $('#atlas-condition').textContent=`${m.lighting==='night'?'Night scene':'Daylight'} · ${m.locations.length} locations · 4 tours`;
    $$('#atlas-cards button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.map===m.id)));
    const locations=$('#atlas-places');locations.replaceChildren();m.locations.forEach((p,i)=>{const b=el('button');b.append(el('b',String(i+1)),document.createTextNode(p.label));b.dataset.place=p.id;b.setAttribute('aria-pressed',String(p.id===place.id));b.onclick=()=>selectPlace(p);locations.append(b);});
    const tours=$('#atlas-tours');tours.replaceChildren();m.tours.forEach(t=>{const b=el('button',t.label);b.dataset.tour=t.id;b.setAttribute('aria-pressed',String(t===tour));b.onclick=()=>selectTour(t);tours.append(b);});
    drawMap();showPhoto();selectTour(tour,{show:false,update:false});setMode('places');showCases();$('#atlas-shell').setAttribute('aria-busy','false');
    if(!initial)urlState();
  }
  $$('[data-atlas-view]').forEach(b=>b.onclick=()=>{view=b.dataset.atlasView;showPhoto();urlState();});
  $$('[data-atlas-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.atlasMode));
  $('#atlas-zoom-in').onclick=()=>zoomBy(1.4);$('#atlas-zoom-out').onclick=()=>zoomBy(1/1.4);
  $('#atlas-reset').onclick=()=>{zoom=1;center=[380,230];updateViewport();};
  svg.addEventListener('pointerdown',e=>{if(e.pointerType!=='mouse'||e.button!==0||e.target.closest('.atlas-pin'))return;drag={x:e.clientX,y:e.clientY,center:[...center]};svg.setPointerCapture(e.pointerId);svg.classList.add('dragging');});
  svg.addEventListener('pointermove',e=>{if(!drag)return;const scale=screenUnit();center=[drag.center[0]-(e.clientX-drag.x)*scale,drag.center[1]-(e.clientY-drag.y)*scale];updateViewport();});
  const endDrag=()=>{drag=null;svg.classList.remove('dragging');};svg.addEventListener('pointerup',endDrag);svg.addEventListener('pointercancel',endDrag);
  svg.addEventListener('wheel',e=>{if(!e.ctrlKey)return;e.preventDefault();zoomBy(e.deltaY<0?1.15:1/1.15);},{passive:false});
  svg.addEventListener('keydown',e=>{if(e.target!==svg)return;if(['+','=','-','0','ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();if(['+','='].includes(e.key))zoomBy(1.4);else if(e.key==='-')zoomBy(1/1.4);else if(e.key==='0')$('#atlas-reset').click();else{const delta=40/zoom;center[0]+=e.key==='ArrowLeft'?-delta:e.key==='ArrowRight'?delta:0;center[1]+=e.key==='ArrowUp'?-delta:e.key==='ArrowDown'?delta:0;updateViewport();}}});
  $('#atlas-open-photo').onclick=()=>{showLarge();dialog.showModal();};$('#atlas-close-photo').onclick=()=>dialog.close();
  $('#atlas-photo-prev').onclick=()=>movePhoto(-1);$('#atlas-photo-next').onclick=()=>movePhoto(1);
  dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});dialog.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'){e.preventDefault();movePhoto(-1);}if(e.key==='ArrowRight'){e.preventDefault();movePhoto(1);}});
  video.addEventListener('timeupdate',updateCursor);video.addEventListener('seeked',updateCursor);video.addEventListener('error',()=>{$('#atlas-media-error').hidden=false;});
  const players=[video,$('#case-video'),$('#intro-film')].filter(Boolean);players.forEach(v=>v.addEventListener('play',()=>players.filter(p=>p!==v).forEach(p=>p.pause())));
  document.addEventListener('visibilitychange',()=>{if(document.hidden)video.pause();});new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)video.pause();}).observe($('#simulator'));
  new ResizeObserver(()=>{if(projection)updateViewport();}).observe(svg);
  $('#atlas-share').onclick=async()=>{if(!map)return;urlState();const u=new URL(location.href);u.hash='simulator';u.searchParams.delete('case');try{await navigator.clipboard.writeText(u.href);const b=$('#atlas-share');b.textContent='Map link copied ✓';setTimeout(()=>b.textContent='Copy map link ↗',2400);}catch{history.replaceState(null,'',u);}};
  Promise.all([fetch('data/maps.json').then(r=>{if(!r.ok)throw Error('maps');return r.json();}),fetch('data/cases.json').then(r=>r.json())]).then(([data,evidence])=>{
    maps=data.maps;cases=evidence.cases;maps.forEach((m,i)=>{const b=el('button',undefined,'atlas-card');b.dataset.map=m.id;b.type='button';const im=el('img');im.src=m.thumbnail;im.alt=`${m.title} lane network`;im.loading='lazy';im.width=272;im.height=144;const text=el('span');text.append(el('strong',m.title),el('small',m.tag));b.append(im,text,el('i',String(i+1).padStart(2,'0')));b.onclick=()=>selectMap(m);$('#atlas-cards').append(b);});
    selectMap(maps.find(m=>m.id===new URL(location.href).searchParams.get('map'))||maps[0],{initial:true});
  }).catch(error=>{$('#atlas-title').textContent='The map atlas could not be loaded.';$('#atlas-description').textContent='Reload the page to load the map media.';$('#atlas-shell').setAttribute('aria-busy','false');console.error(error);});
})();
