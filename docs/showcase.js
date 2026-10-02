/* Static-site interactions. User evidence is never uploaded automatically. */
(() => {
  'use strict';
  const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const tracks = {carla:'CARLA-L2D', realsim:'RealSim', nuscenes:'nuScenes', navsim:'NAVSIM', wod:'WOD-E2E'};
  const kinds = {'closed-loop':'Closed-loop execution','open-loop':'Open-loop prediction',reference:'Reference trajectory',oracle:'Reference-controller execution',candidate:'Open-loop model comparison'};
  const role = c => c.evidence_role || (['reference','oracle'].includes(c.kind)?'benchmark-target':'model-result');
  const roleLabel = c => role(c)==='model-result'?'Model result':'Benchmark target';
  const el = (tag, text, cls) => { const n = document.createElement(tag); if(text !== undefined) n.textContent=text; if(cls) n.className=cls; return n; };
  const say = text => { const t=$('.toast'); t.textContent=text;t.classList.add('visible');setTimeout(()=>t.classList.remove('visible'),2400); };
  const download = (value, name) => { const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)+'\n'],{type:'application/json'}));const a=el('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000); };
  const scrollTo = node => node.scrollIntoView({behavior:reduced.matches?'instant':'smooth',block:'start'});
  let cases=[], filtered=[], current=null, shown=9;
  const modelNames = c => [...new Set([c.model,...(c.models||[]),...Object.keys(c.comparison?.models||{})])];
  window.L2DModels.ready.then(() => { if(current) { $('#case-model-label').replaceChildren(window.L2DModels.link(current.model)); window.L2DModels.references($('#case-paper-links'), modelNames(current)); if(current.comparison) drawTrajectories(); } });
  function modelOptions(){
    const value=$('#case-model').value, track=$('#case-track').value;
    $('#case-model').replaceChildren(new Option('All models & references','all'));
    [...new Set(cases.filter(c=>track==='all'||c.track===track).flatMap(modelNames))].sort().forEach(m=>$('#case-model').add(new Option(m,m)));
    if([...$('#case-model').options].some(o=>o.value===value)) $('#case-model').value=value;
  }
  function applyFilters(){
    const track=$('#case-track').value,model=$('#case-model').value,kind=$('#case-kind').value;
    const words=$('#case-search').value.toLowerCase().trim().split(/\s+/).filter(Boolean);
    filtered=cases.filter(c=>(track==='all'||c.track===track)&&(model==='all'||modelNames(c).includes(model))&&(kind==='all'||role(c)===kind)&&words.every(w=>[c.id,c.title,c.family,c.scene,c.scope?'Scope '+c.scope:'',...c.instructions].join(' ').toLowerCase().includes(w)));
    $$('.case-backends button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.caseTrack===track)));
    shown=9;renderGrid();
    if(filtered.length){
      const selected=filtered.find(c=>c.id===current?.id)||filtered[0];selectCase(selected);
    }else{
      $('#case-video').pause();$('#case-player-shell')?.setAttribute('hidden','');$('#case-viewer').hidden=true;
    }
  }
  function renderGrid(){
    const grid=$('#case-grid');grid.replaceChildren();
    const pcus=new Set(filtered.map(c=>c.track+'/'+c.pcu)).size;
    $('#case-count').textContent=`${filtered.length} ${filtered.length===1?'clip':'clips'} · ${pcus} paired ${pcus===1?'case':'cases'} · ${new Set(filtered.map(c=>c.track)).size} backends`;
    $('#case-empty').hidden=!!filtered.length;$('#case-more').hidden=shown>=filtered.length;
    filtered.slice(0,shown).forEach(c=>{
      const button=el('button',undefined,'case-card');button.type='button';button.dataset.caseId=c.id;button.setAttribute('aria-pressed',String(c.id===current?.id));
      const art=el('div',undefined,'case-card-art'),img=el('img');img.src=c.poster;img.alt=`${tracks[c.track]} · ${c.title} · ${c.model}`;img.loading='lazy';img.width=640;img.height=360;
      art.append(img,el('span',`▶ ${Math.round(c.duration_s)}s`));
      button.dataset.role=role(c);
      const body=el('div',undefined,'case-card-body');body.append(el('span',`${tracks[c.track]}${c.scope?' / SCOPE '+c.scope:''}`),el('strong',c.title),el('p',c.model),el('small',`${roleLabel(c)} · ${kinds[c.kind]}`));button.append(art,body);
      button.addEventListener('click',()=>{selectCase(c);scrollTo($('#case-viewer'));});grid.append(button);
    });
  }
  function selectCase(c){
    current=c;$('#case-viewer').hidden=false;
    const video=$('#case-video');
    if(video.dataset.caseId!==c.id){video.pause();video.dataset.caseId=c.id;video.poster=c.poster;video.src=c.video;video.load();$('#case-video-error').hidden=true;}
    video.playbackRate=Number($('#case-speed').value);
    $('#case-role').textContent=roleLabel(c);$('#case-role').dataset.role=role(c);
    $('#case-badge').textContent=`${tracks[c.track]} · ${c.visualization?'Execution replay':kinds[c.kind]}`;
    const index=filtered.indexOf(c);$('#case-counter').textContent=index>=0?`${index+1} / ${filtered.length}`:'Same-PCU comparison';
    $('#case-title').textContent=c.title;$('#case-model-label').replaceChildren(window.L2DModels.link(c.model));
    window.L2DModels.references($('#case-paper-links'), modelNames(c));
    $('#case-location').textContent=[c.pcu,c.scene,c.scope?'Scope '+c.scope:null].filter(Boolean).join(' / ');
    $('#case-instruction-a').textContent=c.instructions[0];$('#case-instruction-b').textContent=c.instructions[1];
    $('#case-note').textContent=c.visualization?'Model result: saved closed-loop trajectories with camera views rendered from the recorded poses. Dashed gray paths show benchmark targets.':({'closed-loop':'Model result: Alpamayo 1.5 driving in a NuRec reconstruction.', 'open-loop':'Model result: saved instruction-conditioned predictions. Dashed gray paths show benchmark targets.',reference:c.comparison?'Benchmark target: the video shows reference trajectories. The interactive panel below shows actual model predictions.':'Benchmark target: reference trajectories for the two requested behaviors.',oracle:'Benchmark target: execution by the reference controller.',candidate:'Model result: saved forecasts from four driving models under the paired instructions.'}[c.kind]);
    $('#case-download').href=c.video;$('#case-download').download=c.id+'.mp4';
    $('#case-source').textContent=`${tracks[c.track]} · ${c.pcu} · ${c.width} × ${c.height} · ${c.duration_s}s. Download the metadata for the source files, protocol and artifact identifiers.`;
    const variants=cases.filter(x=>x.track===c.track&&x.pcu===c.pcu);$('#case-variant-wrap').hidden=variants.length<2;
    $('#case-variant').replaceChildren(...variants.map(x=>new Option(`${roleLabel(x)} · ${x.model}`,x.id,false,x.id===c.id)));
    $('#case-perspectives').hidden=new Set(variants.map(role)).size<2;
    $$('[data-perspective]').forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.perspective===role(c)));b.onclick=()=>{const next=variants.find(x=>role(x)===b.dataset.perspective);if(next)selectVariant(next);};});
    $$('.case-card').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.caseId===c.id)));
    $('#case-prev').disabled=filtered.length<2;$('#case-next').disabled=filtered.length<2;
    $('#trajectory-inspector').hidden=!c.comparison;
    if(c.comparison){
      const previous=$('#trajectory-model').value, desired=$('#case-model').value;
      const names=Object.keys(c.comparison.models);$('#trajectory-model').replaceChildren(...names.map(n=>new Option(n,n)));
      $('#trajectory-model').value=names.includes(desired)?desired:names.includes(previous)?previous:names[0];drawTrajectories();
    }
    const url=new URL(location.href);url.searchParams.set('case',c.id);history.replaceState(null,'',url);
  }
  function drawTrajectories(){
    const data=current?.comparison;if(!data)return;
    window.L2DModels.references($('#trajectory-paper'), [$('#trajectory-model').value]);
    const svg=$('#trajectory-plot');svg.replaceChildren();
    const add=(tag,attrs,text)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text)n.textContent=text;svg.append(n);return n;};
    const paths=data.models[$('#trajectory-model').value], refs=data.reference;
    // Fix bounds across every model in this PCU. Forward points right, left points up.
    const all=[[0,0],...Object.values(data.models).flatMap(p=>[...p['0'],...p['1']]),...refs.flat()];
    const xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);
    const xmin=Math.min(...xs)-2,xmax=Math.max(...xs)+2,ymin=Math.min(...ys)-2,ymax=Math.max(...ys)+2;
    const scale=Math.min(660/Math.max(5,xmax-xmin),220/Math.max(5,ymax-ymin));
    const sx=x=>380+(x-(xmin+xmax)/2)*scale,sy=y=>145-(y-(ymin+ymax)/2)*scale;
    const span=Math.max(xmax-xmin,ymax-ymin),gridStep=span>100?20:span>40?10:5;
    for(let x=Math.ceil(xmin/gridStep)*gridStep;x<=xmax;x+=gridStep){add('line',{x1:sx(x),x2:sx(x),y1:30,y2:255,stroke:'#dce4d9'});add('text',{x:sx(x),y:274,'text-anchor':'middle','font-size':10,fill:'#63717a'},String(x));}
    for(let y=Math.ceil(ymin/gridStep)*gridStep;y<=ymax;y+=gridStep){add('line',{x1:50,x2:710,y1:sy(y),y2:sy(y),stroke:'#dce4d9'});add('text',{x:40,y:sy(y)+3,'text-anchor':'end','font-size':10,fill:'#63717a'},String(y));}
    add('text',{x:710,y:294,'text-anchor':'end','font-size':10,fill:'#63717a'},'+X forward (m)');add('text',{x:20,y:17,'font-size':10,fill:'#63717a'},'+Y left (m)');
    const count=Number($('#trajectory-time').value);$('#trajectory-time-label').textContent=(count?data.timestamps_s[count-1]:0).toFixed(1)+' s';
    const colors=['#2469c9','#ce6238'];
    [0,1].forEach(arm=>{
      const points=ps=>[[0,0],...ps.slice(0,count)].map(p=>`${sx(p[0])},${sy(p[1])}`).join(' ');
      add('polyline',{points:points(refs[arm]),fill:'none',stroke:colors[arm],'stroke-width':2,'stroke-dasharray':'7 5',opacity:.45});
      add('polyline',{points:points(paths[String(arm)]),fill:'none',stroke:colors[arm],'stroke-width':3.5,'stroke-linejoin':'round'});
      const last=count?paths[String(arm)][count-1]:[0,0];add('circle',{cx:sx(last[0]),cy:sy(last[1]),r:5,fill:colors[arm],stroke:'white','stroke-width':2});
    });
    add('circle',{cx:sx(0),cy:sy(0),r:4,fill:'#173240'});
  }
  function chooseBackend(track){$('#case-track').value=track;$('#case-kind').value='all';$('#case-search').value='';modelOptions();$('#case-model').value='all';applyFilters();}
  $('#case-track').addEventListener('change',()=>chooseBackend($('#case-track').value));
  ['#case-model','#case-kind'].forEach(s=>$(s).addEventListener('change',applyFilters));
  $('#case-search').addEventListener('input',applyFilters);
  $('#case-reset').addEventListener('click',()=>{$('#case-track').value='all';$('#case-kind').value='all';$('#case-search').value='';modelOptions();$('#case-model').value='all';applyFilters();});
  $('#case-more').addEventListener('click',()=>{shown+=9;renderGrid();});
  function selectVariant(c){if($('#case-kind').value!=='all')$('#case-kind').value=role(c);$('#case-model').value='all';applyFilters();selectCase(c);}
  $('#case-variant').addEventListener('change',()=>selectVariant(cases.find(c=>c.id===$('#case-variant').value)));
  ['prev','next'].forEach(direction=>$('#case-'+direction).addEventListener('click',()=>{if(filtered.length)selectCase(filtered[(Math.max(0,filtered.indexOf(current))+(direction==='next'?1:-1)+filtered.length)%filtered.length]);}));
  $('#case-speed').addEventListener('change',()=>{$('#case-video').playbackRate=Number($('#case-speed').value);});
  $('#case-video').addEventListener('error',()=>{$('#case-video-error').hidden=false;});
  $('#case-metadata').addEventListener('click',()=>{if(current)download(current,current.id+'.json');});
  $('#case-share').addEventListener('click',async()=>{if(!current)return;const u=new URL(location.href);u.hash='examples';u.searchParams.set('case',current.id);try{await navigator.clipboard.writeText(u.href);say('Case link copied');}catch{say('Copy this case URL from your address bar');}});
  $('#trajectory-model').addEventListener('change',drawTrajectories);$('#trajectory-time').addEventListener('input',drawTrajectories);
  $$('[data-case-track]').forEach(b=>b.addEventListener('click',()=>{chooseBackend(b.dataset.caseTrack);if(!b.closest('.case-backends'))scrollTo($('#examples'));}));
  fetch('data/cases.json').then(r=>{if(!r.ok)throw Error('manifest');return r.json();}).then(data=>{
    // Interleave backends so the first screen communicates coverage immediately.
    const queues=Object.keys(tracks).map(track=>data.cases.filter(c=>c.track===track).sort((a,b)=>Number(['oracle','reference'].includes(a.kind))-Number(['oracle','reference'].includes(b.kind))));
    cases=[];while(queues.some(q=>q.length))queues.forEach(q=>{if(q.length)cases.push(q.shift());});
    $$('[data-track-count]').forEach(n=>n.textContent=cases.filter(c=>n.dataset.trackCount==='all'||c.track===n.dataset.trackCount).length);
    modelOptions();filtered=cases;renderGrid();
    const requested=new URL(location.href).searchParams.get('case');selectCase(cases.find(c=>c.id===requested)||cases.find(c=>c.kind==='closed-loop')||cases[0]);
  }).catch(()=>{$('#case-title').textContent='The case library could not be loaded.';$('#case-count').textContent='Check the media build and reload this page.';});

  const steps=[
    ['01 / Shared world','Hold the non-language inputs fixed.','A pair starts from the same scene, seed and initial state. Each instruction requests a different, feasible outcome.'],
    ['02 / Paired instructions','Both alternatives must work.','Run the same model independently for each instruction. The evaluator checks the requested contract, rejects its opposite, and verifies execution requirements.'],
    ['03 / Four scopes','From a maneuver to a driving program.','Atomic action, persistent behavior, scene-grounded navigation and long-horizon instruction programs probe different demands on language control. Select a scope below.'],
    ['04 / Five tracks','Complementary evidence, native protocols.','CARLA and RealSim execute closed-loop drives. nuScenes, NAVSIM and WOD-E2E evaluate predicted trajectories in recorded environments. Inspect each backend’s evidence.']
  ];
  const scopes={A:['Atomic action','Choose a requested local maneuver: turn, change lane or park.'],B:['Persistent behavior','Keep a speed, lane or following constraint over the required interval.'],C:['Scene-grounded navigation','Resolve a destination or relation in the observed scene: a parking bay, crosswalk or obstacle.'],D:['Long-horizon instruction program','Execute an ordered program across multiple stages, references and destinations.']};
  let timer=null,motion=null,step=0;
  const stopTour=()=>{clearInterval(timer);timer=null;$('#framework-tour').setAttribute('aria-pressed','false');$('#framework-tour').textContent='▶ Guided tour';};
  function setStep(index){step=index;$('#framework').dataset.step=index;$$('.framework-steps button').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.step)===index)));['label','title','description'].forEach((s,i)=>$('#framework-'+s).textContent=steps[index][i]);$('#framework-case-link').onclick=null;if(index===1)animatePair();}
  function animatePair(){
    cancelAnimationFrame(motion);const start=performance.now();
    const frame=now=>{const p=reduced.matches?1:Math.min(1,(now-start)/2600),ease=1-Math.pow(1-p,2);['a','b'].forEach(a=>{const path=$('#framework-path-'+a),point=path.getPointAtLength(ease*path.getTotalLength()),dot=$('#framework-dot-'+a);dot.setAttribute('cx',point.x);dot.setAttribute('cy',point.y);});if(p<1&&!document.hidden)motion=requestAnimationFrame(frame);};motion=requestAnimationFrame(frame);
  }
  $$('.framework-steps button').forEach(b=>b.addEventListener('click',()=>{stopTour();setStep(Number(b.dataset.step));}));
  $('#framework-tour').addEventListener('click',()=>{if(timer){stopTour();return;}setStep(0);$('#framework-tour').setAttribute('aria-pressed','true');$('#framework-tour').textContent='Ⅱ Pause tour';timer=setInterval(()=>{if(step>=3){stopTour();return;}setStep(step+1);},5000);});
  $('#framework-replay').addEventListener('click',animatePair);
  $$('[data-framework-arm]').forEach(b=>b.addEventListener('click',()=>{$$('[data-framework-arm]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));['a','b'].forEach(a=>$('#framework-path-'+a).style.opacity=a===b.dataset.frameworkArm?1:.25);animatePair();}));
  $$('[data-scope]').forEach(b=>b.addEventListener('click',()=>{stopTour();setStep(2);$$('[data-scope]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));const s=b.dataset.scope;$('#framework-label').textContent='Scope '+s;$('#framework-title').textContent=scopes[s][0];$('#framework-description').textContent=scopes[s][1];$('#framework-case-link').onclick=()=>{$('#case-track').value='all';$('#case-kind').value='all';modelOptions();$('#case-model').value='all';$('#case-search').value='Scope '+s;applyFilters();};}));
  document.addEventListener('visibilitychange',()=>{if(document.hidden){stopTour();cancelAnimationFrame(motion);}});
  new IntersectionObserver(entries=>{if(!entries[0].isIntersecting){stopTour();cancelAnimationFrame(motion);}},{threshold:0}).observe($('#framework'));
  const film=$('#intro-film');
  $$('[data-film-time]').forEach(b=>b.addEventListener('click',()=>{film.currentTime=Number(b.dataset.filmTime);film.play().catch(()=>say('Press play to start the film'));scrollTo(film);}));
  film.addEventListener('timeupdate',()=>{$$('[data-film-time]').forEach((b,i,all)=>b.classList.toggle('active',film.currentTime>=Number(b.dataset.filmTime)&&(!all[i+1]||film.currentTime<Number(all[i+1].dataset.filmTime))));});
  [film,$('#case-video')].forEach(v=>v.addEventListener('play',()=>{[film,$('#case-video')].filter(x=>x!==v).forEach(x=>x.pause());}));
})();
