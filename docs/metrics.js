/* Interactive explanations of the manuscript metrics. Values here are examples;
 * the published result table is read independently from data/results.json.
 * Source: method.tex, Eqs. pair / graded / metric-lbal / l2d; appendix.tex.
 */
(() => {
  'use strict';
  const root = document.querySelector('#metric-explorer');
  if (!root) return;
  const $ = selector => root.querySelector(selector);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const mean = (a, b) => (a + b) / 2;
  const balanced = values => (values.reduce((sum, v) => sum + Math.sqrt(v), 0) / 3) ** 2;
  const composite = (families, safety, driving) => mean(safety, driving) / 100 * (1 + .99 * balanced(families));
  const fmt = n => Number(n).toFixed(2);
  const definitions = {
    SISR: ['Strict instruction success rate', 'strict', 'How often is an individual request fully executed?', 'Each arm receives 1 when the complete requested contract, opposite-contract rejection and all applicable execution conditions hold.', '100 × ⟨(sA + sB) / 2⟩'],
    CFPS: ['Counterfactual paired success', 'strict', 'Can the model complete both alternatives?', 'A pair receives credit only when both strict verdicts are 1. The scene, initial state and seed stay matched across the two executions.', '100 × ⟨sA × sB⟩'],
    SemISR: ['Relaxed semantic instruction success', 'relaxed', 'Does an arm meet its semantic criterion?', 'The arm receives 1 when its full requested criterion or a predefined task-specific relaxation holds, such as a relaxed dwell or geometry condition.', '100 × ⟨(mA + mB) / 2⟩'],
    'Sem-CFPS': ['Relaxed semantic paired success', 'relaxed', 'Do both arms meet their semantic criteria?', 'Both arms must satisfy their full requested criterion or its predefined relaxation. Each task supplies its own semantic rule.', '100 × ⟨mA × mB⟩'],
    SemSISR: ['Exclusive semantic instruction success', 'exclusive', 'Does the prediction select the requested alternative?', 'An open-loop arm succeeds when its predicted trajectory satisfies its own semantic contract and rejects the opposite contract.', 'BA = ownA × (1 − oppositeA); 100 × ⟨(BA + BB) / 2⟩'],
    SemCFPS: ['Exclusive semantic paired success', 'exclusive', 'Do both predictions select their requested alternatives?', 'Both predicted trajectories must pass their own contract and reject the opposite contract under matched inputs.', 'Bj = ownj × (1 − oppositej); 100 × ⟨BA × BB⟩'],
    SCS: ['Graded semantic compliance', 'graded', 'How fully is each request completed?', 'The native task map assigns each arm a compliance value from 0 to 1. SCS averages the two values; geometric, temporal and behavioral criteria define that credit.', '100 × ⟨(cA + cB) / 2⟩'],
    PCS: ['Paired graded compliance', 'graded', 'How much does the weaker alternative complete?', 'PCS takes the lower compliance value in each pair. Improving the weaker arm increases paired credit; the gap to SCS describes imbalance between the arms.', '100 × ⟨min(cA, cB)⟩'],
    PSM: ['Predictive safety margin', 'margin', 'How much time remains before a predicted collision?', 'At each instant, CARLA maps the minimum predicted time to collision (TTC) to a margin capped at 3 seconds. Longer TTC gives more safety credit.', '100 × meanroutes[Σ Δt · clip(TTCmin / 3, 0, 1) / Σ Δt]'],
    Safety: ['Collision- and off-road-free safety', 'safety', 'Does an execution stay collision-free and on-road?', 'A RealSim execution receives safety credit when both conditions hold throughout the evaluation. The reported Safety value averages this binary verdict across scheduled runs.', '100 × meanruns[collision-free AND off-road-free]'],
    DrivingScore: ['Reference-path advancement', 'driving', 'How far along the reference path does the vehicle reach?', 'The evaluator projects the recorded drive onto the arm’s reference path and measures the maximum valid normalized advancement.', '100 × meaneligible runs[max valid reference-path progress]'],
    L2D: ['Language following × driving quality', 'summary', 'How do the language and driving scores combine?', 'Three families summarize strict success, relaxed semantic success and graded compliance. Their balanced mean is combined with the track’s safety and driving scores.', 'L2D = (D / 100) × (1 + 0.99 × Lbal)']
  };
  const groups = {strict:'Strict execution', relaxed:'Relaxed semantics', exclusive:'Prediction semantics', graded:'Graded completion', margin:'Driving quality', safety:'Driving quality', driving:'Driving quality', summary:'Summary'};
  let track, trackId = 'carla', selected = 'SISR';
  const button = (text, action, className = '') => {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = text; b.className = className; b.addEventListener('click', action); return b;
  };
  const read = id => Number($('#mx-' + id).value);
  const on = id => $('#mx-' + id).checked ? 1 : 0;
  const range = (id, label, value, max = 100, step = 1) => `<label class="mx-range" for="mx-${id}"><span>${label}<output id="mx-${id}-value" for="mx-${id}">${value}</output></span><input id="mx-${id}" type="range" min="0" max="${max}" step="${step}" value="${value}"></label>`;
  const check = (id, label, checked) => `<label class="mx-check"><input id="mx-${id}" type="checkbox" ${checked ? 'checked' : ''}><span>${label}</span></label>`;
  const scores = names => `<div class="mx-scores">${names.map((name, i) => `<div class="mx-score${name === selected ? ' selected' : ''}"><span>${name}</span><output id="mx-score-${i}">0.00</output><div class="mx-meter"><i id="mx-meter-${i}"></i></div></div>`).join('')}</div>`;
  function updateScores(values) {
    values.forEach((v, i) => { $('#mx-score-' + i).textContent = fmt(v); $('#mx-meter-' + i).style.width = v + '%'; });
  }
  function updateRanges() {
    root.querySelectorAll('input[type="range"]').forEach(input => {
      $('#' + input.id + '-value').textContent = input.max === '6' ? Number(input.value).toFixed(1) + ' s' : fmt(input.value);
      input.style.setProperty('--fill', (100 * input.value / input.max) + '%');
    });
  }
  const pairArt = () => `<svg class="mx-pair-art" viewBox="0 0 560 145" role="img" aria-label="Credit for arm A and arm B on a zero to one hundred scale">
    <text x="0" y="15" class="mx-svg-caption">ARM-LEVEL CREDIT · 0–100</text>
    <text x="0" y="58" class="mx-svg-label">A</text><text x="0" y="111" class="mx-svg-label">B</text>
    <path d="M32 52H464 M32 105H464" class="mx-rail"/>
    <path d="M32 52H464" pathLength="100" id="mx-arm-a" class="mx-credit mx-blue"/>
    <path d="M32 105H464" pathLength="100" id="mx-arm-b" class="mx-credit mx-orange"/>
    <text id="mx-arm-a-score" x="490" y="58" class="mx-svg-number">100</text><text id="mx-arm-b-score" x="490" y="111" class="mx-svg-number">0</text>
    <text x="32" y="140" class="mx-svg-caption">0</text><text x="448" y="140" class="mx-svg-caption">100</text></svg>`;
  function updatePair(a, b) {
    ['a', 'b'].forEach((arm, i) => {
      const value = [a, b][i]; $('#mx-arm-' + arm).style.strokeDasharray = `${value} 100`;
      $('#mx-arm-' + arm + '-score').textContent = Number(value.toFixed(1));
    });
  }
  function announce(message) { $('#mx-insight').textContent = message; }
  function attach(update) {
    $('#mx-demo').querySelectorAll('input').forEach(input => input.addEventListener('input', () => { updateRanges(); update(); }));
    updateRanges(); update();
  }
  function pairedDemo(kind) {
    const graded = kind === 'graded', exclusive = kind === 'exclusive', relaxed = kind === 'relaxed';
    const names = graded ? ['SCS', 'PCS'] : exclusive ? ['SemSISR', 'SemCFPS'] : relaxed ? ['SemISR', 'Sem-CFPS'] : ['SISR', 'CFPS'];
    const controls = graded ? range('a', 'Arm A · compliance (%)', 100) + range('b', 'Arm B · compliance (%)', 40) : ['a', 'b'].map((arm, i) => `<fieldset class="mx-arm"><legend>Arm ${arm.toUpperCase()}</legend>${exclusive ? check(arm + '-own', 'Own contract satisfied', true) + check(arm + '-opposite', 'Opposite contract satisfied', i === 1) : check(arm, relaxed ? 'Semantic criterion satisfied' : 'Full execution successful', i === 0)}</fieldset>`).join('');
    $('#mx-demo').innerHTML = `<div class="mx-demo-kicker">INTERACTIVE EXAMPLE · ONE PCU / ONE EXPRESSION</div>${pairArt()}<div class="mx-pair-controls">${controls}</div>${scores(names)}<p id="mx-insight" class="mx-insight" aria-live="polite"></p><div class="mx-presets"></div>`;
    const update = () => {
      const a = graded ? read('a') / 100 : exclusive ? on('a-own') * (1 - on('a-opposite')) : on('a');
      const b = graded ? read('b') / 100 : exclusive ? on('b-own') * (1 - on('b-opposite')) : on('b');
      const individual = 100 * mean(a, b), paired = 100 * (graded ? Math.min(a, b) : a * b);
      updatePair(a * 100, b * 100); updateScores([individual, paired]);
      announce(graded ? `The weaker arm contributes ${fmt(paired)} paired credit. The ${fmt(individual - paired)}-point gap equals half the difference between the arms.` : `This pair contributes ${fmt(individual)} to ${names[0]} and ${fmt(paired)} to ${names[1]}. ${a && b ? 'Both alternatives receive credit.' : a || b ? 'Completing the other arm brings paired success to 100.' : 'Each arm needs its own successful verdict.'}`);
    };
    const set = (a, b) => {
      if (graded) { $('#mx-a').value = a; $('#mx-b').value = b; }
      else if (exclusive) { $('#mx-a-own').checked = !!a; $('#mx-b-own').checked = !!b; $('#mx-a-opposite').checked = false; $('#mx-b-opposite').checked = false; }
      else { $('#mx-a').checked = !!a; $('#mx-b').checked = !!b; }
      updateRanges(); update();
    };
    (graded ? [['Balanced · 70 / 70',70,70], ['Uneven · 100 / 40',100,40]] : [['Both complete',1,1], ['Only A completes',1,0], ['Neither completes',0,0]]).forEach(([label,a,b]) => $('.mx-presets').append(button(label, () => set(a,b))));
    attach(update);
  }
  function drivingDemo(kind) {
    if (kind === 'safety') {
      $('#mx-demo').innerHTML = `<div class="mx-demo-kicker">INTERACTIVE EXAMPLE · ONE EXECUTION</div><div class="mx-safety-symbol" aria-hidden="true">✓</div><div class="mx-pair-controls">${check('collision', 'Collision-free throughout', true)}${check('road', 'On-road throughout', true)}</div>${scores(['Safety credit'])}<p id="mx-insight" class="mx-insight" aria-live="polite"></p>`;
      attach(() => { const value = on('collision') * on('road') * 100; updateScores([value]); $('.mx-safety-symbol').textContent = value ? '✓' : '○'; $('.mx-safety-symbol').dataset.pass = String(!!value); announce(`This execution contributes ${value} safety credit. The RealSim table averages the verdicts over all 60 scheduled slots.`); });
      return;
    }
    const margin = kind === 'margin';
    $('#mx-demo').innerHTML = `<div class="mx-demo-kicker">INTERACTIVE EXAMPLE · ${margin ? 'ONE INSTANT' : 'ONE ELIGIBLE EXECUTION'}</div>
      ${margin ? `<svg class="mx-driving-art" viewBox="0 0 560 170" role="img" aria-label="TTC scale from zero to six seconds, with full margin at three seconds"><rect x="280" y="45" width="248" height="70" rx="10" fill="#e0ede3"/><text x="292" y="31" class="mx-svg-caption">FULL MARGIN FROM 3 s</text><path d="M32 80H528" class="mx-rail"/><path d="M32 80H528" class="mx-credit mx-blue" pathLength="100" id="mx-ttc-line"/><circle id="mx-ttc-marker" cx="156" cy="80" r="10" fill="#2469c9" stroke="white" stroke-width="3"/><text x="32" y="141" class="mx-svg-label">0 s</text><text x="280" y="141" class="mx-svg-label">3 s</text><text x="506" y="141" class="mx-svg-label">6 s</text></svg>` : `<svg class="mx-driving-art" viewBox="0 0 560 170" role="img" aria-label="Maximum advancement along an example reference path"><path id="mx-road" d="M35 127 H170 C255 127 225 48 330 48 H520" fill="none" stroke="#e4e9e8" stroke-width="34" stroke-linecap="round"/><path d="M35 127 H170 C255 127 225 48 330 48 H520" fill="none" stroke="white" stroke-width="2" stroke-dasharray="7 7"/><path id="mx-route-progress" d="M35 127 H170 C255 127 225 48 330 48 H520" fill="none" stroke="#2469c9" stroke-width="4" pathLength="100"/><g id="mx-car"><rect x="-13" y="-7" width="26" height="14" rx="4" fill="#173240" stroke="white" stroke-width="2"/><rect x="3" y="-4" width="5" height="8" rx="1" fill="#b6d6ef"/></g><text x="25" y="160" class="mx-svg-caption">START</text><text x="457" y="22" class="mx-svg-caption">PATH END</text></svg>`}
      ${range(margin ? 'ttc' : 'progress', margin ? 'Minimum predicted TTC' : 'Maximum valid path progress (%)', margin ? 1.5 : 65, margin ? 6 : 100, margin ? .1 : 1)}${scores([margin ? 'Instantaneous margin' : 'DrivingScore credit'])}<p id="mx-insight" class="mx-insight" aria-live="polite"></p>`;
    attach(() => {
      if (margin) {
        const ttc = read('ttc'), value = Math.min(ttc / 3, 1) * 100;
        $('#mx-ttc-marker').setAttribute('cx', 32 + ttc / 6 * 496); $('#mx-ttc-line').style.strokeDasharray = `${ttc / 6 * 100} 100`;
        updateScores([value]); announce(`${ttc.toFixed(1)} s TTC → ${fmt(value)} instantaneous margin. PSM time-averages these margins within each route, then equally averages the 24 CARLA diagnostic routes.`);
      } else {
        const value = read('progress'), path = $('#mx-road'), length = path.getTotalLength(), t = value / 100 * length;
        const p = path.getPointAtLength(t), p0 = path.getPointAtLength(Math.max(0,t-1)), p1 = path.getPointAtLength(Math.min(length,t+1));
        $('#mx-car').setAttribute('transform', `translate(${p.x} ${p.y}) rotate(${Math.atan2(p1.y-p0.y,p1.x-p0.x)*180/Math.PI})`);
        $('#mx-route-progress').style.strokeDasharray = `${value} 100`; updateScores([value]);
        announce(trackId === 'carla' ? 'CARLA uses the maximum valid reference-path projection. Hold targets are excluded from this distance-based measure.' : 'RealSim measures the furthest reference-path projection within its 5 m corridor and fixed horizon, then averages over the 60 scheduled slots.');
      }
    });
  }
  function summaryDemo() {
    const safety = trackId === 'carla' ? 'PSM' : 'Safety';
    $('#mx-demo').innerHTML = `<div class="mx-demo-kicker">INTERACTIVE L2D CALCULATOR</div><div class="mx-presets" id="mx-summary-presets"></div><label class="mx-model-select">Start from a reported model<select id="mx-model"><option value="">Custom example</option></select></label>
      <div class="mx-families"><div>${range('strict', 'a₁ · Strict', 70)}<small>(SISR + CFPS) / 2</small></div><div>${range('semantic', 'a₂ · Semantic', 70)}<small>(SemISR + Sem-CFPS) / 2</small></div><div>${range('graded', 'a₃ · Graded', 70)}<small>(SCS + PCS) / 2</small></div></div>
      <div class="mx-driving-inputs">${range('safety', safety, 80)}${range('driving', 'DrivingScore', 80)}</div>
      <div class="mx-equation-flow"><div><span>Language factor · 1 + 0.99Lbal</span><output id="mx-language-factor">0.00</output><small>Lbal = <span id="mx-balanced">0.00</span> · mean(√a)²</small></div><b aria-hidden="true">×</b><div><span>Driving factor · D / 100</span><output id="mx-driving-factor">0.0000</output><small>D = <span id="mx-driving-mean">0.00</span> · mean(${safety}, DrivingScore)</small></div><b aria-hidden="true">→</b></div>${scores(['L2D'])}<p id="mx-insight" class="mx-insight" aria-live="polite"></p>`;
    track.rows.forEach((row, i) => $('#mx-model').add(new Option(row.model, String(i))));
    // Preserve the precision of published values when loading a model; sliders
    // use 0.005 so that the mean of two two-decimal metrics is representable.
    $('#mx-demo').querySelectorAll('input').forEach(input => { input.step = '.005'; });
    const update = () => {
      const families = ['strict', 'semantic', 'graded'].map(read), driving = mean(read('safety'), read('driving'));
      $('#mx-balanced').textContent = fmt(balanced(families)); $('#mx-driving-mean').textContent = fmt(driving);
      $('#mx-language-factor').textContent = fmt(1 + .99 * balanced(families)); $('#mx-driving-factor').textContent = (driving / 100).toFixed(4);
      updateScores([composite(families, read('safety'), read('driving'))]);
      const row = $('#mx-model').value === '' ? null : track.rows[Number($('#mx-model').value)];
      announce(row ? `${row.model} · published L2D: ${fmt(row.L2D)}. This calculation uses the displayed component scores.` : `Language-family mean: ${fmt(families.reduce((a,b)=>a+b,0)/3)}. Balanced language score: ${fmt(balanced(families))}. Try balanced and uneven profiles with the same family mean.`);
    };
    const set = values => { ['strict','semantic','graded','safety','driving'].forEach((id,i) => { $('#mx-'+id).value=values[i]; }); updateRanges(); update(); };
    $('#mx-model').addEventListener('change', () => {
      if ($('#mx-model').value === '') return;
      const row = track.rows[Number($('#mx-model').value)];
      set([mean(row.SISR,row.CFPS),mean(row.SemISR,row['Sem-CFPS']),mean(row.SCS,row.PCS),row[safety],row.DrivingScore]);
    });
    [['Balanced', [70,70,70,80,80]], ['Uneven', [100,100,10,80,80]], ['All 100', [100,100,100,100,100]]].forEach(([label,values]) => $('#mx-summary-presets').append(button(label, () => { $('#mx-model').value='';set(values); })));
    $('#mx-demo').querySelectorAll('input').forEach(input => input.addEventListener('input', () => { $('#mx-model').value=''; }));
    attach(update);
  }
  function selectMetric(metric, focus = false) {
    if (!track?.metrics.includes(metric)) return;
    selected = metric; root.dataset.metric = metric;
    const [title, kind, question, description, formula] = definitions[metric];
    $('#mx-symbol').textContent = metric; $('#mx-title').textContent = title;
    $('#mx-question').textContent = question; $('#mx-description').textContent = description;
    $('#mx-formula').textContent = formula;
    root.querySelectorAll('[data-explain]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.explain === metric)));
    if (['strict','relaxed','exclusive','graded'].includes(kind)) pairedDemo(kind);
    else if (kind === 'summary') summaryDemo(); else drivingDemo(kind);
    const extras = kind === 'relaxed' ? 'The semantic rule is task-specific; opposite-contract rejection is an explicit requirement of the strict verdict.' : kind === 'exclusive' ? 'The semantic verdict is evaluated on the predicted trajectory over the track’s declared horizon.' : kind === 'graded' ? 'Compliance comes from the native task map. Multiple requirements are combined by that map; the sliders directly set its output.' : kind === 'margin' ? 'TTC uses the recorded constant-velocity oriented-box model. A TTC of infinity maps to full margin.' : kind === 'summary' ? 'Compute L2D from the aggregated component scores within each track. The table above keeps its published results as you explore.' : 'The blue and orange arms share the same non-language inputs and are evaluated in separate executions.';
    $('#mx-detail').textContent = kind === 'driving' || kind === 'safety' ? 'Read driving quality alongside the instruction-following scores to understand both aspects of the evaluation.' : extras;
    const paired = ['strict','relaxed','exclusive','graded'].includes(kind);
    $('#mx-aggregation').textContent = paired ? 'From this pair to the leaderboard: average expressions within each PCU, then give every PCU equal weight. All scores are on a 0–100 scale; higher is better.' : kind === 'summary' ? 'The square-root mean rewards strength across all three language families. D combines the two native driving measurements, on the same 0–100 scale.' : 'The diagram shows the local scoring rule. Track-level aggregation produces the 0–100 values in the leaderboard.';
    if (focus) { $('#mx-title').focus({preventScroll:true}); root.scrollIntoView({behavior:reduced.matches?'instant':'smooth', block:'start'}); }
  }
  function update(id, data) {
    const changed = id !== trackId;
    trackId = id; track = data;
    $('#mx-context').textContent = data.label + (['carla','realsim'].includes(id) ? ' · Closed loop' : ' · Open loop');
    $('#mx-tabs').replaceChildren();
    const rows = new Map();
    data.metrics.forEach(metric => {
      const group = groups[definitions[metric][1]];
      if (!rows.has(group)) {
        const row = document.createElement('div'); row.className = 'mx-tab-group';
        const label = document.createElement('span'); label.textContent = group; row.append(label); rows.set(group,row); $('#mx-tabs').append(row);
      }
      const b = button(metric, () => selectMetric(metric)); b.dataset.explain = metric; b.setAttribute('aria-controls','mx-body'); rows.get(group).append(b);
    });
    if (changed || !data.metrics.includes(selected)) selected = data.metrics.includes('SISR') ? 'SISR' : 'SemSISR';
    root.hidden = false; selectMetric(selected);
  }
  window.L2DMetrics = {update, select: selectMetric};
})();
