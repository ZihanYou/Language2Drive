/* Replay the model's recorded CoT using the evidence archive's video clock. */
(() => {
  'use strict';
  const $ = s => document.querySelector(s), video = $('#case-video');
  const panel = $('#case-reasoning'), shell = $('#case-viewer'), layout = $('#case-layout'), cache = new Map();
  const el = (tag, text, cls) => { const n = document.createElement(tag); if(text !== undefined) n.textContent = text; if(cls) n.className = cls; return n; };
  let current = null, data = null, cards = [], positions = [];
  const stamp = seconds => seconds.toFixed(2) + ' s';
  function indexAt(entries, time) {
    let lo = 0, hi = entries.length;
    while(lo < hi) { const mid = (lo + hi) >>> 1; if(entries[mid].video_time_s <= time + 1e-7) lo = mid + 1; else hi = mid; }
    return lo - 1;
  }
  function seek(time) {
    const id = current.id;
    const go = () => { if(current.id !== id) return; video.currentTime = Math.max(0, Math.min(time, video.duration || time)); sync(); };
    video.pause();
    if(video.readyState >= 1) go(); else { video.addEventListener('loadedmetadata', go, {once:true}); video.load(); }
  }
  function sync() {
    if(!data || panel.hidden) return;
    $('#cot-playhead').textContent = 'Video ' + stamp(video.currentTime);
    data.arms.forEach((arm, a) => {
      const index = indexAt(arm.entries, Math.min(video.currentTime, arm.video_end_s));
      if(index === positions[a]) return;
      positions[a] = index;
      const card = cards[a], entry = arm.entries[index];
      card.output.textContent = entry ? entry.text : 'The first model output begins at ' + stamp(arm.entries[0].video_time_s) + '.';
      card.time.textContent = entry ? 'Model clock ' + stamp(entry.source_time_s) : 'Awaiting model output';
      card.step.textContent = `${index + 1} / ${arm.entries.length} outputs`;
      card.instruction.textContent = entry ? entry.instruction : current.instructions[a];
      card.first.hidden = !!entry;
      if(card.active) card.active.removeAttribute('aria-current');
      card.active = card.rows[index];
      if(card.active) card.active.setAttribute('aria-current', 'true');
    });
  }
  function render() {
    const grid = $('#cot-pair'); grid.replaceChildren(); cards = []; positions = [-2, -2];
    data.arms.forEach((arm, a) => {
      const card = el('article', undefined, 'cot-card'); card.dataset.arm = arm.label;
      const header = el('div', undefined, 'cot-card-heading'), badge = el('b', arm.label), time = el('span');
      header.append(badge, el('strong', 'Instruction ' + arm.label), time);
      const instruction = el('p', undefined, 'cot-active-instruction'), output = el('p', undefined, 'cot-output');
      const footer = el('div', undefined, 'cot-card-footer'), step = el('span'), first = el('button', 'Jump to first output');
      first.type = 'button'; first.addEventListener('click', () => seek(arm.entries[0].video_time_s)); footer.append(step, first);
      const details = el('details', undefined, 'cot-transcript'); details.append(el('summary', `Full CoT transcript · ${arm.entries.length} outputs`));
      const list = el('ol'), rows = [];
      arm.entries.forEach(entry => {
        const li = el('li'), button = el('button'); button.type = 'button';
        button.setAttribute('aria-label', `Seek instruction ${arm.label} CoT at video ${stamp(entry.video_time_s)}`);
        button.append(el('span', stamp(entry.video_time_s), 'cot-row-time'), el('span', entry.text));
        button.addEventListener('click', () => seek(entry.video_time_s)); li.append(button); list.append(li); rows.push(button);
      });
      details.append(list); card.append(header, instruction, output, footer, details); grid.append(card);
      cards.push({time, instruction, output, step, first, rows, active:null});
    });
    $('#cot-clock-note').textContent = data.track === 'carla'
      ? 'Replay clock: video time × ' + data.playback_scale.toFixed(2) + ' + each arm’s model-clock origin. Model timestamps are displayed alongside the recorded text.'
      : 'CoT follows the shared camera clock. Click any transcript entry to seek the paired video.';
    $('#cot-download').href = current.reasoning; $('#cot-download').download = current.id + '-cot.json';
    panel.hidden = false; shell.classList.add('has-cot'); layout.hidden = false; panel.scrollTop = 0; sync();
  }
  async function select(c) {
    current = c; data = null; panel.hidden = true; layout.hidden = true;
    shell.classList.remove('has-cot');
    if(!c.reasoning) return;
    try {
      if(!cache.has(c.reasoning)) cache.set(c.reasoning, fetch(c.reasoning).then(r => { if(!r.ok) throw Error('CoT'); return r.json(); }).catch(error => {cache.delete(c.reasoning); throw error;}));
      const value = await cache.get(c.reasoning);
      if(current.id !== c.id) return;
      data = value; render();
    } catch {
      if(current.id !== c.id) return;
      $('#cot-pair').replaceChildren(el('p', 'Reload this case to load its recorded CoT.')); panel.hidden = false;
    }
  }
  ['timeupdate', 'seeking', 'seeked', 'loadedmetadata', 'ended'].forEach(event => video.addEventListener(event, sync));
  layout.addEventListener('click', () => {
    const expanded = shell.classList.toggle('cot-expanded');
    layout.setAttribute('aria-pressed', String(expanded));
    layout.textContent = expanded ? 'Video + CoT ↔' : 'Expand video ↗';
  });
  window.L2DReasoning = {select};
})();
