/* One verified paper registry shared by the leaderboard and case library. */
(() => {
  'use strict';
  let catalog = {};
  const ready = fetch('data/models.json').then(response => {
    if (!response.ok) throw new Error('Model references unavailable');
    return response.json();
  }).then(data => {
    Object.entries(data.models).forEach(([name, model]) => {
      [name, ...(model.aliases || [])].forEach(alias => { catalog[alias] = model; });
    });
  }).catch(() => {}); // The table remains readable if a reference file fails to load.
  function link(name, label = name) {
    const model = catalog[name];
    if (!model) return document.createTextNode(label);
    const a = document.createElement('a');
    a.href = model.paper; a.textContent = label; a.className = 'model-paper';
    a.target = '_blank'; a.rel = 'noopener noreferrer';
    a.title = `${model.paper_label || 'Paper'}: ${model.title}`;
    a.setAttribute('aria-label', `${label} — ${a.title} (opens in a new tab)`);
    return a;
  }
  function references(container, names) {
    container.replaceChildren();
    [...new Set(names)].forEach(name => {
      const model = catalog[name];
      if (!model) return;
      const item = document.createElement('span');
      item.append(link(name, `${name} · ${model.paper_label || 'Paper'} ↗`));
      if (model.release) {
        const release = document.createElement('a'); release.href = model.release;
        release.textContent = 'Model card ↗'; release.target = '_blank'; release.rel = 'noopener noreferrer';
        item.append(release);
      }
      container.append(item);
    });
    container.hidden = !container.childElementCount;
  }
  window.L2DModels = {ready, link, references};
})();
