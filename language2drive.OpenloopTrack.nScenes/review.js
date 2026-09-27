(() => {
  'use strict';
  const labels = {
    observable: '相机视角能观察关键场景',
    arm_0_feasible: '臂 0 可行',
    arm_1_feasible: '臂 1 可行',
    mutually_exclusive: '两臂差异清楚且互斥',
    traffic_clearance: '交通参与者没有明显冲突',
    map_binding: '地图与相机道路连接关系一致'
  };
  const decisions = {approve:'批准', reject:'拒绝', needs_more_evidence:'需更多证据'};
  const create = (tag, cls, content) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (content !== undefined) node.textContent = content;
    return node;
  };
  const option = (value, label) => {
    const node = create('option', '', label); node.value = value; return node;
  };
  const getJSON = async url => {
    const response = await fetch(url, {cache:'no-store'});
    if (!response.ok) throw new Error('读取审核数据失败：HTTP ' + response.status);
    return response.json();
  };
  const toolbar = create('div', 'review-toolbar');
  const reviewerLabel = create('label', '', '审核人 ');
  const reviewer = create('input');
  reviewer.placeholder = '审核人代号'; reviewer.maxLength = 80;
  reviewer.value = localStorage.getItem('nuscenes-reviewer') || '';
  reviewer.addEventListener('input', () => localStorage.setItem('nuscenes-reviewer', reviewer.value));
  reviewerLabel.append(reviewer);
  const summary = create('span', '', '读取审核记录中…');
  const exportLink = create('a', '', '下载审核表 CSV');
  exportLink.href = '/api/export.csv';
  toolbar.append(reviewerLabel, summary, exportLink);
  document.querySelector('h1').after(toolbar);

  const mark = (article, review, status) => {
    article.dataset.decision = review ? review.decision : '';
    status.textContent = review
      ? `已提交：${decisions[review.decision]} · ${review.reviewer} · ${review.submitted_at_utc}`
      : '尚未提交';
    status.classList.remove('review-error');
  };

  getJSON('/api/config').then(config => {
    const byId = new Map(config.cases.map(row => [row.pcu_id, row]));
    const latest = config.latest;
    const refreshSummary = () => {
      const submitted = Object.values(latest);
      const count = type => submitted.filter(row => row.decision === type).length;
      summary.textContent = `共 ${config.cases.length} 例 · 已提交 ${submitted.length} · 批准 ${count('approve')} · 拒绝 ${count('reject')} · 待补证 ${count('needs_more_evidence')}`;
    };
    refreshSummary();
    for (const article of document.querySelectorAll('article')) {
      const pcuId = article.querySelector('h2')?.textContent.trim();
      const row = byId.get(pcuId);
      if (!row) continue;
      for (const img of article.querySelectorAll('img')) img.loading = 'lazy';
      const form = create('form', 'review-form');
      const hint = create('div', '', `${row.family} · ${row.location} · ${row.auto_status}${row.auto_status === 'needs_human_review' ? '' : '（不可批准）'}`);
      const arm0 = create('div', '', `臂 0：${row.arm_0_instruction}`);
      const arm1 = create('div', '', `臂 1：${row.arm_1_instruction}`);
      const grid = create('div', 'review-criteria');
      const criteria = {};
      for (const [key, label] of Object.entries(labels)) {
        const wrapper = create('label', '', label);
        const select = create('select'); select.required = true;
        select.append(option('', '请选择'), option('yes', '是'), option('no', '否'), option('unknown', '不确定'));
        wrapper.append(select); grid.append(wrapper); criteria[key] = select;
      }
      const decisionLabel = create('label', '', '审核结论 ');
      const decision = create('select'); decision.required = true;
      decision.append(option('', '请选择'), option('approve', '批准'), option('reject', '拒绝'), option('needs_more_evidence', '需更多证据'));
      if (row.auto_status !== 'needs_human_review') decision.querySelector('[value="approve"]').disabled = true;
      decisionLabel.append(decision);
      const notesLabel = create('label', '', '审核依据（至少 10 字）');
      const notes = create('textarea'); notes.required = true; notes.minLength = 10; notes.maxLength = 2000;
      notes.placeholder = '记录可观察性、两臂可行性、交通参与者、地图匹配及需复查的证据';
      notesLabel.append(notes);
      const actions = create('div', 'review-actions');
      const submit = create('button', '', '提交本例审核'); submit.type = 'submit';
      const status = create('span', 'review-status'); mark(article, latest[pcuId], status);
      const historyButton = create('button', '', '查看历史'); historyButton.type = 'button';
      const history = create('div', 'review-history'); history.hidden = true;
      historyButton.addEventListener('click', async () => {
        try {
          const result = await getJSON('/api/history?pcu_id=' + encodeURIComponent(pcuId));
          history.textContent = result.history.length
            ? result.history.map(entry => `#${entry.id} ${entry.submitted_at_utc} ${entry.reviewer} ${decisions[entry.decision]}\n${entry.notes}`).join('\n\n')
            : '暂无历史记录';
          history.hidden = !history.hidden;
        } catch (error) { history.textContent = error.message; history.hidden = false; }
      });
      actions.append(submit, historyButton, status);
      form.append(hint, arm0, arm1, grid, decisionLabel, notesLabel, actions, history);
      const prior = latest[pcuId];
      if (prior) {
        for (const key of Object.keys(labels)) criteria[key].value = prior[key];
        decision.value = prior.decision; notes.value = prior.notes;
      }
      form.addEventListener('submit', async event => {
        event.preventDefault();
        if (!reviewer.value.trim()) { reviewer.focus(); status.textContent = '请先填写审核人'; status.classList.add('review-error'); return; }
        const payload = {pcu_id:pcuId, reviewer:reviewer.value.trim(), decision:decision.value, notes:notes.value.trim()};
        for (const [key, select] of Object.entries(criteria)) payload[key] = select.value;
        submit.disabled = true; status.textContent = '提交中…';
        try {
          const response = await fetch('/api/reviews', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
          const result = await response.json();
          if (!response.ok) throw new Error(result.error || '提交失败');
          latest[pcuId] = result.review; mark(article, result.review, status); refreshSummary();
        } catch (error) { status.textContent = error.message; status.classList.add('review-error'); }
        finally { submit.disabled = false; }
      });
      article.querySelector('p').after(form);
    }
  }).catch(error => { summary.textContent = error.message; summary.classList.add('review-error'); });
})();
