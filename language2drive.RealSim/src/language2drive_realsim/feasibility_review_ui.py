"""Same-origin browser form for private paired-evidence reviews."""

REVIEW_SCRIPT = r"""
'use strict';
(async () => {
  const parts = location.pathname.split('/').filter(Boolean);
  const additional = parts.length === 5 && parts[0] === 'additional-feasibility';
  const agent = parts.length === 4 && parts[0] === 'agent-feasibility';
  if (!additional && !agent && (parts.length !== 4 || parts[0] !== 'feasibility')) return;
  const endpoint = (additional ? '/api/additional-feasibility-reviews/' :
    agent ? '/api/agent-feasibility-reviews/' : '/api/feasibility-reviews/') +
    parts.slice(1, additional ? 4 : 3).map(decodeURIComponent).map(encodeURIComponent).join('/');
  const section = document.createElement('section');
  section.innerHTML = `<h2>提交本证据包的人工审核</h2>
    <p>结果保存到独立私有补充审核日志，不改写初审，不自动晋级 PCU。视频不能单独证明连续重建支持，也不等于完整 PCU 源稿终审。轻微且与任务无关的画面瑕疵可接受；影响动作、参照物识别或合法性判断的问题需注明。</p>
    <form><p><label>审核人 <input name="reviewer" required maxlength="100"></label></p>
    <div class="arms"></div>
    <p><label>两臂动作语义确实不同 <select name="semantically_distinct"><option value="null">不确定</option><option value="true">通过</option><option value="false">不通过</option></select></label></p>
    <p><label><input type="checkbox" name="watched_all_camera_videos">我已看完本页两臂四相机的全部视频</label></p>
    <p><label>结论 <select name="decision"><option value="pending">待定：需补证据</option><option value="keep">保留进入后续资格验证</option><option value="reject">拒绝此方案</option></select></label></p>
    <p><label>备注（待定/拒绝必填，注明相机和秒数）<br><textarea name="notes" rows="5" maxlength="4000" style="width:95%"></textarea></label></p>
    <button type="submit" disabled>提交审核结果</button></form><p role="status" aria-live="polite">正在读取审核状态…</p>`;
  document.body.append(section);
  const fields = [['legal_action', '动作合法性'], ['render_quality', '渲染质量'], ['observable', '可观察性']];
  const form = section.querySelector('form'), button = section.querySelector('button'), status = section.querySelector('[role=status]');
  for (const arm of ['R0', 'R1']) {
    const fieldset = document.createElement('fieldset');
    const legend = document.createElement('legend'); legend.textContent = arm; fieldset.append(legend);
    for (const [key, label] of fields) {
      const node = document.createElement('label'); node.textContent = label + ' ';
      const select = document.createElement('select'); select.name = arm + '_' + key;
      for (const [value, text] of [['null','不确定'],['true','通过'],['false','不通过']]) {
        const option = document.createElement('option'); option.value = value; option.textContent = text; select.append(option);
      }
      node.append(select); fieldset.append(node, document.createTextNode('　'));
    }
    form.querySelector('.arms').append(fieldset);
  }
  let snapshot;
  const element = name => form.elements.namedItem(name);
  const tristate = name => JSON.parse(element(name).value);
  try {
    const response = await fetch(endpoint, {cache: 'no-store'});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || '读取失败');
    snapshot = data;
    const saved = data.record;
    if (saved) {
      for (const name of ['reviewer','decision','notes']) element(name).value = saved[name];
      element('semantically_distinct').value = JSON.stringify(saved.semantically_distinct);
      element('watched_all_camera_videos').checked = saved.watched_all_camera_videos;
      for (const arm of ['R0','R1']) for (const [key] of fields) element(arm + '_' + key).value = JSON.stringify(saved.arms[arm][key]);
    }
    status.textContent = data.previous_review_stale ? '证据已变化，旧审核只保留为历史；请重新观看并逐项审核。' :
      saved ? '已载入版本 ' + data.revision + '，可修改后再次提交。' : '尚未提交补充审核。';
    button.disabled = false;
  } catch (error) { status.textContent = '无法读取审核状态：' + error.message; }
  form.addEventListener('submit', async event => {
    event.preventDefault(); if (!snapshot) return;
    button.disabled = true; status.textContent = '正在保存…';
    const payload = {binding: snapshot.binding, expected_revision: snapshot.revision,
      reviewer: element('reviewer').value, decision: element('decision').value, notes: element('notes').value,
      semantically_distinct: tristate('semantically_distinct'),
      watched_all_camera_videos: element('watched_all_camera_videos').checked, arms: {}};
    for (const arm of ['R0','R1']) payload.arms[arm] = Object.fromEntries(fields.map(([key]) => [key, tristate(arm + '_' + key)]));
    try {
      const response = await fetch(endpoint, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
      const data = await response.json(); if (!response.ok) throw new Error(data.error || '提交失败');
      snapshot.revision = data.record.revision;
      status.textContent = '已保存，版本 ' + data.record.revision + '。该结果不是正式资格通过。';
    } catch (error) { status.textContent = '未保存：' + error.message; }
    finally { button.disabled = false; }
  });
})();
"""
