"""Chinese complete-source review UI; no default approval and no bulk approve."""

PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PCU 源稿终审</title><style>
body{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:auto;padding:24px;color:#18202a}
button,select,textarea,input{font:inherit;padding:7px;margin:5px}button{cursor:pointer}
label{display:block;margin:8px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:12px;max-height:65vh;overflow:auto}
.hold,.error{color:#a31919;white-space:pre-wrap}.notice{background:#fff3cf;padding:12px}
video{width:min(100%,540px)}section{border-top:1px solid #ccc;padding-top:12px;margin-top:18px}
#queue{max-height:320px;overflow:auto}textarea{display:block;width:90%;min-height:80px}
</style><h1>30 候选：完整源稿终审</h1>
<p class="notice">本页审核具体 PCU 源稿版本，不继承旧视频审核或口头批量通过。审核通过仍不是正式资格，不会产生 oracle witness。允许轻微非任务画面瑕疵，不放宽交通合法性。</p>
<p>逐个选择候选，核对源稿、地图和下列精确会话的视频，再填写两臂与整体判断。标有强制挂起的源稿只能待定／拒绝，须补证据后重新生成版本。</p>
<div id="status" role="status"></div><div id="queue"></div><main id="detail"></main>
<script src="/source-review.js" defer></script></html>'''

SCRIPT = r'''
'use strict';
const $=id=>document.getElementById(id);
function el(tag,text,parent){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(parent)parent.append(e);return e;}
async function api(url,payload){const response=await fetch(url,payload?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}:{});const body=await response.json();if(!response.ok)throw Error(body.error||response.status);return body;}
function error(e){$('status').textContent=e.message;$('status').className='error';}
const labels={unreviewed:'未审核',keep:'已通过源稿终审',pending:'待定',reject:'拒绝'};
async function queue(){const data=await api('/api/source-reviews');$('queue').replaceChildren();for(const [i,row] of data.sources.entries()){const b=el('button',`${i+1}. ${row.proposal_key.replace('opportunity-','').slice(0,7)} ${row.task_family} · ${labels[row.decision]}${row.hold?' · 强制挂起':''}`,$('queue'));b.disabled=Boolean(row.error);b.title=row.error||row.pcu_id;b.onclick=()=>show(row.pcu_id).catch(error);}}
function tri(parent,label,value){const line=el('label',label,parent);const input=el('select',undefined,line);for(const [v,t] of [['','未确认'],['true','通过'],['false','不通过']]){const option=el('option',t,input);option.value=v;}input.value=value===true?'true':value===false?'false':'';return ()=>input.value===''?null:input.value==='true';}
async function show(id){
 const snap=await api('/api/source-reviews/'+encodeURIComponent(id));const main=$('detail');main.replaceChildren();const old=snap.record||{};
 el('h2',snap.target.proposal_key+' · '+snap.target.task_family,main);
 el('p',`PCU ${id}；当前记录修订 ${snap.revision}。源稿版本绑定：${snap.binding.source.mtime_ns} 纳秒（精确字符串）。`,main);
 if(snap.previous_review_stale)el('p','旧审核已失效，本页不会继承其通过状态。',main).className='hold';
 if(snap.target.mandatory_review_holds.length)el('pre','强制挂起（不能直接通过）：\n'+JSON.stringify(snap.target.mandatory_review_holds,null,2),main).className='hold';
 const source=el('details',undefined,main);el('summary','完整源稿（含双臂指令、合同、路由、范围、原始证据与剩余事项）',source);el('pre',JSON.stringify(snap.source,null,2),source);
 const armText=el('section',undefined,main);el('h3','两种语言选择与对应合同',armText);for(const arm of ['R0','R1']){el('h4',arm,armText);el('pre',JSON.stringify(snap.source.pcu.arms[arm],null,2),armText);}
 const evidence=el('section',undefined,main);el('h3','源稿绑定的地图与数值证据',evidence);el('p','来源证据保留历史修订说明；应以本页完整源稿、当前会话地图及下列实际视频为当前审核对象。',evidence);for(const link of snap.target.evidence_links){const p=el('p',undefined,evidence);const a=el('a',link.label,p);a.href=link.url;a.target='_blank';a.rel='noopener';}
 el('p','下列视频按源稿实际会话 ID 绑定，排除了同一机会下不属于本版本的旧／失败会话。三种子不代表三个独立物理场景。',main);
 for(const packet of snap.target.packets){const box=el('details',undefined,main);el('summary',`种子 ${packet.seed}：${packet.packet}（双臂各四相机）`,box);el('pre',packet.session_ids.join('\n'),box);for(const v of packet.videos){el('p',v.arm+' / '+v.camera,box);const video=el('video',undefined,box);video.controls=true;video.preload='none';video.src=v.url;}}
 const form=el('form',undefined,main);el('h3','填写完整源稿终审',form);const reviewer=el('input',undefined,el('label','审核人',form));reviewer.required=true;reviewer.maxLength=100;reviewer.value=old.reviewer||'';
 const fields={},checks={},arms={};for(const [key,label] of Object.entries(snap.target.review_fields))fields[key]=tri(form,label,old.fields?.[key]);
 for(const [key,label] of Object.entries(snap.checks))checks[key]=tri(form,label,old.checks?.[key]);
 const armLabels={legal_action:'交通合法性',render_quality:'渲染质量',observable:'任务可观察性'};
 for(const arm of ['R0','R1']){arms[arm]={};el('h4',arm+' 独立判断',form);for(const [key,label] of Object.entries(armLabels))arms[arm][key]=tri(form,label,old.arms?.[arm]?.[key]);}
 const decision=el('select',undefined,el('label','最终结论',form));for(const [value,label] of [['pending','待定'],['keep','通过此完整源稿'],['reject','拒绝']]){const option=el('option',label,decision);option.value=value;option.disabled=value==='keep'&&snap.target.mandatory_review_holds.length>0;}decision.value=old.decision||'pending';
 const notes=el('textarea',undefined,el('label','审核依据（必填；请写明地图、相机与秒数，尤其信号／停车／让行）',form));notes.required=true;notes.maxLength=4000;notes.value=old.notes||'';
 const submit=el('button','保存源稿终审结果',form);submit.type='submit';
 form.onsubmit=async event=>{event.preventDefault();submit.disabled=true;const values=group=>Object.fromEntries(Object.entries(group).map(([key,get])=>[key,get()]));try{await api('/api/source-reviews/'+encodeURIComponent(id),{binding:snap.binding,expected_revision:snap.revision,reviewer:reviewer.value,decision:decision.value,notes:notes.value,fields:values(fields),checks:values(checks),arms:Object.fromEntries(Object.entries(arms).map(([arm,group])=>[arm,values(group)]))});await queue();await show(id);$('status').className='';$('status').textContent='已保存到独立源稿审核日志；旧草案和视频审核未修改，正式资格仍未生成。';}catch(e){error(e);submit.disabled=false;}};
}
queue().catch(error);
'''
