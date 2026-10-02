(() => {
  'use strict';
  const $=s=>document.querySelector(s),form=$('#submission-form');let packet=null,markdown='';
  function fail(message){$('#submission-error').textContent=message;$('#submission-error').hidden=false;}
  function invalidate(){packet=null;$('#submission-actions').hidden=true;$('#submission-status').textContent='Changes made. Validate again to update the packet.';}
  form.addEventListener('input',invalidate);
  function prepare(event){
    event.preventDefault();$('#submission-error').hidden=true;$('#submission-actions').hidden=true;packet=null;
    if(!form.reportValidity())return;
    const fields=Object.fromEntries(new FormData(form));
    for(const [key,value] of Object.entries(fields))fields[key]=value.trim();
    if(Object.keys(fields).some(k=>k!=='notes'&&!fields[k])){fail('Complete all required fields.');return;}
    let metrics;
    try{metrics=JSON.parse(fields.metrics);}catch{fail('Native metrics must be valid JSON.');return;}
    if(!metrics||Array.isArray(metrics)||typeof metrics!=='object'||!Object.keys(metrics).length||Object.values(metrics).some(v=>typeof v!=='number'||!Number.isFinite(v))){fail('Use a nonempty JSON object with finite numeric metric values.');return;}
    const coverage={};for(const key of ['scheduled','completed','failed','missing']){coverage[key]=Number(fields[key]);if(!Number.isSafeInteger(coverage[key])||coverage[key]<0){fail('Coverage counts must be nonnegative whole numbers.');return;}}
    if(!coverage.scheduled||coverage.completed+coverage.failed+coverage.missing!==coverage.scheduled){fail('Completed + failed + missing must equal scheduled rollouts.');return;}
    let evidence;try{evidence=new URL(fields.evidence_url);}catch{fail('Provide a valid evidence URL.');return;}
    if(evidence.protocol!=='https:'||evidence.username||evidence.password){fail('Use an HTTPS evidence link without embedded credentials.');return;}
    if(fields.training_regime==='other'&&!fields.notes){fail('Describe the training regime in Notes / deviations.');return;}
    packet={schema:'language2drive.result-submission.v1',created_at:new Date().toISOString(),...fields,metrics,coverage};
    for(const key of Object.keys(coverage))delete packet[key];
    const identity=`Model: ${fields.model}\nCheckpoint: ${fields.checkpoint}\nCheckpoint revision: ${fields.checkpoint_revision}\nAdapter revision: ${fields.adapter_revision}\nTraining regime: ${fields.training_regime}\nTrack: ${fields.track}\nProfile / scorer: ${fields.profile}\nRelease / split: ${fields.release_split}\nSeeds: ${fields.seeds}\nWordings: ${fields.wordings}`;
    const evidenceText=`Native metrics:\n${JSON.stringify(metrics,null,2)}\n\nExecution coverage:\n${JSON.stringify(coverage,null,2)}\n\nEvidence: ${fields.evidence_url}\n\nNotes: ${fields.notes||'None supplied.'}`;
    const reproduce=`Runtime:\n${fields.runtime}\n\nCommands:\n${fields.commands}`;
    markdown=`# Language2Drive result submission\n\n## Model and protocol\n\n${identity}\n\n## Evidence\n\n${evidenceText}\n\n## Reproduction\n\n${reproduce}\n`;
    $('#submission-preview').textContent=markdown;
    const url=new URL('https://github.com/ZihanYou/Language2Drive/issues/new');url.searchParams.set('template','benchmark-result.yml');url.searchParams.set('title',`[Result] ${fields.model} / ${fields.track} / ${fields.profile}`);url.searchParams.set('identity',identity);url.searchParams.set('evidence',evidenceText);url.searchParams.set('reproduce',reproduce);
    let message='Packet format is valid. Download a copy and review the prefilled issue on GitHub before submitting.';
    if(url.href.length>7000){
      url.searchParams.set('identity',`Model: ${fields.model}\nTrack: ${fields.track}\nFull identity and protocol: see attached submission packet.`);
      url.searchParams.set('evidence','Attach the downloaded submission JSON or Markdown. It contains the full metrics, coverage and evidence link.');
      url.searchParams.set('reproduce','See the attached submission packet for full runtime settings and commands.');
      message='The packet exceeds the issue URL limit. Download the full JSON or Markdown and attach it to the GitHub issue; the short draft contains no full result data.';
    }
    $('#github-submission').href=url.href;$('#submission-status').textContent=message;$('#submission-actions').hidden=false;
  }
  form.addEventListener('submit',prepare);
  function save(value,extension,type){if(!packet)return;const u=URL.createObjectURL(new Blob([value],{type})),a=document.createElement('a');a.href=u;a.download=`language2drive-${packet.track}-submission.${extension}`;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
  $('#download-json').addEventListener('click',()=>save(JSON.stringify(packet,null,2)+'\n','json','application/json'));
  $('#download-markdown').addEventListener('click',()=>save(markdown,'md','text/markdown'));
  $('#submission-import').addEventListener('change',async event=>{
    const file=event.target.files[0];if(!file)return;invalidate();$('#submission-error').hidden=true;
    if(file.size>100000){fail('Submission JSON must be smaller than 100 KB.');return;}
    try{
      const data=JSON.parse(await file.text());if(data.schema!=='language2drive.result-submission.v1')throw Error('Expected a Language2Drive submission packet.');
      for(const field of form.elements){if(!field.name)continue;let value=data[field.name];if(['scheduled','completed','failed','missing'].includes(field.name))value=data.coverage?.[field.name];if(field.name==='metrics')value=JSON.stringify(data.metrics,null,2);if(typeof value==='string'||typeof value==='number')field.value=String(value);}
      $('#submission-status').textContent='Saved packet loaded. Review the fields and validate it again.';
    }catch(error){fail('Could not import this packet: '+error.message);}
  });
})();
