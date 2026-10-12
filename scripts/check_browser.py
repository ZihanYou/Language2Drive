"""Optional browser check: pip install playwright && playwright install chromium."""
from http.server import ThreadingHTTPServer
from serve_preview import PreviewHandler
from functools import partial
from threading import Thread
from playwright.sync_api import sync_playwright
from pathlib import Path
from check_metrics_browser import check_metrics
from check_reasoning_browser import check_reasoning
root=Path(__file__).resolve().parents[1]
Path('/tmp/l2d-showcase').mkdir(exist_ok=True)
server=ThreadingHTTPServer(('127.0.0.1',0),partial(PreviewHandler,directory=str(root/'docs')))
Thread(target=server.serve_forever,daemon=True).start()
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 for width in (1440,390):
  page=browser.new_page(viewport={'width':width,'height':1000},device_scale_factor=1)
  errors=[]
  page.on('pageerror',lambda error:errors.append(str(error)))
  page.goto(f'http://127.0.0.1:{server.server_port}/',wait_until='networkidle')
  page.wait_for_selector('#results-table tbody tr')
  page.wait_for_selector('.case-card')
  assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), f'overflow at {width}'
  page.locator('img').evaluate_all('(images)=>images.forEach(i=>i.loading="eager")')
  page.wait_for_function('Array.from(document.images).every(i=>i.complete && i.naturalWidth > 0)')
  assert page.locator('img').evaluate_all('(images)=>images.every(i=>i.complete && i.naturalWidth > 0)')
  for track,count in [('carla',6),('realsim',8),('nuscenes',4),('navsim',5),('wod',4)]:
   page.locator(f'[data-track="{track}"]').click()
   assert page.locator('#results-table tbody tr').count()==count
  check_metrics(page, width)
  check_reasoning(page, width)
  page.locator('[data-code="adapter"]').click()
  assert 'def create_adapter' in page.locator('#quick-code').inner_text()
  if width==1440:
   page.locator('[data-film-time="104"]').click()
   page.wait_for_function('document.querySelector("#intro-film").currentTime >= 104 && document.querySelector("#intro-film").readyState >= 2')
   assert page.locator('#intro-film').evaluate('(v)=>Math.abs(v.duration-120)<.1')
   assert page.locator('#case-video').evaluate('(v)=>v.paused')
   page.locator('#intro-film').evaluate('(v)=>v.pause()')
  # The first gallery screen must expose all five backends.
  first_cards=page.locator('.case-card-body').all_text_contents()
  for name in ['CARLA-L2D','RealSim','nuScenes','NAVSIM','WOD-E2E']:
   assert any(name in text for text in first_cards), name
  page.locator('#case-track').select_option('nuscenes')
  assert page.locator('#case-paper-links .model-paper').count() == 1
  assert page.locator('#case-variant-wrap').is_visible()
  page.locator('#case-variant').select_option(index=1)
  assert 'nuscenes-' in page.url
  page.locator('#case-video').evaluate('(v)=>{v.muted=true;return v.play()}')
  page.wait_for_function('document.querySelector("#case-video").currentTime > 0.1')
  page.locator('#case-speed').select_option('1.5')
  assert page.locator('#case-video').evaluate('(v)=>v.playbackRate')==1.5
  page.locator('#case-video').evaluate('(v)=>v.pause()')
  page.locator('#case-track').select_option('navsim')
  assert page.locator('#trajectory-inspector').is_visible()
  assert page.locator('#trajectory-model option').count()==5
  assert page.locator('#case-paper-links .model-paper').count()==5
  assert page.locator('#trajectory-paper .model-paper').count()==1
  old=page.locator('#trajectory-plot polyline').nth(1).get_attribute('points')
  page.locator('#trajectory-model').select_option(index=1)
  assert page.locator('#trajectory-paper .model-paper').inner_text().startswith(page.locator('#trajectory-model').input_value())
  assert page.locator('#trajectory-plot polyline').nth(1).get_attribute('points')!=old
  page.locator('#trajectory-time').fill('4');page.locator('#trajectory-time').dispatch_event('input')
  assert page.locator('#trajectory-time-label').inner_text()=='2.0 s'
  if width==1440:
   page.add_style_tag(content='.header{visibility:hidden}')
   page.locator('#case-viewer').screenshot(path='/tmp/l2d-showcase/gallery-desktop.png')
   page.add_style_tag(content='.header{visibility:visible}')
  # Role labels must identify the displayed video, independent of the model inspector.
  assert page.locator('#case-role').inner_text()=='Benchmark target'
  page.locator('#case-kind').select_option('benchmark-target')
  page.locator('#case-search').fill('no-such-case-123')
  # A backend entry resets stale constraints, including those from reference-only NAVSIM.
  page.locator('.case-backends [data-case-track="carla"]').click()
  assert page.locator('#case-count').inner_text().startswith('20 clips')
  assert page.locator('#case-role').inner_text()=='Model result'
  assert page.locator('#case-video').get_attribute('src').startswith('assets/cases/carla-')
  assert page.locator('#case-kind').input_value()=='all'
  assert page.locator('#case-search').input_value()==''
  page.locator('#case-track').select_option('realsim')
  page.locator('#case-kind').select_option('benchmark-target')
  assert page.locator('#case-count').inner_text().startswith('6 clips')
  assert page.locator('#case-role').inner_text()=='Benchmark target'
  page.locator('[data-perspective="model-result"]').click()
  assert page.locator('#case-role').inner_text()=='Model result'
  assert page.locator('.construction-figure').evaluate('(n)=>getComputedStyle(n).backgroundColor')=='rgb(255, 255, 255)'
  page.wait_for_selector('.scope-map')
  for scope,pcu in [('A','A-001'),('B','B-009'),('C','C-025'),('D','D-001')]:
   page.locator(f'[data-scope="{scope}"]').click()
   assert pcu in page.locator('#scope-case-name').inner_text()
   assert page.locator('.scope-map .scope-route').count()==2
   assert page.locator('#scope-watch').get_attribute('href').endswith(pcu.lower()+'#examples')
  page.locator('#scope-progress').fill('25')
  assert page.locator('#scope-progress-label').inner_text()=='25%'
  before=page.locator('.scope-car').first.get_attribute('transform')
  page.locator('#scope-progress').fill('75')
  assert before!=page.locator('.scope-car').first.get_attribute('transform')
  page.locator('#scope-play').click()
  page.wait_for_function('document.querySelector("#scope-progress").value > 75')
  page.locator('#scope-play').click()
  page.locator('#scope-watch').click()
  assert page.locator('#case-count').inner_text().startswith('1 clip')
  assert 'carla-formal-d-001' in page.url
  page.locator('#case-search').fill('no-such-case-123')
  assert page.locator('#case-empty').is_visible()
  assert page.locator('#case-viewer').is_hidden()
  page.locator('#case-reset').click()
  page.locator('.framework-steps [data-step="1"]').click()
  assert page.locator('#framework').get_attribute('data-step')=='1'
  page.locator('[data-framework-arm="b"]').click()
  assert page.locator('[data-framework-arm="b"]').get_attribute('aria-pressed')=='true'
  page.locator('[data-scope="D"]').click()
  assert 'Long-horizon' in page.locator('#framework-title').inner_text()
  page.add_style_tag(content='.header{visibility:hidden}')
  page.locator('#framework').screenshot(path=f'/tmp/l2d-showcase/framework-{width}.png')
  page.add_style_tag(content='.header{visibility:visible}')
  if width==390:
   page.locator('.menu-toggle').click()
   assert page.locator('.menu-toggle').get_attribute('aria-expanded')=='true'
   page.locator('#nav a').first.click()
   assert page.locator('.menu-toggle').get_attribute('aria-expanded')=='false'
  page.locator('[data-track="carla"]').click()
  page.evaluate('window.scrollTo(0,0)')
  page.screenshot(path=f'/tmp/l2d-page-{width}.png',full_page=True)
  assert not errors, errors
  print('PASS',width,'layout, assets, results, gallery, video playback, trajectories, framework, navigation')
  page.goto(f'http://127.0.0.1:{server.server_port}/submit.html')
  assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
  fields={'model':'Browser QA model','checkpoint':'test/model','checkpoint_revision':'abc123','adapter_revision':'def456','profile':'corrected-v1','release_split':'qa/dev','seeds':'1','wordings':'canonical','runtime':'QA only; no benchmark result','metrics':'{"CFPS": 1.25}','scheduled':'10','completed':'8','failed':'1','missing':'0','evidence_url':'https://example.org/evidence','commands':'language2drive evaluate --help'}
  for key,value in fields.items():page.locator(f'[name="{key}"]').fill(value)
  page.locator('button[type="submit"]').click()
  assert 'must equal' in page.locator('#submission-error').inner_text()
  assert page.locator('#submission-actions').is_hidden()
  page.locator('[name="missing"]').fill('1');page.locator('button[type="submit"]').click()
  assert page.locator('#submission-actions').is_visible()
  from urllib.parse import urlparse,parse_qs
  url=page.locator('#github-submission').get_attribute('href');query=parse_qs(urlparse(url).query)
  assert query['template']==['benchmark-result.yml']
  assert all(name in query for name in ['identity','evidence','reproduce'])
  with page.expect_download() as info:page.locator('#download-json').click()
  import json
  payload=json.loads(Path(info.value.path()).read_text())
  assert payload['coverage']=={'scheduled':10,'completed':8,'failed':1,'missing':1}
  page.locator('[name="model"]').fill('Edited model')
  assert page.locator('#submission-actions').is_hidden()
  page.locator('#submission-import').set_input_files({'name':'saved.json','mimeType':'application/json','buffer':json.dumps(payload).encode()})
  page.wait_for_function('document.querySelector("[name=model]").value === "Browser QA model"')
  page.locator('button[type="submit"]').click()
  if width==1440:
   page.locator('[name="commands"]').fill('QA command '+('large result evidence '*400))
   page.locator('button[type="submit"]').click()
   assert 'exceeds the issue URL limit' in page.locator('#submission-status').inner_text()
   assert len(page.locator('#github-submission').get_attribute('href'))<7000
   page.locator('[name="commands"]').fill(fields['commands']);page.locator('button[type="submit"]').click()
  page.screenshot(path=f'/tmp/l2d-showcase/submission-{width}.png',full_page=True)
  assert not errors,errors
  print('PASS',width,'submission accounting, draft URL, download, edit invalidation, JSON import')
 # Verify seekable media and reduced-motion behavior, without publishing anything.
 import urllib.request
 data=json.loads((root/'docs/data/cases.json').read_text())
 request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/'+data['cases'][0]['video'],headers={'Range':'bytes=100-199'})
 with urllib.request.urlopen(request) as response:
  assert response.status==206 and len(response.read())==100
 page=browser.new_page(reduced_motion='reduce')
 page.goto(f'http://127.0.0.1:{server.server_port}/')
 page.locator('#framework-replay').click()
 page.wait_for_function('Number(document.querySelector("#framework-dot-b").getAttribute("cy")) === 14')
 page.locator('#scope-play').click()
 assert page.locator('#scope-progress').input_value()=='100'
 page.locator('[data-explain="SCS"]').click()
 assert page.locator('.mx-credit').first.evaluate('(n)=>getComputedStyle(n).transitionDuration')=='0s'
 print('PASS range requests and reduced-motion trajectory rendering')
 browser.close()
server.shutdown()
