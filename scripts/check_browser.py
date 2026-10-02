"""Optional browser check: pip install playwright && playwright install chromium."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
from threading import Thread
from playwright.sync_api import sync_playwright
from pathlib import Path
root=Path(__file__).resolve().parents[1]
server=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=str(root/'docs')))
Thread(target=server.serve_forever,daemon=True).start()
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 for width in (1440,390):
  page=browser.new_page(viewport={'width':width,'height':1000},device_scale_factor=1)
  errors=[]
  page.on('pageerror',lambda error:errors.append(str(error)))
  page.goto(f'http://127.0.0.1:{server.server_port}/',wait_until='networkidle')
  page.wait_for_selector('#results-table tbody tr')
  assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), f'overflow at {width}'
  page.locator('img').evaluate_all('(images)=>images.forEach(i=>i.loading="eager")')
  page.wait_for_function('Array.from(document.images).every(i=>i.complete && i.naturalWidth > 0)')
  assert page.locator('img').evaluate_all('(images)=>images.every(i=>i.complete && i.naturalWidth > 0)')
  for track,count in [('carla',6),('realsim',8),('nuscenes',4),('navsim',5),('wod',4)]:
   page.locator(f'[data-track="{track}"]').click()
   assert page.locator('#results-table tbody tr').count()==count
  page.locator('[data-code="adapter"]').click()
  assert 'def create_adapter' in page.locator('#quick-code').inner_text()
  if width==390:
   page.locator('.menu-toggle').click()
   assert page.locator('.menu-toggle').get_attribute('aria-expanded')=='true'
   page.locator('#nav a').first.click()
   assert page.locator('.menu-toggle').get_attribute('aria-expanded')=='false'
  page.locator('[data-track="carla"]').click()
  page.evaluate('window.scrollTo(0,0)')
  page.screenshot(path=f'/tmp/l2d-page-{width}.png',full_page=True)
  assert not errors, errors
  print('PASS',width,'layout, assets, five result tabs, adapter snippet, navigation')
 browser.close()
server.shutdown()
