"""Exercise the six-map atlas and CARLA target views with a real browser."""
import json
from pathlib import Path

DOCS=Path(__file__).resolve().parents[1]/'docs'

def check_simulator(page,width):
    maps=json.loads((DOCS/'data/maps.json').read_text())['maps']
    page.wait_for_selector('.atlas-card')
    assert page.locator('.atlas-card').count()==6
    for m in maps:
        page.locator(f'.atlas-card[data-map="{m["id"]}"]').click()
        assert page.locator('#atlas-title').inner_text()==m['title']
        assert page.locator('.atlas-pin').count()==len(m['locations'])
        assert page.locator('#atlas-tours button').count()==4
        # Fitted pins remain clickable inside the map on narrow screens.
        bounds=page.locator('#atlas-map').evaluate('''svg=>{
            const s=svg.getBoundingClientRect();
            return [...svg.querySelectorAll('.atlas-pin circle')].map(n=>{
                const r=n.getBoundingClientRect();
                return {left:r.left-s.left,right:r.right-s.right,
                        top:r.top-s.top,bottom:r.bottom-s.bottom,width:r.width};
            });
        }''')
        assert all(b['left']>=-1 and b['right']<=1 and b['top']>=-1
                   and b['bottom']<=1 and b['width']>=27 for b in bounds)
        for i in (0,len(m['locations'])-1):
            p=m['locations'][i]
            page.locator(f'#atlas-places [data-place="{p["id"]}"]').click()
            assert page.locator('#atlas-place-title').inner_text()==p['label']
            assert 'X '+format(p['x'],'.1f') in page.locator('#atlas-place-coordinates').inner_text()
            for view in ('overview','forward','reverse'):
                page.locator(f'[data-atlas-view="{view}"]').click()
                page.wait_for_function('(path)=>document.querySelector("#atlas-photo").src.endsWith(path) && document.querySelector("#atlas-photo").naturalWidth>0',arg=p['views'][view]['path'])
                assert page.locator('#atlas-photo-download').get_attribute('href')==p['views'][view]['path']
        page.locator('#atlas-open-photo').click()
        assert page.locator('#atlas-lightbox').is_visible()
        page.locator('#atlas-photo-next').click()
        assert m['locations'][0]['label'] in page.locator('#atlas-lightbox-title').inner_text()
        page.keyboard.press('Escape')
        assert page.locator('#atlas-lightbox').is_hidden()
        before=page.locator('#atlas-map').get_attribute('viewBox')
        page.locator('#atlas-zoom-in').click()
        assert page.locator('#atlas-map').get_attribute('viewBox')!=before
        page.locator('#atlas-reset').click()
        assert page.locator('#atlas-map').get_attribute('viewBox')==before
        # The first map pin is keyboard-operable as well as clickable.
        page.locator('.atlas-pin').first.focus();page.keyboard.press('Enter')
        assert page.locator('.atlas-pin').first.get_attribute('aria-pressed')=='true'
        page.locator('[data-atlas-mode="tours"]').click()
        for t in m['tours']:
            page.locator(f'[data-tour="{t["id"]}"]').click()
            assert page.locator('#atlas-video').get_attribute('src')==t['media']['path']
            assert page.locator('#atlas-tour-info').inner_text().startswith(t['label'])
        # Stream a full-length tour, seek, and verify the cursor against native coordinates.
        page.locator('#atlas-video').evaluate('(v)=>{v.muted=true;return v.play()}')
        page.wait_for_function('document.querySelector("#atlas-video").currentTime>.1')
        assert page.locator('#atlas-video').evaluate('(v)=>Math.abs(v.duration-45)<.1')
        page.locator('#atlas-video').evaluate('(v)=>{v.pause();v.currentTime=10;}')
        page.wait_for_function('document.querySelector("#atlas-video").readyState>=2 && Math.abs(document.querySelector("#atlas-video").currentTime-10)<.1')
        page.wait_for_timeout(100)
        t=m['tours'][-1];p=next(p for p in t['route'] if p[0]==10)
        pts=[p for l in m['lanes'] for p in l['points']]
        xmin,xmax=min(p[0] for p in pts),max(p[0] for p in pts)
        ymin,ymax=min(p[1] for p in pts),max(p[1] for p in pts)
        scale=min(650/max(1,xmax-xmin),350/max(1,ymax-ymin))
        assert abs(float(page.locator('.atlas-cursor').get_attribute('cx'))-(380+(p[1]-(xmin+xmax)/2)*scale))<.05
        assert abs(float(page.locator('.atlas-cursor').get_attribute('cy'))-(230+(p[2]-(ymin+ymax)/2)*scale))<.05
        page.locator('[data-atlas-mode="places"]').click()
        assert page.locator('#atlas-video').evaluate('(v)=>v.paused')
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.locator('.atlas-card').first.click()
    page.locator('#atlas-case-links button').first.click()
    assert page.locator('#case-role').inner_text()=='Benchmark target'
    assert page.locator('#case-camera').is_visible()
    assert page.locator('#case-reasoning').is_hidden()
    page.locator('[data-target-camera="front"]').click()
    assert page.locator('#case-video').get_attribute('src').endswith('-front.mp4')
    page.locator('#case-video').evaluate('(v)=>{v.muted=true;return v.play()}')
    page.wait_for_function('document.querySelector("#case-video").currentTime>.1')
    page.locator('#case-video').evaluate('(v)=>v.pause()')
    page.locator('[data-target-camera="exterior"]').click()
    assert page.locator('#case-video').get_attribute('src').endswith('-exterior.mp4')
    page.locator('[data-perspective="model-result"]').click()
    assert page.locator('#case-role').inner_text()=='Model result'
    assert page.locator('#case-reasoning').is_visible()
    assert page.locator('#case-camera').is_hidden()
    # Restore the interleaved opening gallery for the existing coverage checks.
    page.locator('#case-reset').click()
    page.add_style_tag(content='.header{visibility:hidden}')
    page.locator('#atlas-shell').screenshot(path=f'/tmp/l2d-showcase/simulator-{width}.png')
    page.add_style_tag(content='.header{visibility:visible}')
    print('PASS',width,'six maps, 54 locations, photo views, lightbox, keyboard, zoom, 24 tours, native camera synchronization and target/model switching')
