"""Check recorded text, seeking, paired clock alignment and case switching."""
import json
from pathlib import Path


def check_reasoning(page, width):
    docs = Path(__file__).resolve().parents[1] / 'docs'
    cases = json.loads((docs / 'data/cases.json').read_text())['cases']
    assert not any(c['track']=='navsim' and c['family']=='branch_choice' for c in cases)
    page.locator('#case-reset').click()
    for track in ['carla', 'realsim']:
        page.locator('#case-track').select_option(track)
        page.locator('#case-model').select_option('Alpamayo 1.5')
        selected = [c for c in cases if c['track']==track and c['model']=='Alpamayo 1.5']
        for case in selected:
            page.locator('#case-search').fill(case['id'])
            page.wait_for_function('(id)=>document.querySelector("#case-video").dataset.caseId===id && !document.querySelector("#case-reasoning").hidden && document.querySelectorAll(".cot-card").length===2',arg=case['id'])
            cot = json.loads((docs / case['reasoning']).read_text())
            page.locator('#case-video').evaluate('(v)=>v.load()')
            page.wait_for_function('document.querySelector("#case-video").readyState>=1')
            t = min(3.5, min(a['video_end_s'] for a in cot['arms']))
            page.locator('#case-video').evaluate('(v,t)=>v.currentTime=t',t)
            page.wait_for_function('(t)=>document.querySelector("#cot-playhead").textContent===`Video ${t.toFixed(2)} s`',arg=t)
            for a, arm in enumerate(cot['arms']):
                entries = [e for e in arm['entries'] if e['video_time_s']<=t+1e-7]
                assert page.locator('.cot-output').nth(a).text_content()==entries[-1]['text']
                assert page.locator('.cot-transcript li').nth(0).is_hidden()
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        # Full transcript exposes original text and seeks to its recorded output.
        arm = cot['arms'][0]
        page.locator('.cot-transcript summary').first.click()
        index = min(3, len(arm['entries'])-1)
        entry = arm['entries'][index]
        page.locator('.cot-transcript').first.locator('button').nth(index).click()
        page.wait_for_function('(t)=>Math.abs(document.querySelector("#case-video").currentTime-t)<.02',arg=entry['video_time_s'])
        assert page.locator('.cot-output').first.text_content()==entry['text']
        with page.expect_download() as info:
            page.locator('#cot-download').click()
        assert json.loads(Path(info.value.path()).read_text())==cot
        page.add_style_tag(content='.header{visibility:hidden}html{scroll-behavior:auto}')
        page.locator('.cot-transcript summary').first.click()
        page.locator('#case-reasoning').screenshot(path=f'/tmp/l2d-showcase/cot-{track}-{width}.png')
        if width>=1100:
            page.locator('.case-evidence-layout').evaluate('(n)=>window.scrollTo(0,window.scrollY+n.getBoundingClientRect().top-140)')
            page.wait_for_function('Math.abs(document.querySelector(".case-evidence-layout").getBoundingClientRect().top-140)<2')
            boxes = page.evaluate('()=>[document.querySelector(".case-media"),document.querySelector("#case-reasoning")].map(n=>{const b=n.getBoundingClientRect();return {x:b.x,y:b.y,width:b.width}})')
            assert boxes[1]['x']>=boxes[0]['x']+boxes[0]['width']-1
            assert abs(boxes[0]['y']-boxes[1]['y'])<1
            page.locator('.case-evidence-layout').screenshot(path=f'/tmp/l2d-showcase/cot-player-{track}-{width}.png')
            page.locator('#case-layout').click()
            assert page.locator('#case-layout').get_attribute('aria-pressed')=='true'
            assert page.locator('#case-video').bounding_box()['width']>boxes[0]['width']
            page.locator('#case-layout').click()
        else:
            assert page.locator('#case-layout').is_hidden()
        page.add_style_tag(content='.header{visibility:visible}')
    # A compressed CARLA replay uses the same model clock as its archived CSV.
    page.locator('#case-track').select_option('carla')
    page.locator('#case-search').fill('carla-formal-d-026')
    page.wait_for_function('document.querySelector("#case-video").dataset.caseId==="carla-formal-d-026" && !document.querySelector("#case-reasoning").hidden')
    page.locator('#case-video').evaluate('(v)=>v.load()')
    page.wait_for_function('document.querySelector("#case-video").readyState>=1')
    page.locator('#case-video').evaluate('(v)=>v.currentTime=20')
    page.wait_for_function('document.querySelector("#cot-playhead").textContent==="Video 20.00 s"')
    case = next(c for c in cases if c['id']=='carla-formal-d-026')
    cot = json.loads((docs / case['reasoning']).read_text())
    assert cot['playback_scale']>4
    for a, arm in enumerate(cot['arms']):
        expected = [e for e in arm['entries'] if e['video_time_s']<=20+1e-7][-1]
        assert page.locator('.cot-output').nth(a).text_content()==expected['text']
    # Switching to a target/open-loop case must remove the preceding model's CoT.
    page.locator('#case-track').select_option('navsim')
    assert page.locator('#case-count').inner_text().startswith('7 clips')
    assert page.locator('#case-reasoning').is_hidden()
    assert not any('Left turn or straight' in text for text in page.locator('.case-card').all_text_contents())
    page.locator('#case-track').select_option('realsim')
    page.locator('#case-kind').select_option('benchmark-target')
    assert page.locator('#case-reasoning').is_hidden()
    page.locator('#case-reset').click()
    print('PASS',width,'28 paired CoT cases, exact text, native clocks, transcript seeking, download and case switching')
