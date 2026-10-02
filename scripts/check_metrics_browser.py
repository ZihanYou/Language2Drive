"""Browser assertions for metric semantics, track switching and paper entry points.

Called by check_browser.py with the existing live preview and browser page.
"""
import json
from pathlib import Path


def check_metrics(page, width):
    results = json.loads((Path(__file__).resolve().parents[1] / 'docs/data/results.json').read_text())

    def select(metric):
        page.locator(f'[data-explain="{metric}"]').click()
        assert page.locator('#metric-explorer').get_attribute('data-metric') == metric

    def output(index=0):
        return float(page.locator(f'#mx-score-{index}').inner_text())

    def slider(name, value):
        page.locator(f'#mx-{name}').fill(str(value))
        page.locator(f'#mx-{name}').dispatch_event('input')

    for track, table in results['tracks'].items():
        page.locator(f'[data-track="{track}"]').click()
        assert page.locator('#results-table tbody .model-paper').count() == len(table['rows'])
        assert page.locator('#results-table tbody .model-paper').evaluate_all(
            '(links)=>links.every(a=>a.href.startsWith("https://arxiv.org/abs/") && a.rel.includes("noopener"))')
        assert page.locator('[data-explain]').count() == len(table['metrics'])
        for metric in table['metrics']:
            select(metric)
            assert page.locator('#mx-title').inner_text()
            assert page.locator('#mx-formula').inner_text()
            assert page.evaluate('''() => {
                const ids = [...document.querySelectorAll('[id]')].map(n => n.id);
                return ids.length === new Set(ids).size;
            }'''), (track, metric, 'duplicate id')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width, track, metric)
        # Every table heading offers a direct, accessible explanation entry point.
        page.get_by_role('button', name='Explain PCS', exact=True).click()
        assert page.locator('#metric-explorer').get_attribute('data-metric') == 'PCS'
        assert page.locator('#mx-title').evaluate('(node)=>document.activeElement===node')
        slider('a', 100); slider('b', 40)
        assert (output(), output(1)) == (70, 40)
        page.get_by_role('button', name='Balanced · 70 / 70', exact=True).click()
        assert (output(), output(1)) == (70, 70)
        slider('a', 0); slider('b', 0)
        assert (output(), output(1)) == (0, 0)

        if track in ('carla', 'realsim'):
            for metric in ['SISR', 'CFPS', 'SemISR', 'Sem-CFPS']:
                select(metric)
                assert (output(), output(1)) == (50, 0)
                page.locator('#mx-b').check()
                assert (output(), output(1)) == (100, 100)
                page.locator('#mx-a').uncheck(); page.locator('#mx-b').uncheck()
                assert (output(), output(1)) == (0, 0)
            select('DrivingScore'); slider('progress', 75)
            assert output() == 75
            select('L2D')
            before = page.locator('#results-table tbody').inner_text()
            for i, row in enumerate(table['rows']):
                page.locator('#mx-model').select_option(str(i))
                assert abs(output() - row['L2D']) <= .011, (track, row['model'], output(), row['L2D'])
            page.get_by_role('button', name='Balanced', exact=True).click()
            balanced = output()
            page.get_by_role('button', name='Uneven', exact=True).click()
            assert output() < balanced
            assert 'Language-family mean: 70.00' in page.locator('#mx-insight').inner_text()
            page.get_by_role('button', name='All 100', exact=True).click()
            assert output() == 100
            assert page.locator('#mx-driving-mean').inner_text() == '100.00'
            slider('safety', 0); slider('driving', 0)
            assert output() == 0
            slider('safety', 100); slider('driving', 100)
            for family in ['strict', 'semantic', 'graded']: slider(family, 0)
            assert output() == 1
            assert page.locator('#results-table tbody').inner_text() == before
            if track == 'carla':
                select('PSM')
                for ttc, margin in [(0,0),(1.5,50),(3,100),(6,100)]:
                    slider('ttc', ttc); assert output() == margin
            else:
                select('Safety')
                assert output() == 100
                page.locator('#mx-collision').uncheck(); assert output() == 0
                page.locator('#mx-collision').check(); page.locator('#mx-road').uncheck()
                assert output() == 0
        else:
            for metric in ['SemSISR', 'SemCFPS']:
                select(metric)
                assert (output(), output(1)) == (50, 0)
                page.locator('#mx-b-opposite').uncheck()
                assert (output(), output(1)) == (100, 100)
                page.locator('#mx-a-own').uncheck()
                assert (output(), output(1)) == (50, 0)
                page.locator('#mx-b-opposite').check()
                assert (output(), output(1)) == (0, 0)
    page.locator('[data-track="carla"]').click()
    select('SISR')
    # Native checkbox and slider keyboard interaction.
    page.locator('#mx-b').focus(); page.keyboard.press('Space')
    assert output(1) == 100
    select('PCS'); page.locator('#mx-b').focus(); page.keyboard.press('ArrowRight')
    assert output(1) == 41
    select('SISR')
    page.add_style_tag(content='html{scroll-behavior:auto !important}.header{visibility:hidden}.skip-link{visibility:hidden}')
    page.locator('#metric-explorer').screenshot(path=f'/tmp/l2d-showcase/metrics-{width}.png')
    select('L2D'); page.locator('#mx-model').select_option('0')
    page.locator('#metric-explorer').screenshot(path=f'/tmp/l2d-showcase/l2d-calculator-{width}.png')
    page.add_style_tag(content='.header{visibility:visible}')
    print('PASS', width, 'all metric definitions, paired verdicts, graded credit, driving, L2D, keyboard and paper links')
