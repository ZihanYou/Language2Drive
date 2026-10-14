'use strict';
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const toast = (message) => { $('.toast').textContent = message; $('.toast').classList.add('visible'); clearTimeout(toast.timer); toast.timer = setTimeout(() => $('.toast').classList.remove('visible'), 2400); };
async function copy(text) { try { await navigator.clipboard.writeText(text); toast('Copied to clipboard'); } catch { toast('Select the code and copy it manually'); } }
$('.menu-toggle').addEventListener('click', () => { const open = !$('#nav').classList.contains('open'); $('#nav').classList.toggle('open', open); $('.menu-toggle').setAttribute('aria-expanded', String(open)); });
$$('#nav a').forEach(a => a.addEventListener('click', () => { $('#nav').classList.remove('open'); $('.menu-toggle').setAttribute('aria-expanded', 'false'); }));
$$('.arm').forEach(button => button.addEventListener('click', () => { $('.experiment').dataset.active = button.dataset.arm; $$('.arm').forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button)); }); }));
const snippets = {
install: `python -m pip install https://github.com/ZihanYou/Language2Drive/releases/download/v0.1.0/language2drive-0.1.0-py3-none-any.whl
language2drive init --workspace ./l2d-workspace
cd l2d-workspace

# Probe official sources; provision selected environments
language2drive setup --tracks openloop

# Optional resource download / simulator provisioning
language2drive setup --tracks all --runtime --download`,
adapter: `from l2d_benchmark import Request

class MyAdapter:
    def predict(self, request: Request):
        # Your model observes request.observation and instruction
        xy = my_model(request.observation, request.instruction)
        # Convert frame and time grid explicitly in your adapter
        return {
            "coordinate_frame": request.output_spec.coordinate_frame,
            "reference_point": request.output_spec.reference_point,
            "timestamps_s": list(request.output_spec.timestamps_s),
            "xy_m": xy,
        }

def create_adapter(config):
    return MyAdapter()`,
evaluate: `# Configure the downloaded resource layout once
language2drive configure --workspace . --gpu 0 \
  --output .l2d/evaluation.json
language2drive gpu-check --config .l2d/evaluation.json

# One adapter; CARLA + RealSim + all three open-loop backends
language2drive run --config .l2d/evaluation.json \
  --adapter examples/my_model.py --model-config model.json \
  --model-python /path/to/model-env/bin/python \
  --tracks carla,realsim,openloop --output outputs/my-model

# Configure your datasets and simulator resources.
# outputs/my-model/summary.json records each track separately.`
};
$$('[data-code]').forEach(button => button.addEventListener('click', () => { $('#quick-code').textContent = snippets[button.dataset.code]; $$('[data-code]').forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button)); }); }));
$('#copy-code').addEventListener('click', () => copy($('#quick-code').textContent));
$('#copy-citation').addEventListener('click', () => copy($('#bibtex').textContent));
let results; let activeTrack = 'carla'; let activeMetric = 'L2D';
function render() {
 const track = results.tracks[activeTrack];
 if (!track.metrics.includes(activeMetric)) activeMetric = track.metrics.includes('L2D') ? 'L2D' : 'SemCFPS';
 $('#results-profile').textContent = track.profile + ' · 0–100 ↑';
 const select = $('#sort-metric'); select.replaceChildren();
 track.metrics.forEach(metric => { const option = document.createElement('option'); option.value = metric; option.textContent = metric; option.selected = metric === activeMetric; select.append(option); });
 const head = $('#results-table thead'); head.replaceChildren(); const header = document.createElement('tr');
 ['System', ...track.metrics].forEach(metric => {
  const th = document.createElement('th'); th.scope = 'col';
  if (metric === 'System') th.textContent = metric;
  else {
   const button = document.createElement('button'); button.type = 'button'; button.className = 'metric-header';
   button.textContent = metric === activeMetric ? metric + ' ↓' : metric;
   button.setAttribute('aria-label', 'Explain ' + metric); button.setAttribute('aria-controls', 'metric-explorer');
   button.addEventListener('click', () => window.L2DMetrics.select(metric, true)); th.append(button);
  }
  header.append(th);
 }); head.append(header);
 const body = $('#results-table tbody'); body.replaceChildren();
 const rows = [...track.rows].sort((a,b) => b[activeMetric] - a[activeMetric] || a.model.localeCompare(b.model));
 rows.forEach((row, index) => { const tr = document.createElement('tr'); const name = document.createElement('td'); const rank = document.createElement('span'); rank.className = 'rank'; rank.textContent = String(index+1).padStart(2,'0'); name.append(rank, window.L2DModels.link(row.model)); tr.append(name);
 track.metrics.forEach(metric => { const td = document.createElement('td'); td.textContent = row[metric].toFixed(activeTrack === 'navsim' ? 3 : 2); const best = Math.max(...rows.map(r => r[metric])); if (row[metric] === best && best > 0) td.classList.add('best'); if (metric === activeMetric) td.classList.add('sort-column'); tr.append(td); }); body.append(tr); });
 window.L2DMetrics.update(activeTrack, track);
 $('#results-note').textContent = activeTrack === 'navsim' ? 'Paper evaluation profile · five driving models · instruction-conditioned trajectory prediction.' : activeTrack === 'carla' ? 'Six driving systems · four instruction scopes · paired closed-loop evaluation.' : 'Training-free evaluation · native driving metrics and paired instruction-following scores.';
 $('#results-panel').setAttribute('aria-labelledby', 'tab-' + activeTrack);
 $$('.results-tabs button').forEach(b => { const selected = b.dataset.track === activeTrack; b.setAttribute('aria-selected', String(selected)); b.tabIndex = selected ? 0 : -1; });
}
$$('.results-tabs button').forEach((button, index, buttons) => { button.addEventListener('click', () => { if (!results) return; activeTrack = button.dataset.track; render(); }); button.addEventListener('keydown', event => { let next; if(event.key === 'ArrowRight') next=(index+1)%buttons.length; if(event.key === 'ArrowLeft') next=(index+buttons.length-1)%buttons.length; if(event.key === 'Home') next=0; if(event.key === 'End') next=buttons.length-1; if(next !== undefined) {event.preventDefault(); buttons[next].focus(); buttons[next].click();} }); });
$('#sort-metric').addEventListener('change', event => { activeMetric = event.target.value; render(); window.L2DMetrics.select(activeMetric); });
Promise.all([window.L2DModels.ready, fetch('data/results.json').then(response => { if(!response.ok) throw new Error('Result data unavailable'); return response.json(); })]).then(([,data]) => { results=data; render(); }).catch(() => { $('#results-profile').textContent='Result data could not load. Read the tables in the repository README.'; });
