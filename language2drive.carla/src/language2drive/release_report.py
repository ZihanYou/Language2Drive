"""Render a concise benchmark report from the recorded evaluation scope."""
from pathlib import Path
from .io import atomic_json, atomic_text, load_json


def render(score_path, output, *, project_root=None, json_output=None):
    value = load_json(score_path)
    if value.get('schema_version') != 'language2drive.release_score.v1':
        raise ValueError('Run l2d score --plan PLAN before generating a release report.')
    m = value['metrics']
    lines = ['# Language2Drive CARLA evaluation\n',
             f"Model: `{value['model']}` · Track: `{value['track']}` · Workload: `{value['workload']}`\n",
             f"Plan SHA-256: `{value['plan_sha256']}`\n",
             f"Expected runs: **{m.get('expected_runs', len(value['scope']['run_ids']))}**\n"]
    def fmt(x):
        return 'N/A' if x is None else f'{100 * x:.2f}%'
    if value['workload'] == 'core':
        lines += [f"Valid results: **{m.get('valid_result_runs', 0)}**\n",
                  '| Layer | SISR | SemISR | CFPS | Sem-CFPS |\n|---|---:|---:|---:|---:|\n']
        for layer in 'ABCD':
            vals = [m.get(k, {}).get(layer) for k in ('SISR_by_layer', 'SemISR_by_layer', 'CFPS', 'Sem-CFPS')]
            lines.append('| ' + layer + ' | ' + ' | '.join(map(fmt, vals)) + ' |\n')
        lines.append('\nRates use every expected run or pair in the saved scope. Missing and invalid results remain in the denominator.\n')
    else:
        lines.append('\nThe JSON report contains the workload-specific metrics and per-run records.\n')
    atomic_text(Path(output), '\n'.join(lines))
    if json_output:
        atomic_json(Path(json_output), value)
    return value
