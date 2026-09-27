"""Export a portable benchmark bundle; never copy machine credentials or scenes."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from language2drive_realsim.compiler import compile_source
from language2drive_realsim.runtime_client import build_run_plan


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def portable_cohort(source, authoring_root):
    """Keep scoring annotations intact; rebase only known authoring references.

    These references are historical review records, not execution inputs. They
    are explicitly marked as external instead of pretending those files ship.
    """
    cohort = {k: deepcopy(source[k]) for k in ('schema_version', 'pcus', 'release_profile')}
    prefix = str(authoring_root.resolve()) + '/'

    def rebase(value):
        if isinstance(value, list):
            return [rebase(item) for item in value]
        if isinstance(value, str) and value.startswith(prefix):
            return '${AUTHORING_ROOT}/' + value[len(prefix):]
        return value

    # These declarations are part of validation, not optional presentation
    # metadata. Preserve the approved allocation, rationale and limitations.
    if 'allocation_evidence_refs' in cohort['release_profile']:
        cohort['release_profile']['allocation_evidence_refs'] = rebase(
            cohort['release_profile']['allocation_evidence_refs'])

    fields = [('render_validity_protocol', 'pilot_evidence_refs'),
              ('trigger_referent', 'frame_refs'), ('shared_lane_scope_basis', 'source_scope')]
    for pcu in cohort['pcus']:
        annotations = pcu.get('private_annotations', {})
        for section, key in fields:
            if key in annotations.get(section, {}):
                annotations[section][key] = rebase(annotations[section][key])
    return cohort


def export(source_path, template, driver_paths, controller, output):
    source = json.loads(source_path.read_text())
    # Runtime templates are stored as AUTHORING_ROOT/.runtime/CAMPAIGN.
    cohort = portable_cohort(source, template.resolve().parent.parent)
    if len(cohort['pcus']) != 30:
        raise ValueError('expected the frozen 30-PCU cohort')
    text = json.dumps(cohort)
    if re.search(r'(?:/(?:NEW_EDS|home|root)/|hf_[A-Za-z0-9]{20,}|10\.0\.0\.)', text):
        raise ValueError('cohort contains machine-local references; inspect before export')
    public, sidecar, audit = compile_source(cohort)
    # Keep the original three-seed source intact. Select the paper's first
    # expression / seed in the compiled execution view only.
    sidecar['sessions'] = {k:v for k,v in sidecar['sessions'].items() if v['seed'] == 1701}
    public['runs'] = [r for r in public['runs'] if r['session_uuid'] in sidecar['sessions']]
    audit['pairs'] = [row for row in audit['pairs'] if row['seed'] == 1701]
    audit['pass'] = len(audit['pairs']) == 30 and all(row['pass'] for row in audit['pairs'])
    plan = build_run_plan(public, sidecar)
    if plan['run_count'] != 60:
        raise ValueError('expected 60 paired slots')
    groups, covered = [], set()
    for group in sorted(template.glob('group-*')):
        old_plan = json.loads((group / 'run_plan.private.json').read_text())
        ids = sorted({r['pcu_id'] for r in old_plan['runs']})
        if covered.intersection(ids):
            raise ValueError('duplicate PCUs in execution groups')
        covered.update(ids)
        user = json.loads((group / 'user-config.yaml').read_text())
        user['scene_provider']['usdz']['data_dir'] = '${SCENES}'
        user['simulation_config']['force_gt_frame_cache']['cache_dir'] = '${OUTPUT}/frame-cache'
        # Keep actual model experiment rig (1080p), not source preview sizes.
        groups.append(dict(id=group.name, pcu_ids=ids, user_config=user))
    if covered != {p['pcu_id'] for p in cohort['pcus']}:
        raise ValueError('runtime groups do not cover exactly 30 PCUs')
    scene_ids = sorted({p['scene']['scene_id_private'].removeprefix('clipgt-') for p in cohort['pcus']})
    scene_files = [f'sample_set/26.01_release/{s}/{s}.usdz' for s in scene_ids]
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'pcus.json', cohort)
    write(output / 'compiled/public_workload.json', public)
    write(output / 'compiled/sidecar.json', sidecar)
    write(output / 'compiled/run_plan.json', plan)
    write(output / 'compiled/leakage_audit.json', audit)
    # The model service gets instruction schedules ONLY, not evaluator targets.
    write(output / 'compiled/language.json', dict(schema_version=sidecar['schema_version'],
        sessions={k:dict(instruction_schedule=v['instruction_schedule']) for k,v in sidecar['sessions'].items()}))
    write(output / 'runtime/groups.json', groups)
    write(output / 'runtime/eval.json', json.loads((template / 'group-00/eval-config.yaml').read_text()))
    write(output / 'runtime/controller.json', json.loads(controller.read_text()))
    for name, path in driver_paths.items():
        driver = json.loads(path.read_text())
        driver['output_dir'] = '${OUTPUT}/model-driver'
        driver['model']['checkpoint_path'] = '${CHECKPOINT}'
        write(output / f'runtime/driver-{name}.json', driver)
    write(output / 'assets.json', dict(dataset=dict(repo_id='nvidia/PhysicalAI-Autonomous-Vehicles-NuRec',
        repo_type='dataset', revision='26.01', files=scene_files),
        renderer_image='nvcr.io/nvidia/nre/nre-ga:26.04',
        models={'alpamayo2': 'nvidia/Alpamayo2-Super', 'alpamayo15': 'nvidia/Alpamayo-1.5-10B'},
        scene_count=len(scene_ids)))
    write(output / 'manifest.json', dict(schema_version='l2d.portable_bundle.v1',
        pcu_count=30, scene_count=len(scene_ids), scheduled_slots=60, seed=1701,
        family_counts=dict(Counter(p['task_family'] for p in cohort['pcus'])),
        alpasim_commit='affc2eab209fa43bdfa2f26c0f8d437922d78a68',
        source_pcu_payloads_unchanged=False, source_review_logs_included=False,
        scoring_geometry_contracts_and_oracles_unchanged=True,
        authoring_provenance=dict(files_included=False, root_placeholder='${AUTHORING_ROOT}',
            runtime_dependency=False, note='Historical authoring review files are external; '
            'only their path references were rebased. Runtime and scoring do not read these files.'),
        reconstruction_and_weights_included=False,
        redistribution='private research bundle; upstream scene and model terms apply'))
    print(json.dumps(dict(bundle=str(output), pcus=30, slots=60, scenes=len(scene_ids))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--runtime-template', type=Path, required=True)
    parser.add_argument('--controller', type=Path, required=True)
    parser.add_argument('--driver-alpamayo2', type=Path, required=True)
    parser.add_argument('--driver-alpamayo15', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    export(a.source, a.runtime_template, dict(alpamayo2=a.driver_alpamayo2,
        alpamayo15=a.driver_alpamayo15), a.controller, a.output)
