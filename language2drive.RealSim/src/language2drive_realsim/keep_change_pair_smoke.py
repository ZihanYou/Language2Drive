"""Bounded delegated keep/change pilots, independently regenerated before use."""
from pathlib import Path
import os
import subprocess
import sys
import uuid

import numpy as np
import yaml

from .agent_curation import load_agent_candidate, validate_agent_runs
from .constants import COUPLED_SEEDS, ORACLE_CONTROLLER_ID
from .io import load_json, atomic_json
from .keep_change_regions import LINKED_VERSION
from .path_turning import turning_budget
from .planned_lane_contract import audit_planned_lane_contract, VERSION as FIRST_COMMIT_VERSION

SCOPE = "real_counterfactual_pair_integration_not_qualified_oracle"


def reference(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=str(stat.st_mtime_ns))


def validate_keep_change_search(search, candidate):
    """Validate the new protocol only; never promote legacy or partial searches."""
    pair = candidate['selected_arm_pair']
    if (search.get('scope') != 'unreviewed_offline_lane_path_search'
        or search.get('semantic_prefilter_version') not in (LINKED_VERSION,FIRST_COMMIT_VERSION)
        or search.get('complete') is not True or search.get('qualified') is not False
        or search.get('input_human_reviewed') is not False
        or search.get('original_native_map_unchanged') is not True
        or search.get('live_submission_supported_by_this_search') is not False
        or search.get('selected_arm_pair') != pair
        or pair.get('kind') not in ('keep_vs_left','keep_vs_right')
        or [(a['arm'],a['role']) for a in pair['arms']] != [
            ('R0','keep'),('R1','change_left' if pair['kind']=='keep_vs_left' else 'change_right')]
        or len({a['lane_id'] for a in pair['arms']}) != 2):
        raise ValueError('complete, unreviewed linked-original keep/change hypothesis required')
    for sk,ck in (('proposal_key','proposal_key'),('artifact_uuid','artifact_uuid'),
                  ('scene_id','scene_id_private'),('review_revision','review_revision')):
        if search.get(sk) != candidate[ck]:
            raise ValueError('keep/change search identity mismatch')
    if (search.get('start_timestamp_us') != candidate['scene_execution_proposal_not_frozen']['start_timestamp_us']
        or search.get('rollout_duration_s') != 7. or search.get('intervention_s') != 1.7
        or search.get('sampling_period_s') != .1):
        raise ValueError('keep/change original window or timing mismatch')
    selected = search.get('selected_parameters')
    attempts = [a for a in search.get('attempts',[]) if a.get('parameters') == selected]
    if not isinstance(selected,dict) or len(attempts) != 1:
        raise ValueError('one uniquely selected keep/change hypothesis required')
    attempt = attempts[0]
    if (attempt.get('sampled_prefilter_passed') is not True
        or attempt.get('semantic_prefilter',{}).get('passed') is not True):
        raise ValueError('selected keep/change semantics failed')
    for arm,key in (('R0','keep_audit'),('R1','audit')):
        road = attempt.get('planned_native_road',{}).get(arm,{})
        motion = attempt.get(key,{})
        values = [motion.get(k) for k in ('minimum_2d_clearance_m',
            'max_planar_acceleration_mps2','max_lateral_acceleration_mps2')]
        if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in values)
            or values[0] < .5 or values[1] > 3.5 or values[2] > 3.
            or motion.get('overlap_sample_count') != 0
            or road.get('passed') is not True or road.get('samples') != 131
            or road.get('interval_s') != [0.,13.] or road.get('original_geometry_modified') is not False
            or road.get('native_alarm_times_s') != {'offroad':[], 'wrong_lane':[]}
            or attempt.get('turning_prefilter',{}).get(arm,{}).get('passed') is not True):
            raise ValueError('full original native road, motion and traffic checks required')
    if (search.get('post_task_return_hypothesis') is not None
        and attempt.get('return_geometry_rig_not_swept_volume',{}).get('returned_to_source_exclusively') is not True):
        raise ValueError('complete exclusive post-task return required')
    plans = search.get('selected_plans')
    if not isinstance(plans,dict) or set(plans) != {'R0','R1'}:
        raise ValueError('exactly two selected keep/change plans required')
    prefixes = []
    for arm,plan in plans.items():
        if plan.get('controller') != ORACLE_CONTROLLER_ID or plan.get('coordinate_frame') != 'nre_local':
            raise ValueError('keep/change controller or frame mismatch')
        rows = np.asarray([[w[k] for k in ('time_s','x','y','z','yaw_rad')] for w in plan['waypoints']])
        if (rows.shape != (131,5) or not np.isfinite(rows).all()
            or not np.array_equal(rows[:,0],np.round(np.arange(131)/10.,6))
            or not turning_budget(rows[:,0],rows[:,1:4],rows[:,4])['passed']):
            raise ValueError('dense finite full 13-second steering-bounded plans required')
        prefixes.append(rows[rows[:,0] <= 1.7].tolist())
    if prefixes[0] != prefixes[1]:
        raise ValueError('keep/change preintervention plans differ')
    if search['semantic_prefilter_version'] == FIRST_COMMIT_VERSION:
        checked=audit_planned_lane_contract(plans,search['rig_to_aabb_center_matrix'],
            {a:search['linked_original_section_hypothesis'][a]['original_sections'] for a in ('R0','R1')})
        if (not checked['all_own_pass_opposing_fail']
            or checked != attempt.get('planned_lane_contract_precheck')):
            raise ValueError('complete unchanged planned original first-commit contract required')
    return plans


def prepare_keep_change_pair(run_dir, search_path, seed=1701):
    root = Path(__file__).resolve().parents[2]
    run_dir, search_path = Path(run_dir).resolve(), Path(search_path).resolve()
    if type(seed) is not int or seed not in COUPLED_SEEDS:
        raise ValueError('supported coupled seed required')
    if not run_dir.is_relative_to(root/'.runtime') or not search_path.is_relative_to(root/'.runtime'):
        raise ValueError('project-local keep/change pilot inputs required')
    recheck_path = run_dir/'keep-change-preparation-recheck.private.json'
    if any((run_dir/name).exists() for name in ('lane-pair-plan.private.json','lane-pair-sidecar.private.json',
        'lane-pair-return.private.json','speed-pair-plan.private.json',recheck_path.name)):
        raise ValueError('preserve previous preparations and actual evidence')
    launch,search = load_json(run_dir/'launch.private.json'),load_json(search_path)
    if (launch.get('scope') != 'exploratory_runtime_not_oracle_witness'
        or launch.get('driver_binding') != 'registered_static_for_asl_authority'):
        raise ValueError('prepared static exploratory runtime required')
    candidate = validate_agent_runs(root,[run_dir])
    source = Path(launch['agent_candidate_source']).resolve()
    expected_refs = [reference(source),reference(candidate['scene_execution_proposal_not_frozen']['artifact_path'])]
    if search.get('source_references') != expected_refs:
        raise ValueError('current source and original asset bindings required')
    if search.get('semantic_prefilter_version') != FIRST_COMMIT_VERSION:
        raise ValueError('new keep/change preparation requires full original first-commit precheck; old pilots remain readable')
    plans = validate_keep_change_search(search,candidate)
    params=search['selected_parameters']
    command=[sys.executable,str(root/'scripts/search_lane_pair.py'),str(run_dir),
        '--offline-candidate',str(source),'--output-name',recheck_path.name,'--linked-original-sections',
        '--precheck-original-lane-contract',
        '--minimum-corridor-length-m',str(search['minimum_forward_corridor_length_requested_m']),
        '--speed-fractions',str(params['speed_fraction']),'--lane-change-starts-s',str(params['lane_change_start_s']),
        '--lane-change-duration-s',str(params['lane_change_duration_s']),
        '--target-progress-mode',search['target_progress_mode']]
    if search['observed_prefix_history_enabled']:
        command.append('--observed-prefix-history')
    if search['completed_stop_heading_hypothesis']:
        command.append('--hold-completed-stop-heading')
    returning=search['post_task_return_hypothesis']
    if returning is not None:
        command += ['--return-start-s',str(returning['start_s']),'--return-duration-s',str(returning['duration_s'])]
    env={k:v for k,v in os.environ.items() if k.lower() not in {'http_proxy','https_proxy','all_proxy'}}
    env.update(CUDA_VISIBLE_DEVICES='',POLARS_MAX_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (run_dir/'keep-change-preparation-recheck.log').open('x') as log:
        subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    recheck=load_json(recheck_path)
    if (validate_keep_change_search(recheck,candidate) != plans
        or recheck['linked_original_section_hypothesis'] != search['linked_original_section_hypothesis']
        or recheck['source_references'] != expected_refs
        or recheck['selected_parameters'] != params
        or validate_agent_runs(root,[run_dir]) != candidate or load_json(search_path) != search):
        raise ValueError('keep/change input, original joins or regenerated plans changed')
    sidecar=dict(schema_version='l2d.realsim.sidecar.v0.1',execution_scope=SCOPE,hypothesis_kind='lane',
        lane_pair_kind=candidate['selected_arm_pair']['kind'],task_family_proposal='A5_lane_change_keep',
        human_reviewed=False,sessions={},keep_change_search_reference=reference(recheck_path))
    selected_attempt=next(a for a in recheck['attempts'] if a['parameters']==params)
    sidecar['predeclared_original_lane_contract']=selected_attempt['planned_lane_contract_precheck']
    plan=dict(source_binding=SCOPE,run_count=2,runs=[],hypothesis_kind='lane',
              diagnostic_search_reference=reference(recheck_path))
    scene=candidate['scene_execution_proposal_not_frozen']
    for arm in ('R0','R1'):
        session=str(uuid.uuid4())
        sidecar['sessions'][session]=dict(arm_id=arm,seed=seed,start_timestamp_us=scene['start_timestamp_us'],
            instruction_issued_at_us=scene['start_timestamp_us']+1700000,oracle_plan=plans[arm])
        plan['runs'].append(dict(scenario_id=candidate['scene_id_private'],session_uuid=session,
                                start_time_offset_us=scene['start_time_offset_us'],random_seed=seed))
    user_path=run_dir/'user-config.yaml'
    user=yaml.safe_load(user_path.read_text())
    user['simulation_config']['n_sim_steps']=70
    launch['execution_scope']=SCOPE
    launch['keep_change_hypothesis']=dict(qualified=False,human_reviewed=False,
        source_search_reference=reference(search_path),regenerated_search_reference=reference(recheck_path))
    atomic_json(run_dir/'lane-pair-sidecar.private.json',sidecar)
    atomic_json(run_dir/'lane-pair-plan.private.json',plan)
    atomic_json(run_dir/'launch.private.json',launch)
    user_path.write_text(yaml.safe_dump(user,sort_keys=False))
    return dict(scope=SCOPE,sessions=2,seed=seed,qualified=False,human_reviewed=False)


def validate_keep_change_runtime_protocol(run_dir):
    """Recheck linked plans, source stats, timing and sessions before submission."""
    root=Path(__file__).resolve().parents[2]
    run_dir=Path(run_dir).resolve()
    candidate=validate_agent_runs(root,[run_dir])
    launch=load_json(run_dir/'launch.private.json')
    sidecar=load_json(run_dir/'lane-pair-sidecar.private.json')
    plan=load_json(run_dir/'lane-pair-plan.private.json')
    hypothesis=launch.get('keep_change_hypothesis')
    if (not isinstance(hypothesis,dict) or hypothesis.get('qualified') is not False
        or hypothesis.get('human_reviewed') is not False):
        raise ValueError('explicit unqualified keep/change launch binding required')
    recheck=run_dir/'keep-change-preparation-recheck.private.json'
    expected=reference(recheck)
    if (hypothesis.get('regenerated_search_reference') != expected
        or sidecar.get('keep_change_search_reference') != expected
        or plan.get('diagnostic_search_reference') != expected):
        raise ValueError('keep/change recheck references changed')
    original_ref=hypothesis['source_search_reference']
    original_path=Path(original_ref['path']).resolve()
    if not original_path.is_relative_to(root/'.runtime') or reference(original_path) != original_ref:
        raise ValueError('original keep/change search changed')
    search=load_json(recheck)
    original=load_json(original_path)
    plans=validate_keep_change_search(search,candidate)
    if search['semantic_prefilter_version'] == FIRST_COMMIT_VERSION:
        selected=next(a for a in search['attempts'] if a['parameters']==search['selected_parameters'])
        if sidecar.get('predeclared_original_lane_contract') != selected['planned_lane_contract_precheck']:
            raise ValueError('predeclared original lane contract changed in actual sidecar')
    if (validate_keep_change_search(original,candidate) != plans
        or original['linked_original_section_hypothesis'] != search['linked_original_section_hypothesis']
        or search['source_references'] != [reference(launch['agent_candidate_source']),
            reference(candidate['scene_execution_proposal_not_frozen']['artifact_path'])]):
        raise ValueError('keep/change original plans, geometry or asset changed')
    user=yaml.safe_load((run_dir/'user-config.yaml').read_text())
    if (user['simulation_config']['n_sim_steps'] != 70 or plan.get('run_count') != 2
        or len(plan.get('runs',[])) != 2 or len(sidecar.get('sessions',{})) != 2
        or plan.get('source_binding') != SCOPE or sidecar.get('execution_scope') != SCOPE
        or sidecar.get('lane_pair_kind') != candidate['selected_arm_pair']['kind']
        or sidecar.get('human_reviewed') is not False):
        raise ValueError('keep/change duration or pair binding changed')
    arms,seeds=set(),set()
    scene=candidate['scene_execution_proposal_not_frozen']
    session_ids=set()
    for run in plan['runs']:
        session=sidecar['sessions'].get(run['session_uuid'],{})
        arm=session.get('arm_id')
        seed=session.get('seed')
        if (arm not in plans or type(seed) is not int or seed not in COUPLED_SEEDS
            or run['random_seed'] != seed or run['scenario_id'] != candidate['scene_id_private']
            or run['start_time_offset_us'] != scene['start_time_offset_us']
            or session.get('start_timestamp_us') != scene['start_timestamp_us']
            or session.get('instruction_issued_at_us') != scene['start_timestamp_us']+1700000
            or session.get('oracle_plan') != plans[arm]):
            raise ValueError('keep/change actual session plan, window or seed changed')
        arms.add(arm)
        seeds.add(seed)
        session_ids.add(run['session_uuid'])
    if arms != {'R0','R1'} or len(seeds) != 1 or len(session_ids) != 2:
        raise ValueError('two distinct keep/change arms and one coupled seed required')
    return {'validated':True,'qualified':False,'seed':next(iter(seeds))}
