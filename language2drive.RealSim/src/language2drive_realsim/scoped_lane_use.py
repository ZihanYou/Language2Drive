"""Original B2 lane-label exclusions before loading a complete scene."""
from copy import deepcopy
import io
from pathlib import Path
import zipfile


def audit_scoped_lane_use(candidate, raw_lanes):
    pair = candidate['selected_arm_pair']
    kind = pair['kind']
    if kind not in ('keep_vs_left', 'keep_vs_right'):
        raise ValueError('original keep/adjacent B2 pair required')
    expected = [('R0','keep'),('R1','change_left' if kind == 'keep_vs_left' else 'change_right')]
    if [(a['arm'],a['role']) for a in pair['arms']] != expected:
        raise ValueError('exact keep/adjacent B2 roles required')
    ids = [str(a['lane_id']) for a in pair['arms']]
    if len(set(ids)) != 2:
        raise ValueError('two distinct original lanes required')
    clip = candidate['scene_id_private'].removeprefix('clipgt-')
    checks = []
    for arm, identifier in zip(('R0','R1'),ids):
        matches = [r for r in raw_lanes if r['key']['map_id'] == identifier]
        if len(matches) != 1 or matches[0]['key']['clip_id'] != clip:
            raise ValueError('one original matching clip/lane record required')
        row = matches[0]
        use, vehicles = row['lane']['use_types'], row['lane']['vehicle_types']
        if any(not isinstance(v,list) or any(not isinstance(x,str) for x in v) for v in (use,vehicles)):
            raise ValueError('original lane label lists required')
        reasons = []
        if 'SHOULDER_LANE' in use:
            reasons.append('explicit_original_shoulder')
        if vehicles and 'CAR' not in vehicles:
            reasons.append('explicit_original_vehicle_types_exclude_car')
        checks.append(dict(arm=arm,source_lane_record=deepcopy(row),exclusion_reasons=reasons))
    if len({c['source_lane_record']['key']['map_id_version'] for c in checks}) != 1:
        raise ValueError('original lane map versions differ')
    return dict(scope='original_B2_selected_lane_use_exclusion_not_legality',qualified=False,human_reviewed=False,
        legality_approved=False,full_corridor_use_not_checked=True,checks=checks,
        passed=not any(c['exclusion_reasons'] for c in checks))


def read_scoped_lane_use(candidate):
    import pyarrow.parquet as pq
    path = Path(candidate['scene_execution_proposal_not_frozen']['artifact_path']).resolve()
    before = path.stat()
    with zipfile.ZipFile(path) as archive:
        rows = pq.read_table(io.BytesIO(archive.read('clipgt/lane.parquet')),use_threads=False).to_pylist()
    after = path.stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
        raise ValueError('asset changed during original lane-label read')
    return dict(audit_scoped_lane_use(candidate,rows),source_artifact=str(path),
        source_size_bytes=after.st_size,source_mtime_ns=str(after.st_mtime_ns))
