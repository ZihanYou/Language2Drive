"""Version-bound human PCU review. Never infer approval from pilot videos.

The inventory binds small source/evidence files and large videos by exact stat,
without hashing assets. Reviewed source materialization is separate from both
the immutable drafts and formal oracle qualification.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import threading

from .constants import SCHEMA_VERSION
from .feasibility_review import ARM_FIELDS, CAMERAS
from .io import atomic_json, load_json
from .keep_change_pair_smoke import reference
from .release_profile import reduced_release_profile
from .schema import validate_pcu, validate_source

INVENTORY_SCHEMA = 'l2d.realsim.source_review_inventory.v0.1'
LOG_SCHEMA = 'l2d.realsim.source_reviews.v0.1'
FIELD_LABELS = {
    'shared_non_language_input.route_neutrality_reviewed': '共同路由支持两臂，不泄漏目标选择',
    'private_annotations.legality_reviewed': '两臂全程交通合法性（含信号、停车、让行）',
    'private_annotations.balance_reviewed': '两种语言选择公平、语义互斥且无额外任务难度偏置',
    'private_annotations.render_reviewed': '完整视频渲染可用于任务判断（允许轻微非任务瑕疵）',
    'private_annotations.shared_lane_scope_basis.same_major_route_reviewed': 'B2 两臂属于同一主要路线，共用同一实体范围',
    'private_annotations.shared_lane_scope_basis.marker_identity_reviewed': 'B2 范围标志身份与指令一致',
    'private_annotations.shared_lane_scope_basis.marker_visibility_reviewed': 'B2 标志在模型输入中及时可见',
    'private_annotations.trigger_referent.identity_reviewed': 'B3 触发参照物身份与指令一致',
    'private_annotations.trigger_referent.visibility_reviewed': 'B3 参照物在模型输入中及时可见',
    'private_annotations.trigger_referent.trigger_geometry_reviewed': 'B3 触发几何与真实参照物及时间关系一致',
}
CHECK_LABELS = {
    'complete_source_read': '已核对完整源稿、双臂指令及私有空间合同',
    'all_bound_videos_watched': '已观看本页绑定的全部双臂四相机视频',
    'language_contract_consistent': '每个语言变体准确对应自身合同，且与另一臂互斥',
    'scope_and_trigger_observable': '发令、范围和触发条件在模型输入中可观察且有可执行时间',
}


def _inside(root, path):
    candidate = Path(path)
    candidate = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError('证据路径不在项目内')
    return candidate


def _checked_ref(root, ref):
    path = _inside(root, ref['path'])
    if reference(path) != ref:
        raise ValueError('源稿或证据已变化，请重建版本化终审库存：' + str(path))
    return path


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _session_ids(value):
    return {row['session_uuid'] for row in _walk(value) if isinstance(row.get('session_uuid'), str)}


def review_fields(pcu):
    result = {}
    def collect(value, prefix=''):
        if isinstance(value, dict):
            for key, child in value.items():
                path = prefix + '.' + key if prefix else key
                if key.endswith('_reviewed') and type(child) is bool:
                    if path not in FIELD_LABELS:
                        raise ValueError('未定义的源稿审核项：' + path)
                    result[path] = FIELD_LABELS[path]
                else:
                    collect(child, path)
        elif isinstance(value, list):
            for number, child in enumerate(value):
                collect(child, f'{prefix}[{number}]')
    collect(pcu)
    return result


def prepare_source_review(root, index_path, output):
    """Build an immutable review target, selecting videos by actual session ID."""
    root = Path(root).resolve()
    index_path, output = _inside(root, index_path), _inside(root, output)
    if output.exists():
        raise ValueError('终审库存已存在；请使用新版本路径')
    index_ref = reference(index_path)
    index = load_json(index_path)
    profile = reduced_release_profile(index)
    if profile is None:
        raise ValueError('需要明确授权的执行配额')
    _checked_ref(root, index['profile_reference'])
    if load_json(index['profile_reference']['path']) != profile:
        raise ValueError('索引配额与绑定配置不一致')
    retained = {r['path']: r for r in index['retained_source_references']}
    static = root / '.runtime/curation_review'
    rows, pcus = [], []
    for item in index['pcus']:
        path = _inside(root, item['path'])
        ref = retained.get(str(path))
        if ref is None:
            raise ValueError('源稿缺少索引绑定')
        _checked_ref(root, ref)
        draft = load_json(path)
        pcu = draft['pcu']
        validate_pcu(pcu, allow_pending_reviews=True)
        if (pcu['pcu_id'] != item['pcu_id'] or draft['proposal_key'] != item['proposal_key']
            or pcu['task_family'] != item['task_family'] or draft['qualified'] is not False
            or draft['final_human_review'] != 'pending'):
            raise ValueError('源稿身份或待审状态不一致')
        batch_path = _checked_ref(root, draft['batch_binding'])
        slot = next(r for r in load_json(batch_path)['drafts'] if r['draft_id'] == draft['source_draft_id'])
        if slot['source_proposal_key'] != item['proposal_key'] or slot['task_family'] != item['task_family']:
            raise ValueError('源稿与历史批次不一致')
        refs = [ref, draft['batch_binding']]
        evidence, links = [], []
        for name in draft['evidence_paths']:
            evidence_path = _inside(root, name)
            refs.append(reference(evidence_path))
            evidence.append(load_json(evidence_path))
            if evidence_path.is_relative_to(static):
                links.append({'label': '来源证据：' + evidence_path.parent.name + '/' + evidence_path.name,
                              'url': '/' + str(evidence_path.relative_to(static))})
                page = evidence_path.parent / 'index.private.html'
                if page.is_file():
                    links.append({'label': '空间地图：' + page.parent.name, 'url': '/' + str(page.relative_to(static))})
                    refs.append(reference(page))
                    for svg in page.parent.glob('*.svg'):
                        refs.append(reference(svg))
        # Prefer the assembled current contract audit: its six fresh sessions
        # supersede ba590's retained historical post-hoc pair, for example.
        sessions = _session_ids(draft.get('assembled_actual_contract_audit', {}))
        if not sessions:
            sessions = _session_ids(evidence)
        if not sessions:
            raise ValueError('没有源稿绑定的实际会话：' + item['pcu_id'])
        packets, covered = [], set()
        for category in ('agent-feasibility', 'feasibility'):
            for inventory_path in sorted((static / category / item['proposal_key']).glob('*/inventory.private.json')):
                inventory = load_json(inventory_path)
                packet_sessions = {Path(f['source_asl']).parent.name for f in inventory.get('frames', [])}
                if not packet_sessions or not packet_sessions.issubset(sessions):
                    continue
                videos = inventory.get('videos', [])
                if (inventory.get('scope') != 'human_review_packet_not_qualification'
                    or inventory.get('proposal_key') != item['proposal_key'] or len(packet_sessions) != 2
                    or len(videos) != 8 or {(v['arm'], v['camera']) for v in videos}
                    != {(a, c) for a in ('R0', 'R1') for c in CAMERAS}):
                    raise ValueError('双臂证据包不完整')
                packet_refs = [reference(inventory_path)]
                for video in videos:
                    if Path(video['file']).name != video['file']:
                        raise ValueError('视频路径越界')
                    video_path = _inside(root, inventory_path.parent / video['file'])
                    if not video_path.is_relative_to(inventory_path.parent.resolve()):
                        raise ValueError('视频路径越界')
                    video_ref = reference(video_path)
                    if video_ref['size_bytes'] != video['size_bytes']:
                        raise ValueError('视频大小不符')
                    packet_refs.append(video_ref)
                for frame in inventory['frames']:
                    packet_refs.append(reference(_inside(root, frame['source_asl'])))
                refs.extend(packet_refs)
                for page in (static / 'spatial' / item['proposal_key'] / inventory_path.parent.name / 'index.private.html',
                             static / 'spatial' / inventory_path.parent.name / 'index.private.html'):
                    if page.is_file():
                        links.append({'label': '当前会话空间地图：' + inventory_path.parent.name,
                                      'url': '/' + str(page.relative_to(static))})
                        refs.append(reference(page))
                        refs.extend(reference(svg) for svg in page.parent.glob('*.svg'))
                packets.append(dict(seed=inventory['seed'], packet=inventory_path.parent.name,
                    session_ids=sorted(packet_sessions),
                    videos=[dict(arm=v['arm'], camera=v['camera'],
                        url='/' + str((inventory_path.parent / v['file']).relative_to(static))) for v in videos]))
                covered.update(packet_sessions)
        if covered != sessions:
            raise ValueError('缺少源稿实际会话的视频：' + item['pcu_id'])
        holds = deepcopy(draft.get('mandatory_review_holds', []))
        holds += [deepcopy(h) for h in index.get('mandatory_review_holds', []) if h.get('pcu_id') == pcu['pcu_id']]
        # Preserve even a source-only legal flag rather than trusting the index
        # to have copied it correctly. Only a new evidence-bound source may clear it.
        if any(r.get('legality_hold') is True for r in _walk(draft)) and not holds:
            holds.append({'legality_hold': True, 'reason': '源稿仍有合法性挂起'})
        unique_refs = list({r['path']: r for r in refs}.values())
        rows.append(dict(pcu_id=pcu['pcu_id'], task_family=pcu['task_family'], proposal_key=item['proposal_key'],
            source=ref, references=unique_refs, review_fields=review_fields(pcu), packets=packets,
            evidence_links=list({r['url']: r for r in links}.values()), mandatory_review_holds=holds,
            actual_session_ids=sorted(sessions)))
        pcus.append(pcu)
    if (len(pcus) != profile['total_pcus'] or len({p['pcu_id'] for p in pcus}) != len(pcus)
        or len({p['meta_group_id'] for p in pcus}) != len(pcus)
        or Counter(p['primary_layer'] for p in pcus) != Counter(profile['layer_targets'])
        or Counter(p['task_family'] for p in pcus) != Counter(profile['family_targets'])):
        raise ValueError('候选数量、唯一性或配额不符')
    _checked_ref(root, index_ref)
    for row in rows:
        for ref in row['references']:
            _checked_ref(root, ref)
    result = dict(schema_version=INVENTORY_SCHEMA, index_reference=index_ref,
        profile_reference=index['profile_reference'], release_profile=profile, sources=rows,
        source_count=len(rows), human_approval_created=False, qualified=False,
        formal_witnesses=0, asset_hashes_computed=0)
    atomic_json(output, result)
    return {'inventory': str(output), 'sources': len(rows), 'qualified': False}


class SourceReviewStore:
    def __init__(self, root, inventory_path, decisions_path):
        self.root = Path(root).resolve()
        self.inventory_path = _inside(self.root, inventory_path)
        self.decisions_path = _inside(self.root, decisions_path)
        if (self.decisions_path == self.inventory_path
            or not self.decisions_path.is_relative_to(self.root / 'data/private/source_reviews')):
            raise ValueError('源稿审核只允许写入独立 data/private/source_reviews 目录')
        self._lock = threading.RLock()

    @contextmanager
    def _transaction(self):
        with self._lock:
            self.decisions_path.parent.mkdir(parents=True, exist_ok=True)
            with self.decisions_path.with_suffix('.lock').open('a') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def _inventory(self):
        inventory = load_json(self.inventory_path)
        if inventory.get('schema_version') != INVENTORY_SCHEMA:
            raise ValueError('终审库存格式无效')
        _checked_ref(self.root, inventory['index_reference'])
        _checked_ref(self.root, inventory['profile_reference'])
        return inventory

    def _log(self):
        if not self.decisions_path.exists():
            return dict(schema_version=LOG_SCHEMA, decisions={}, audit_log=[])
        log = load_json(self.decisions_path)
        if (log.get('schema_version') != LOG_SCHEMA or not isinstance(log.get('decisions'), dict)
            or not isinstance(log.get('audit_log'), list)):
            raise ValueError('源稿终审日志格式无效')
        return log

    def snapshot(self, pcu_id):
        with self._lock:
            row = next((r for r in self._inventory()['sources'] if r['pcu_id'] == pcu_id), None)
            if row is None:
                raise ValueError('未知 PCU')
            for ref in row['references']:
                _checked_ref(self.root, ref)
            source = load_json(row['source']['path'])
            binding = dict(inventory=reference(self.inventory_path), source=row['source'], pcu_id=pcu_id)
            record = self._log()['decisions'].get(pcu_id)
            stale = bool(record and record['binding'] != binding)
            return dict(binding=binding, revision=record['revision'] if record else 0,
                record=None if stale else record, previous_review_stale=stale, source=source,
                target=row, checks=CHECK_LABELS, qualified=False)

    def overview(self):
        inventory, log = self._inventory(), self._log()
        inventory_ref = reference(self.inventory_path)
        rows = []
        for row in inventory['sources']:
            record = log['decisions'].get(row['pcu_id'])
            binding = dict(inventory=inventory_ref, source=row['source'], pcu_id=row['pcu_id'])
            fresh = bool(record and record['binding'] == binding)
            # A stale file must never still be listed as approved in the queue.
            try:
                for ref in row['references']:
                    _checked_ref(self.root, ref)
                error = None
            except (OSError, ValueError) as exc:
                error = str(exc)
                fresh = False
            rows.append(dict(pcu_id=row['pcu_id'], task_family=row['task_family'],
                proposal_key=row['proposal_key'], hold=bool(row['mandatory_review_holds']),
                decision=record['decision'] if fresh else 'unreviewed', error=error))
        return dict(sources=rows, qualified=False, formal_witnesses=0)

    def submit(self, pcu_id, payload):
        with self._transaction():
            snap = self.snapshot(pcu_id)
            allowed = {'binding', 'expected_revision', 'reviewer', 'decision', 'notes', 'fields', 'checks', 'arms'}
            if not isinstance(payload, dict) or set(payload) != allowed:
                raise ValueError('提交字段不完整或包含资格／源稿修改字段')
            if payload['binding'] != snap['binding']:
                raise ValueError('源稿版本已变化，请刷新')
            if type(payload['expected_revision']) is not int or payload['expected_revision'] != snap['revision']:
                raise ValueError('审核已被其他页面更新，请刷新')
            reviewer, notes = payload['reviewer'], payload['notes']
            if not isinstance(reviewer, str) or not 1 <= len(reviewer.strip()) <= 100:
                raise ValueError('请填写 1–100 字审核人')
            if not isinstance(notes, str) or not 1 <= len(notes.strip()) <= 4000:
                raise ValueError('请填写 1–4000 字审核依据；待定／拒绝注明原因')
            decision = payload['decision']
            if not isinstance(decision, str) or decision not in {'keep', 'pending', 'reject'}:
                raise ValueError('无效审核结论')
            for name, keys in [('fields', snap['target']['review_fields']), ('checks', CHECK_LABELS)]:
                if not isinstance(payload[name], dict) or set(payload[name]) != set(keys):
                    raise ValueError('必须逐项核对源稿审核字段')
            arms = payload['arms']
            if (not isinstance(arms, dict) or set(arms) != {'R0', 'R1'}
                or any(not isinstance(v, dict) or set(v) != set(ARM_FIELDS) for v in arms.values())):
                raise ValueError('必须分别核对两臂')
            values = [*payload['fields'].values(), *payload['checks'].values(),
                      *(v for arm in arms.values() for v in arm.values())]
            if any(v is not None and type(v) is not bool for v in values):
                raise ValueError('审核值必须为 true、false 或 null')
            if decision == 'keep':
                if snap['target']['mandatory_review_holds']:
                    raise ValueError('源稿有强制挂起项；须先补证据并生成新版本，不能直接通过')
                if any(v is not True for v in values):
                    raise ValueError('通过前须明确完成全部审核项')
            record = dict(binding=snap['binding'], revision=snap['revision'] + 1,
                reviewer=reviewer.strip(), notes=notes.strip(), decision=decision,
                fields=deepcopy(payload['fields']), checks=deepcopy(payload['checks']), arms=deepcopy(arms),
                reviewed_at=datetime.now(timezone.utc).isoformat(), scope='complete_source_human_review_not_qualification',
                qualified=False, formal_witnesses=0)
            log = self._log()
            # Recheck after validation before committing the separate log.
            if self.snapshot(pcu_id)['binding'] != snap['binding']:
                raise ValueError('审核期间源稿发生变化')
            log['decisions'][pcu_id] = record
            log['audit_log'].append(deepcopy(record))
            atomic_json(self.decisions_path, log)
            return dict(record=record, source_modified=False, qualified=False)

    def materialize(self, output):
        """Create a new strict source only after every exact source is approved."""
        return self._materialize(output, accepted_subset=False)

    def stage_accepted(self, output):
        """Stage only exact approvals for CPU preflight, never a smaller release.

        Session IDs bind the entire source. A compiled subset must be rebuilt
        after completing the final cohort; its runs are not final witnesses.
        """
        return self._materialize(output, accepted_subset=True)

    def _materialize(self, output, *, accepted_subset):
        output = _inside(self.root, output)
        directory = 'reviewed_staging' if accepted_subset else 'reviewed_sources'
        if output.exists() or not output.is_relative_to(self.root / 'data/private' / directory):
            raise ValueError('仅允许新建 data/private/' + directory + ' 下的源文件')
        with self._transaction():
            if output.exists():
                raise ValueError('仅允许新建源文件；输出已存在')
            inventory = self._inventory()
            pcus, receipts, excluded = [], [], []
            for row in inventory['sources']:
                snap = self.snapshot(row['pcu_id'])
                record = snap['record']
                if not record or record['decision'] != 'keep' or row['mandatory_review_holds']:
                    if accepted_subset:
                        excluded.append(dict(pcu_id=row['pcu_id'], task_family=row['task_family'],
                            proposal_key=row['proposal_key'],
                            decision=record['decision'] if record else 'unreviewed',
                            previous_review_stale=snap['previous_review_stale'],
                            mandatory_review_holds=deepcopy(row['mandatory_review_holds']),
                            source_review_receipt=deepcopy(record)))
                        continue
                    raise ValueError('尚未完成当前源稿终审：' + row['pcu_id'])
                expected = set(row['review_fields'])
                if (set(record['fields']) != expected or set(record['checks']) != set(CHECK_LABELS)
                    or any(v is not True for v in [*record['fields'].values(), *record['checks'].values()])
                    or set(record['arms']) != {'R0', 'R1'}
                    or any(set(a) != set(ARM_FIELDS) or any(v is not True for v in a.values()) for a in record['arms'].values())):
                    raise ValueError('终审记录未明确通过所有字段')
                pcu = deepcopy(snap['source']['pcu'])
                for field in expected:
                    target = pcu
                    keys = field.split('.')
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = True
                validate_pcu(pcu)
                pcus.append(pcu)
                receipts.append(deepcopy(record))
            if accepted_subset:
                if not pcus:
                    raise ValueError('没有可预检的当前通过源稿')
                if not excluded:
                    raise ValueError('全部源稿已通过，请使用 materialize-reviewed-source')
            result = dict(schema_version=SCHEMA_VERSION, release_profile=inventory['release_profile'], pcus=pcus,
                source_review_receipts=receipts, source_review_log=reference(self.decisions_path))
            if accepted_subset:
                result.update(scope='approved_subset_preflight_not_release', release_ready=False,
                    qualified=False, formal_witnesses=0, excluded_sources=excluded,
                    source_review_inventory=reference(self.inventory_path),
                    final_cohort_recompile_required=True)
            validate_source(result, require_exact_release=not accepted_subset)
            for row in inventory['sources']:
                self.snapshot(row['pcu_id'])
            atomic_json(output, result)
            return dict(output=str(output), strict_source_pcus=len(pcus), qualified=False, formal_witnesses=0,
                accepted_subset=accepted_subset, excluded_count=len(excluded))


def migrate_unchanged_reviews(root, previous_inventory, previous_decisions, inventory, output):
    """Explicitly carry exact source decisions to a new inventory, never new approval.

    Match the entire target row, including holds, evidence refs, videos and fields.
    Preserve original reviewer/time/revision and a full receipt in migration history.
    Default snapshot behavior remains fail-closed for every other stale inventory.
    """
    root = Path(root).resolve()
    paths = [_inside(root, p) for p in (previous_inventory, previous_decisions, inventory, output)]
    if len(set(paths)) != 4:
        raise ValueError('迁移必须使用不同的旧库存、旧日志、新库存和新日志路径')
    old = SourceReviewStore(root, paths[0], paths[1])
    new = SourceReviewStore(root, paths[2], paths[3])
    with old._transaction(), new._transaction():
        if new.decisions_path.exists():
            raise ValueError('迁移只能新建日志，不能覆盖已有审核')
        refs = [reference(p) for p in paths[:3]]
        before, after = old._inventory(), new._inventory()
        for inv in (before, after):
            if reduced_release_profile(inv) is None:
                raise ValueError('迁移需要有效的显式配额')
            if load_json(_checked_ref(root, inv['profile_reference'])) != inv['release_profile']:
                raise ValueError('审核库存配额与配置不一致')
            if len({r['pcu_id'] for r in inv['sources']}) != len(inv['sources']):
                raise ValueError('迁移库存有重复 PCU')
        old_rows = {r['pcu_id']: r for r in before['sources']}
        log = old._log()
        decisions, skipped, receipts = {}, [], []
        for target in after['sources']:
            pcu_id = target['pcu_id']
            new_snap = new.snapshot(pcu_id)  # Revalidate every new target's evidence.
            previous = old_rows.get(pcu_id)
            if previous is None or previous != target:
                skipped.append(dict(pcu_id=pcu_id, reason='new_target' if previous is None else 'changed_target'))
                continue
            snap = old.snapshot(pcu_id)
            record = snap['record']
            if record is None:
                skipped.append(dict(pcu_id=pcu_id, reason='missing_or_stale_review'))
                continue
            # Do not carry hand-edited malformed decisions into a fresh binding.
            if (any(not isinstance(record.get(k), dict) for k in ('fields', 'checks', 'arms'))
                or any(not isinstance(a, dict) for a in record['arms'].values())):
                raise ValueError('原审核记录不完整或含无效通过结论')
            values = [*record.get('fields', {}).values(), *record.get('checks', {}).values(),
                      *(v for a in record.get('arms', {}).values() for v in a.values())]
            if (record.get('scope') != 'complete_source_human_review_not_qualification'
                or record.get('qualified') is not False or record.get('formal_witnesses') != 0
                or type(record.get('revision')) is not int or record['revision'] < 1
                or record.get('decision') not in {'keep', 'reject', 'pending'}
                or not isinstance(record.get('reviewer'), str) or not record['reviewer'].strip()
                or not isinstance(record.get('reviewed_at'), str) or not record['reviewed_at']
                or set(record.get('fields', {})) != set(target['review_fields'])
                or set(record.get('checks', {})) != set(CHECK_LABELS)
                or set(record.get('arms', {})) != {'R0', 'R1'}
                or any(set(a) != set(ARM_FIELDS) for a in record['arms'].values())
                or any(v is not None and type(v) is not bool for v in values)
                or (record['decision'] == 'keep' and
                    (target['mandatory_review_holds'] or any(v is not True for v in values)))):
                raise ValueError('原审核记录不完整或含无效通过结论')
            migrated = deepcopy(record)
            migrated['binding'] = new_snap['binding']
            # A migration is an engineering event, not a newly authored human review.
            migrated['migration_provenance'] = dict(previous_inventory=refs[0], previous_log=refs[1],
                previous_binding=deepcopy(record['binding']), original_review_unchanged=True)
            decisions[pcu_id] = migrated
            receipts.append(dict(pcu_id=pcu_id, original_record=deepcopy(record), migrated_record=deepcopy(migrated)))
        event = dict(scope='exact_unchanged_source_review_migration_not_new_human_approval',
            migrated_at=datetime.now(timezone.utc).isoformat(), source_references=refs,
            receipts=receipts, skipped=skipped,
            retired_pcu_ids=sorted(set(old_rows)-{r['pcu_id'] for r in after['sources']}),
            profile_changed=before['release_profile'] != after['release_profile'],
            cohort_approval_inherited=False, human_approvals_created=0, formal_witnesses=0)
        for ref in refs:
            _checked_ref(root, ref)
        for target in after['sources']:
            for ref in target['references']:
                _checked_ref(root, ref)
        result = dict(schema_version=LOG_SCHEMA, decisions=decisions,
            audit_log=deepcopy(log['audit_log'])+[event], inventory_migration=event)
        atomic_json(new.decisions_path, result)
        return dict(output=str(new.decisions_path), records_migrated=len(decisions),
            decision_counts=dict(Counter(r['decision'] for r in decisions.values())),
            unreviewed_count=len(skipped), human_approvals_created=0, qualified=False, formal_witnesses=0)
