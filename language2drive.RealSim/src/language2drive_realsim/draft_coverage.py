"""Explicit user-authorized B5/A6 omissions for drafts, never qualification."""
from copy import deepcopy
from pathlib import Path

from .family_allocation import family_targets
from .io import load_json

KEY = "authoring_coverage_reallocation"
PROFILE = "omit_B5_within_B_draft_v1"
DONOR = "B5_relative_stopping_region"
RECEIVER = "B1_segment_speed_band"
A6_PROFILE = "omit_A6_and_B5_within_layers_draft_v1"
A6_DONOR = "A6_pull_over_continue"
A5_RECEIVER = "A5_lane_change_keep"


def draft_family_targets(batch):
    base = family_targets({k: v for k, v in batch.items() if k != KEY})
    if KEY not in batch:
        return base
    d = batch[KEY]
    expected = dict(base)
    expected[RECEIVER] += expected[DONOR]
    expected[DONOR] = 0
    combined = isinstance(d, dict) and d.get("profile_id") == A6_PROFILE
    omitted = [DONOR]
    if combined:
        previous = d.get("previous_declaration")
        if not isinstance(previous,dict) or previous.get("profile_id") != PROFILE:
            raise ValueError("combined omission must preserve the prior B5 declaration")
        draft_family_targets({**batch, KEY: previous})
        expected[A5_RECEIVER] += expected[A6_DONOR]
        expected[A6_DONOR] = 0
        omitted = [A6_DONOR,DONOR]
    if (not isinstance(d, dict) or d.get("profile_id") not in (PROFILE,A6_PROFILE)
        or batch.get("scope") != "asset_bound_authoring_drafts_not_qualified_pcus"
        or d.get("formal_release_profile_approved") is not False
        or d.get("base_targets") != base or d.get("targets") != expected
        or any(type(n) is not int for n in d.get("targets", {}).values())
        or d.get("omitted_families") != omitted
        or d.get("model_evaluation_used_for_allocation") is not False
        or not isinstance(d.get("reason"), str) or not d["reason"].strip()
        or not isinstance(d.get("authorization_ref"), str) or not d["authorization_ref"].strip()):
        raise ValueError("explicit draft-only within-B B5 omission declaration required")
    refs = d.get("availability_evidence_refs")
    if (not isinstance(refs, list) or not refs
        or any(not isinstance(p, str) or not p.strip() for p in refs)
        or len(set(refs)) != len(refs)):
        raise ValueError("distinct availability evidence references required")
    return expected


def omit_unbound_b5(root, batch, *, reason, evidence_refs):
    """Copy a batch and reset only untouched B5 hypotheses into empty B1 slots.

    The original rows remain in the copied history. No source or semantic
    evidence is relabeled; actual B1 assembly must independently bind evidence.
    """
    from .agent_drafts import TEMPLATES, validate_draft_batch
    validate_draft_batch(batch)
    if KEY in batch:
        raise ValueError("preserve existing coverage declarations; do not apply twice")
    root = Path(root).resolve()
    auth_ref = "config/task_coverage_relaxation.private.json"
    auth = load_json(root / auth_ref)
    if (auth.get("agent_may_omit_hard_to_source_task_families") is not True
        or auth.get("target_total_pcus") != 50
        or auth.get("target_real_oracle_witnesses") != 300
        or auth.get("require_final_human_review_before_release") is not True
        or auth.get("missing_evidence_may_be_marked_passed") is not False
        or auth.get("relabel_existing_pcus_as_other_task_families") is not False
        or DONOR not in auth.get("immediate_search_pauses", [])):
        raise ValueError("explicit current user authority to omit B5 required")
    if not isinstance(evidence_refs, list) or not evidence_refs:
        raise ValueError("local availability evidence required")
    for ref in evidence_refs:
        if not isinstance(ref, str):
            raise ValueError("local availability evidence required")
        path = (root / ref).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("existing project-local availability evidence required")
    rows = [r for r in batch["drafts"] if r["task_family"] == DONOR]
    if not rows:
        raise ValueError("untouched B5 slots required")
    forbidden = {"authored_pcu_path", "exploratory_evidence", "agent_pilot_evidence",
                 "agent_branch_pilot_run", "map_bound_contract_draft", "agent_visual_observations",
                 "visual_grounding_lead", "visual_observations"}
    for row in rows:
        if (row.get("family_binding_status") != "unverified_hypothesis"
            or row.get("grounding_bindings") != {"R0": None, "R1": None}
            or forbidden.intersection(row)
            or any(a.get("semantic_contract") is not None or a.get("instruction_schedule") is not None
                   for a in row["arms"].values())):
            raise ValueError("preserve bound, authored, or reviewed B5 evidence")
    result = deepcopy(batch)
    base = family_targets(batch)
    targets = dict(base)
    targets[RECEIVER] += targets[DONOR]
    targets[DONOR] = 0
    result[KEY] = {"profile_id": PROFILE, "base_targets": base, "targets": targets,
                   "omitted_families": [DONOR], "reason": reason,
                   "authorization_ref": auth_ref, "availability_evidence_refs": deepcopy(evidence_refs),
                   "formal_release_profile_approved": False, "model_evaluation_used_for_allocation": False,
                   "original_rows": deepcopy(rows)}
    for row in result["drafts"]:
        if row["task_family"] != DONOR:
            continue
        row["task_family"] = RECEIVER
        row["arms"] = {arm: {"instruction_template_not_executable": text,
                             "instruction_schedule": None, "semantic_contract": None}
                       for arm, text in zip(("R0", "R1"), TEMPLATES["B1"])}
        row["required_bindings"] = ["task_family_and_arm_mapping", "window_and_spatial_contract",
            "legal_action_and_balance", "reconstruction_support", "safe_plans_all_seeds",
            "formal_oracle_witnesses", "final_human_review"]
        row["initial_human_decision_reference"] = None
    result["family_targets"] = targets
    validate_draft_batch(result)
    return result


def omit_unbound_a6(root, batch, *, reason, evidence_refs):
    """Add an explicit A6-to-A5 draft allocation; preserve prior B5 history.

    Only untouched A6 hypotheses are reset. Existing A5 sources are not copied
    or relabeled; replacement opportunities still require independent evidence.
    """
    from .agent_drafts import TEMPLATES, validate_draft_batch
    validate_draft_batch(batch)
    if batch.get(KEY,{}).get('profile_id') != PROFILE:
        raise ValueError('one prior B5-only draft declaration required; do not apply twice')
    root=Path(root).resolve()
    auth_ref='config/task_coverage_relaxation.private.json'
    auth=load_json(root/auth_ref)
    if (auth.get('agent_may_omit_hard_to_source_task_families') is not True
        or auth.get('target_total_pcus') != 50 or auth.get('target_real_oracle_witnesses') != 300
        or auth.get('require_final_human_review_before_release') is not True
        or auth.get('missing_evidence_may_be_marked_passed') is not False
        or auth.get('relabel_existing_pcus_as_other_task_families') is not False
        or A6_DONOR not in auth.get('immediate_search_pauses',[])):
        raise ValueError('explicit current A6 omission authority required')
    if not isinstance(reason,str) or not reason.strip() or not isinstance(evidence_refs,list) or not evidence_refs:
        raise ValueError('reason and current availability evidence required')
    for ref in evidence_refs:
        if not isinstance(ref,str) or not ref.strip():
            raise ValueError('existing project-local availability evidence required')
        path=(root/ref).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('existing project-local availability evidence required')
    rows=[r for r in batch['drafts'] if r['task_family']==A6_DONOR]
    forbidden={'authored_pcu_path','exploratory_evidence','agent_pilot_evidence','agent_branch_pilot_run',
               'map_bound_contract_draft','agent_visual_observations','visual_grounding_lead','visual_observations'}
    if not rows:
        raise ValueError('untouched A6 slots required')
    for row in rows:
        if (row.get('family_binding_status') != 'unverified_hypothesis'
            or row.get('grounding_bindings') != {'R0':None,'R1':None} or forbidden.intersection(row)
            or any(a.get('semantic_contract') is not None or a.get('instruction_schedule') is not None
                   for a in row['arms'].values())):
            raise ValueError('preserve bound, authored or reviewed A6 evidence')
    result=deepcopy(batch)
    targets=draft_family_targets(batch)
    targets[A5_RECEIVER] += targets[A6_DONOR]
    targets[A6_DONOR]=0
    result[KEY]=dict(profile_id=A6_PROFILE,base_targets=deepcopy(batch[KEY]['base_targets']),targets=targets,
        omitted_families=[A6_DONOR,DONOR],reason=reason,authorization_ref=auth_ref,
        availability_evidence_refs=deepcopy(evidence_refs),formal_release_profile_approved=False,
        model_evaluation_used_for_allocation=False,original_rows=deepcopy(rows),previous_declaration=deepcopy(batch[KEY]))
    for row in result['drafts']:
        if row['task_family'] != A6_DONOR:
            continue
        row['task_family']=A5_RECEIVER
        row['arms']={arm:dict(instruction_template_not_executable=text,instruction_schedule=None,semantic_contract=None)
                     for arm,text in zip(('R0','R1'),TEMPLATES['A5'])}
        row['required_bindings']=['task_family_and_arm_mapping','window_and_spatial_contract','legal_action_and_balance',
            'reconstruction_support','safe_plans_all_seeds','formal_oracle_witnesses','final_human_review']
        row['initial_human_decision_reference']=None
    result['family_targets']=targets
    validate_draft_batch(result)
    return result
