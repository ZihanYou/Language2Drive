"""Explicit cohort-size revisions; never infer an allocation from available data."""
from copy import deepcopy

from .constants import ARMS, COUPLED_SEEDS, FAMILY_TARGETS, LAYER_TARGETS

REDUCED_PROFILE_ID = "l2d.realsim.release.30.v0.1"


def reduced_release_profile(source):
    """Validate the trusted source's approved, pre-model 30-PCU declaration.

    This is a declaration validator, not an authorization service. Callers must
    only author the declaration after the user confirms its actual allocation.
    A pending scope request is deliberately not an executable release profile.
    """
    if "release_profile" not in source:
        return None
    p = source["release_profile"]
    if not isinstance(p, dict) or p.get("profile_id") != REDUCED_PROFILE_ID:
        raise ValueError("explicit supported 30-PCU release profile required")
    if any(k in source for k in ("b4_family_reallocation", "c_family_reallocation", "authoring_coverage_reallocation")):
        raise ValueError("reduced release profile cannot mix legacy allocation declarations")
    layers, families = p.get("layer_targets"), p.get("family_targets")
    if (type(p.get("total_pcus")) is not int or p["total_pcus"] != 30
        or not isinstance(layers, dict) or set(layers) != set(LAYER_TARGETS)
        or any(type(n) is not int or n < 0 for n in layers.values()) or sum(layers.values()) != 30
        or not isinstance(families, dict) or set(families) != set(FAMILY_TARGETS)
        or any(type(n) is not int or n < 0 for n in families.values())
        or any(sum(n for f, n in families.items() if f.startswith(layer)) != count for layer, count in layers.items())):
        raise ValueError("30-PCU profile needs complete consistent integer layer and family quotas")
    if (p.get("coupled_seeds") != list(COUPLED_SEEDS)
        or any(type(s) is not int for s in p.get("coupled_seeds", []))
        or p.get("arms") != list(ARMS)
        or type(p.get("oracle_witnesses")) is not int or p["oracle_witnesses"] != 30 * len(ARMS) * len(COUPLED_SEEDS)):
        raise ValueError("30-PCU profile must retain both arms and all three coupled seeds: 180 witnesses")
    auth = p.get("authorization")
    refs = p.get("allocation_evidence_refs")
    omitted = sorted(layer for layer, count in layers.items() if count == 0)
    if (not isinstance(auth, dict) or auth.get("source") != "user_chat"
        or auth.get("layer_and_family_allocation_confirmed") is not True
        or not isinstance(auth.get("instruction_ref"), str) or not auth["instruction_ref"].strip()
        or p.get("omitted_layers") != omitted
        or not isinstance(p.get("reason"), str) or not p["reason"].strip()
        or not isinstance(refs, list) or not refs
        or any(not isinstance(ref, str) or not ref.strip() for ref in refs) or len(set(refs)) != len(refs)
        or p.get("frozen_before_model_evaluation") is not True
        or p.get("model_results_used_for_allocation") is not False
        or p.get("qualification_requirements_unchanged") is not True
        or p.get("final_human_review_required") is not True):
        raise ValueError("reduced profile needs confirmed allocation, declared omissions and pre-model evidence; pending authority is insufficient")
    return deepcopy(p)


def release_layer_targets(source):
    profile = reduced_release_profile(source)
    return dict(LAYER_TARGETS) if profile is None else profile["layer_targets"]
