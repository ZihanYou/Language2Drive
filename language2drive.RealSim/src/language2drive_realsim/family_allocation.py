"""Default family quotas, specification fallbacks and explicit revised cohorts."""
from .constants import FAMILY_TARGETS, LAYER_TARGETS
from .release_profile import reduced_release_profile


def family_targets(source):
    """Keep defaults unless an evidenced, pre-model allocation is declared.

    Size revisions need a separately confirmed release profile. Neither they
    nor family fallbacks change task definitions, visibility or oracle gates.
    In particular an A task cannot become C by changing its label.
    The declaration enters the normal frozen source hash and provenance.
    """
    if "authoring_coverage_reallocation" in source:
        raise ValueError("draft-only coverage adjustment is not an approved formal release profile")
    profile = reduced_release_profile(source)
    if profile is not None:
        return profile["family_targets"]
    targets = dict(FAMILY_TARGETS)
    b4 = source.get("b4_family_reallocation")
    if b4 is not None:
        families_b = {f for f in targets if f.startswith("B")}
        original_b4 = "B4_object_relative_passing"
        if not isinstance(b4, dict):
            raise ValueError("B4 fallback must be an explicit declaration")
        counts_b = b4.get("targets")
        if (not isinstance(counts_b, dict) or set(counts_b) != families_b
            or any(type(n) is not int or n < 0 for n in counts_b.values())
            or sum(counts_b.values()) != LAYER_TARGETS["B"]
            or not 0 <= counts_b[original_b4] < FAMILY_TARGETS[original_b4]
            or any(counts_b[f] < FAMILY_TARGETS[f] for f in families_b if f != original_b4)):
            raise ValueError("B4 fallback may replace only one or two B4 slots within B; retain all other quotas")
        reason_b = b4.get("reason")
        evidence_b = b4.get("b4_infeasibility_evidence_refs")
        conditions = b4.get("unmet_b4_conditions")
        permitted = {"two_legal_sides", "valid_reconstruction", "paired_safe_oracle"}
        if (not isinstance(reason_b, str) or not reason_b.strip()
            or not isinstance(evidence_b, list) or not evidence_b
            or any(not isinstance(ref, str) or not ref.strip() for ref in evidence_b)
            or len(evidence_b) != len(set(evidence_b))
            or not isinstance(conditions, list) or not conditions
            or any(not isinstance(c, str) or c not in permitted for c in conditions)
            or len(conditions) != len(set(conditions))
            or b4.get("frozen_before_model_evaluation") is not True):
            raise ValueError("B4 fallback requires unmet specification conditions, evidence, reason and pre-model freeze")
        targets.update(counts_b)
    declaration = source.get("c_family_reallocation")
    if declaration is None:
        return targets
    families = {f for f in targets if f.startswith("C")}
    if not isinstance(declaration, dict):
        raise ValueError("C family reallocation must be an explicit declaration")
    counts = declaration.get("targets")
    if (not isinstance(counts, dict) or set(counts) != families
        or any(type(n) is not int or n < 0 for n in counts.values())
        or sum(counts.values()) != LAYER_TARGETS["C"]):
        raise ValueError("C reallocation must preserve exactly 12 C PCUs and cannot change A/B families")
    reason = declaration.get("reason")
    evidence = declaration.get("asset_availability_evidence_refs")
    if (not isinstance(reason, str) or not reason.strip()
        or not isinstance(evidence, list) or not evidence
        or any(not isinstance(ref, str) or not ref.strip() for ref in evidence)
        or len(evidence) != len(set(evidence))
        or declaration.get("frozen_before_model_evaluation") is not True):
        raise ValueError("C reallocation requires asset-availability evidence, reason and pre-model freeze")
    targets.update(counts)
    return targets
