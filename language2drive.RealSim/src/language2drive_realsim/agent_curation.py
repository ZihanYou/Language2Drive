"""Explicit user-delegated pilot authority, never a fabricated human approval."""
from copy import deepcopy
from pathlib import Path

from .io import load_json
from .search_input import load_search_input

AUTHORITY = "agent_under_user_delegation_pending_final_review"


def load_agent_candidate(root, candidate_path):
    root, path = Path(root).resolve(), Path(candidate_path).resolve()
    if not path.is_relative_to(root / ".runtime"):
        raise ValueError("agent input must be project-local offline evidence")
    policy = load_json(root / "config/curation_workflow.private.json")
    if (policy.get("mode") != "agent_authoring_then_final_human_review"
        or policy.get("authorization", {}).get("source") != "user_chat"
        or policy.get("agent_may_screen_and_prepare_experiments") is not True
        or policy.get("require_final_human_review_before_release") is not True
        or policy.get("missing_evidence_may_be_marked_passed") is not False):
        raise ValueError("explicit user-delegated agent pilot authority required")
    candidate, offline = load_search_input(path.parent, path)
    if not offline or candidate.get("input_errors"):
        raise ValueError("input-ready bound offline candidate required")
    result = deepcopy(candidate)
    result.update(curation_authority=AUTHORITY, human_reviewed=False, final_human_review="pending")
    return result


def validate_agent_runs(root, run_dirs):
    """Recheck delegated inputs before drafting from real pilot evidence."""
    root = Path(root).resolve()
    candidates = []
    for directory in run_dirs:
        directory = Path(directory).resolve()
        if not directory.is_relative_to(root / ".runtime"):
            raise ValueError("agent run must be project-local")
        launch = load_json(directory / "launch.private.json")
        candidate = load_agent_candidate(root, launch["agent_candidate_source"])
        if (candidate != launch["candidate"] or launch.get("curation_authority") != AUTHORITY
            or launch.get("final_human_review_required") is not True):
            raise ValueError("agent launch no longer matches delegated input")
        candidates.append(candidate)
    if not candidates or any(c != candidates[0] for c in candidates):
        raise ValueError("one identical delegated candidate required")
    return candidates[0]
