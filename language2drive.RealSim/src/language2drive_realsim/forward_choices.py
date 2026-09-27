"""Explicit private planning choices, never inferred navigation or map edits."""
from copy import deepcopy


def validate_grouped_forward_choices(value, groups):
    """Absent legacy choices stay empty; present groups must be complete."""
    if value is None:
        return {group: {} for group in groups}
    if not isinstance(value, dict) or set(value) != set(groups):
        raise ValueError("exact explicit forward-choice groups required")
    for choices in value.values():
        if (not isinstance(choices, dict) or len(choices) > 8
            or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v
                   or k == v or k == "-1" or v == "-1" for k, v in choices.items())):
            raise ValueError("at most eight valid explicit forward choices per group required")
    return deepcopy(value)


def parse_forward_choice_items(items):
    pairs = [item.split("=") for item in items]
    if any(len(p) != 2 for p in pairs) or len({p[0] for p in pairs}) != len(pairs):
        raise ValueError("distinct FROM=TO forward choices required")
    return validate_grouped_forward_choices({"path": dict(pairs)}, ("path",))["path"]
