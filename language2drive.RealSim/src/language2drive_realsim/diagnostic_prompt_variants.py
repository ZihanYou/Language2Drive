"""Explicit diagnostic prompt variants; not enabled in any existing worker.

Using a variant requires a separately identified adapter release/experiment.
No score, trajectory parser, checkpoint or legacy prompt is changed here.
"""

IMPROMPTU_SIX_POINT_PROFILE = "impromptu-six-point-3s-20260917"


def impromptu_six_point_prompt(legacy_prompt: str) -> str:
    """Resolve the known opening-horizon contradiction, changing nothing else.

    Fail closed if the caller supplies a different prompt contract. The caller
    must first use its existing prompt builder to validate state/instruction.
    """
    opening = "over the next 10 timesteps, given the vehicle's intent"
    closing = "Return the six future waypoints for the next 3 seconds, sampled at "
    if (
        not isinstance(legacy_prompt, str)
        or legacy_prompt.count(opening) != 1
        or legacy_prompt.count("10 timesteps") != 1
        or legacy_prompt.count(closing) != 1
        or "0.5-second intervals, in meters inside <PLANNING> and </PLANNING>."
        not in legacy_prompt
    ):
        raise ValueError("unexpected legacy Impromptu prompt contract")
    return legacy_prompt.replace(
        opening,
        "over the next 6 timesteps (3 seconds), given the vehicle's intent",
        1,
    )
