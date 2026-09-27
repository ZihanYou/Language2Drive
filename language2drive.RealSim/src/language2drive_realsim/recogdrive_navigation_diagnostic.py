"""Opt-in task-text navigation diagnostic for ReCogDrive NAVSIM-transfer.

This module deliberately uses only the trusted instruction text.  It does not
inspect routes, target points, scene labels, simulator state, or ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


PROFILE = "task-text-navigation-20260917"
_LEGACY_COMMAND = "3. Active navigation command: [GO STRAIGHT]"

_COMMANDS = {
    "TURN LEFT": (1.0, 0.0, 0.0, 0.0),
    "GO STRAIGHT": (0.0, 1.0, 0.0, 0.0),
    "TURN RIGHT": (0.0, 0.0, 1.0, 0.0),
}

# These cover the frozen benchmark's actual route wording and its symmetric
# left-turn forms.  They are intentionally anchored so that, for example,
# "a sign on the right" cannot become a turn command.
_RIGHT = tuple(
    re.compile(pattern)
    for pattern in (
        r"turn right(?: into the next entrance| at the upcoming intersection)?",
        r"make a right turn(?: at the intersection ahead)?",
        r"take the upcoming right-hand entrance",
        r"use the next entrance on your right",
        r"at the next intersection,? take the road to the right",
        r"go right",
    )
)
_LEFT = tuple(
    re.compile(pattern)
    for pattern in (
        r"turn left(?: into the next entrance| at the upcoming intersection)?",
        r"make a left turn(?: at the intersection ahead)?",
        r"take the upcoming left-hand entrance",
        r"use the next entrance on your left",
        r"at the next intersection,? take the road to the left",
        r"go left",
    )
)
_STRAIGHT = tuple(
    re.compile(pattern)
    for pattern in (
        r"continue straight on the current road",
        r"keep going straight along this road",
        r"follow the current road straight ahead",
        r"go straight",
        r"continue straight",
    )
)
_LANE_LANGUAGE = re.compile(
    r"(?:\b(?:change|switch|move)\b.*\blane\b|\badjacent lane\b|"
    r"\b(?:left|right) lane\b|\blane immediately\b)"
)
_CONDITIONAL_OR_SEQUENCE = re.compile(
    r"\b(?:if|unless|until|while|after|before|once|then|followed by)\b"
)


@dataclass(frozen=True)
class NavigationSelection:
    """Auditable command selected from task text alone."""

    profile: str
    label: str
    command: tuple[float, float, float, float]
    explicit: bool
    reason: str


def _fallback(reason: str) -> NavigationSelection:
    return NavigationSelection(
        profile=PROFILE,
        label="GO STRAIGHT",
        command=_COMMANDS["GO STRAIGHT"],
        explicit=False,
        reason=reason,
    )


def _matches(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(pattern.fullmatch(text) is not None for pattern in patterns)


def select_navigation(instruction: str) -> NavigationSelection:
    """Select a NAVSIM command only when *instruction* is unambiguous.

    Unknown, lane-change, conditional, compound, and conflicting text keeps
    the legacy straight prior and is marked as a non-explicit fallback.
    """

    if not isinstance(instruction, str):
        raise TypeError("instruction must be a string")
    text = " ".join(instruction.strip().casefold().split()).rstrip(".!?").strip()
    if not text:
        return _fallback("empty_instruction")
    if _LANE_LANGUAGE.search(text):
        return _fallback("lane_instruction_not_route_turn")
    if _CONDITIONAL_OR_SEQUENCE.search(text):
        return _fallback("conditional_or_compound_instruction")

    matches = []
    if _matches(_LEFT, text):
        matches.append("TURN LEFT")
    if _matches(_STRAIGHT, text):
        matches.append("GO STRAIGHT")
    if _matches(_RIGHT, text):
        matches.append("TURN RIGHT")
    if len(matches) != 1:
        return _fallback("ambiguous_or_unsupported_instruction")
    label = matches[0]
    return NavigationSelection(
        profile=PROFILE,
        label=label,
        command=_COMMANDS[label],
        explicit=True,
        reason="explicit_task_text_command",
    )


def rewrite_navigation_prompt(
    legacy_prompt: str, selection: NavigationSelection
) -> str:
    """Replace exactly the frozen prompt's single legacy command statement."""

    if not isinstance(legacy_prompt, str):
        raise TypeError("legacy_prompt must be a string")
    if not isinstance(selection, NavigationSelection):
        raise TypeError("selection must be a NavigationSelection")
    if selection.profile != PROFILE:
        raise ValueError("navigation selection has the wrong diagnostic profile")
    if selection.label not in _COMMANDS or selection.command != _COMMANDS[selection.label]:
        raise ValueError("navigation label and command one-hot disagree")
    if legacy_prompt.count(_LEGACY_COMMAND) != 1:
        raise ValueError("legacy prompt must contain exactly one frozen command statement")
    replacement = f"3. Active navigation command: [{selection.label}]"
    return legacy_prompt.replace(_LEGACY_COMMAND, replacement, 1)
