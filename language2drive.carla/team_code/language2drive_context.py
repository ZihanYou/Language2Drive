"""Small, dependency-light Language2Drive evaluator/agent contract.

The standardized track deliberately exposes only the instruction and protocol
version. Route/run identity and random seeds remain evaluator-owned because a
route entry id can be joined with the public XML to recover the counterfactual
arm. In particular, the Standardized payload must not grow fields
for a target point, dense route, lane, gate, polygon, or oracle trajectory.
Those belong in evaluator-only sidecars.

This module enforces an evaluator API boundary, not operating-system isolation:
Bench2Drive loads evaluator and submitted agent code in one trusted process.
Adversarial-agent secrecy therefore requires a future process/container boundary.
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Optional


PUBLIC_CONTEXT_SCHEMA = "language2drive.public_context.v1"
PRIVILEGED_ORACLE_CAPABILITY = "language2drive.privileged_oracle.v1"
L2D_TRACK_LEGACY = "legacy"
L2D_TRACK_STANDARDIZED = "standardized"
L2D_TRACK_NATIVE = "native"
VALID_L2D_TRACKS = frozenset(
    {L2D_TRACK_LEGACY, L2D_TRACK_STANDARDIZED, L2D_TRACK_NATIVE}
)


class PublicContextError(ValueError):
    """Raised when an evaluator tries to expose an invalid public context."""


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint is too incomplete for a valid evaluation."""


def bind_privileged_oracle_context(
    agent: Any,
    private_context: Optional[Mapping[str, Any]],
    track: str,
) -> bool:
    """Bind evaluator-private state only to an explicitly privileged oracle.

    Native-System agents normally receive a dense global plan but never receive
    the evaluator sidecar.  A witness oracle is the sole exception: it must opt
    in with an exact, versioned class capability and implement a dedicated
    setter.  Standardized evaluation fails *before* invoking that setter, so an
    accidentally selected oracle cannot turn into a sidecar disclosure.

    This remains an in-process engineering boundary, not a sandbox against a
    malicious agent; the trusted-oracle capability is intentionally narrow and
    auditable.
    """

    capability = getattr(
        agent, "language2drive_private_context_capability", None
    )
    if capability != PRIVILEGED_ORACLE_CAPABILITY:
        return False
    normalized_track = normalize_l2d_track(track)
    if normalized_track != L2D_TRACK_NATIVE:
        raise PublicContextError(
            "privileged oracle private context is allowed only on the native track"
        )
    if not isinstance(private_context, Mapping) or not isinstance(
        private_context.get("route"), Mapping
    ):
        raise PublicContextError(
            "privileged oracle requires one evaluator-loaded private route context"
        )
    setter = getattr(agent, "set_language2drive_private_context", None)
    if not callable(setter):
        raise PublicContextError(
            "privileged oracle capability requires "
            "set_language2drive_private_context(private_context)"
        )
    setter(copy.deepcopy(private_context))
    return True


def normalize_l2d_track(value: Optional[str], strict_vln: bool = False) -> str:
    """Normalize the public track name while retaining the old strict flag.

    ``L2D_STRICT_VLN=1`` is a backwards-compatible alias for the standardized
    track only when no explicit CLI track was supplied.
    """

    if value is None or not str(value).strip():
        return L2D_TRACK_STANDARDIZED if strict_vln else L2D_TRACK_LEGACY
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "off": L2D_TRACK_LEGACY,
        "v0": L2D_TRACK_LEGACY,
        "strict": L2D_TRACK_STANDARDIZED,
        "standard": L2D_TRACK_STANDARDIZED,
        "native_system": L2D_TRACK_NATIVE,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALID_L2D_TRACKS:
        raise PublicContextError(
            "Unknown Language2Drive track {!r}; expected one of {}".format(
                value, ", ".join(sorted(VALID_L2D_TRACKS))
            )
        )
    return normalized


@dataclass(frozen=True)
class Language2DrivePublicContext:
    """The complete information intentionally visible to an L2D agent."""

    instruction: str
    track: str = L2D_TRACK_STANDARDIZED
    instruction_version: int = 0
    schema_version: str = PUBLIC_CONTEXT_SCHEMA
    route_name: Optional[str] = None
    seed: Optional[int] = None

    @classmethod
    def from_mapping(
        cls, context: Mapping[str, Any]
    ) -> "Language2DrivePublicContext":
        if not isinstance(context, Mapping):
            raise PublicContextError("public_context must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(context) - allowed)
        if unknown:
            raise PublicContextError(
                "public_context contains non-public or unsupported fields: {}".format(
                    ", ".join(unknown)
                )
            )

        instruction = str(context.get("instruction", "")).strip()
        track = normalize_l2d_track(context.get("track"))
        route_name_value = context.get("route_name")
        route_name = (
            None
            if route_name_value is None
            else str(route_name_value).strip() or None
        )
        seed_value = context.get("seed")
        try:
            instruction_version = int(context.get("instruction_version", 0))
            seed = None if seed_value is None else int(seed_value)
        except (TypeError, ValueError) as exc:
            raise PublicContextError(
                "seed (when present) and instruction_version must be integers"
            ) from exc

        forbidden_standardized = sorted(
            {"route_name", "seed"}.intersection(context)
        )
        if track == L2D_TRACK_STANDARDIZED and forbidden_standardized:
            raise PublicContextError(
                "standardized public_context contains evaluator-owned fields: {}".format(
                    ", ".join(forbidden_standardized)
                )
            )
        if track != L2D_TRACK_STANDARDIZED and not route_name:
            raise PublicContextError("public_context.route_name must be non-empty")
        if instruction_version < 0:
            raise PublicContextError("instruction_version must be non-negative")
        if track == L2D_TRACK_STANDARDIZED and not instruction:
            raise PublicContextError(
                "standardized track requires a non-empty instruction"
            )
        schema_version = str(
            context.get("schema_version", PUBLIC_CONTEXT_SCHEMA)
        ).strip()
        if schema_version != PUBLIC_CONTEXT_SCHEMA:
            raise PublicContextError(
                "unsupported public_context schema {!r}".format(schema_version)
            )
        return cls(
            instruction=instruction,
            route_name=route_name,
            seed=seed,
            track=track,
            instruction_version=instruction_version,
            schema_version=schema_version,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Standardized key inspection cannot recover route identity.
        return {key: value for key, value in payload.items() if value is not None}


class PublicContextState:
    """Single-route context holder which overwrites, never merges, state."""

    def __init__(self) -> None:
        self._context: Optional[Language2DrivePublicContext] = None

    def bind(
        self, context: Mapping[str, Any]
    ) -> Language2DrivePublicContext:
        # A deep copy prevents the evaluator from mutating the bound context and
        # prevents stale fields from a previous route surviving a dictionary merge.
        parsed = Language2DrivePublicContext.from_mapping(copy.deepcopy(context))
        self._context = parsed
        return parsed

    @property
    def context(self) -> Optional[Language2DrivePublicContext]:
        return self._context


def make_public_context(
    *, instruction: Optional[str], route_name: str, seed: int, track: str
) -> Dict[str, Any]:
    """Create and validate the evaluator-to-agent payload for one route.

    Standardized discards route_name and seed. The evaluator retains them for
    private sidecar lookup, ledgers, and deterministic CARLA setup; neither is
    an agent input.
    """

    normalized_track = normalize_l2d_track(track)
    payload = {
        "schema_version": PUBLIC_CONTEXT_SCHEMA,
        "track": normalized_track,
        "instruction": instruction or "",
        "instruction_version": 0,
    }
    if normalized_track != L2D_TRACK_STANDARDIZED:
        payload.update({"route_name": route_name, "seed": seed})
    return Language2DrivePublicContext.from_mapping(payload).to_dict()


def seed_everything(seed: int, include_torch: bool = True) -> Dict[str, Any]:
    """Seed Python, NumPy, and (when available) Torch deterministically."""

    normalized_seed = int(seed)
    # CPython reads PYTHONHASHSEED only at process start, so changing it to the
    # per-rollout CARLA seed here cannot affect this process. It can, however,
    # leak that private seed to same-process agent code and child processes.
    # Keep hash randomization deterministic with one arm-independent constant.
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(normalized_seed)
    report: Dict[str, Any] = {
        "seed": normalized_seed,
        "python": True,
        "python_hash_seed": 0,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }

    try:
        import numpy as np

        np.random.seed(normalized_seed)
        report["numpy"] = True
    except ImportError:
        report["numpy"] = False

    if include_torch:
        try:
            import torch

            torch.manual_seed(normalized_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(normalized_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            report["torch"] = True
            report["cuda"] = bool(torch.cuda.is_available())
        except ImportError:
            report["torch"] = False
            report["cuda"] = False
    return report


def _numel(value: Any) -> int:
    numel = getattr(value, "numel", None)
    return int(numel()) if callable(numel) else 0


def _is_critical_action_key(key: str) -> bool:
    """Identify parameters that implement AutoVLA's action-token interface."""

    return (
        key.endswith("embed_tokens.weight")
        or key.endswith("lm_head.weight")
        or "action_tokenizer" in key
        or "planner" in key
        or "action_head" in key
    )


def validate_checkpoint_coverage(
    model_state: Mapping[str, Any],
    checkpoint_state: Mapping[str, Any],
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    *,
    minimum_coverage: float = 0.98,
) -> Dict[str, Any]:
    """Return a serializable load report or fail on an invalid partial load.

    Coverage is parameter-element weighted, not merely a key count.  A missing
    action-token embedding/output head is always fatal even if the very large
    vision/language backbone otherwise makes total coverage look high.
    """

    if not 0.0 < float(minimum_coverage) <= 1.0:
        raise ValueError("minimum_coverage must be in (0, 1]")
    if not model_state:
        raise CheckpointCompatibilityError("model has an empty state_dict")
    if not checkpoint_state:
        raise CheckpointCompatibilityError("checkpoint has an empty state_dict")

    missing = tuple(sorted(set(str(key) for key in missing_keys)))
    unexpected = tuple(sorted(set(str(key) for key in unexpected_keys)))
    loaded_keys = tuple(sorted(set(model_state).intersection(checkpoint_state)))
    total_elements = sum(_numel(value) for value in model_state.values())
    loaded_elements = sum(_numel(model_state[key]) for key in loaded_keys)
    if total_elements:
        coverage = loaded_elements / total_elements
    else:
        coverage = len(loaded_keys) / max(len(model_state), 1)

    critical_model_keys = tuple(
        sorted(key for key in model_state if _is_critical_action_key(key))
    )
    missing_critical = tuple(
        key for key in critical_model_keys if key not in checkpoint_state
    )
    report = {
        "coverage": round(float(coverage), 8),
        "minimum_coverage": float(minimum_coverage),
        "model_key_count": len(model_state),
        "loaded_key_count": len(loaded_keys),
        "model_parameter_elements": total_elements,
        "loaded_parameter_elements": loaded_elements,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "critical_keys": list(critical_model_keys),
        "missing_critical_keys": list(missing_critical),
    }
    failures = []
    if coverage < float(minimum_coverage):
        failures.append(
            "parameter coverage {:.6f} is below {:.6f}".format(
                coverage, float(minimum_coverage)
            )
        )
    if missing_critical:
        failures.append(
            "missing critical planner/action parameters: {}".format(
                ", ".join(missing_critical)
            )
        )
    if failures:
        raise CheckpointCompatibilityError("; ".join(failures))
    return report
