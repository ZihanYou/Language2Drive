"""Explicit host-runtime entry point for the observed-velocity handoff profile."""

from __future__ import annotations

from . import handoff_observed_velocity, host_runtime


def main() -> None:
    handoff_observed_velocity.install()
    try:
        host_runtime.main()
    finally:
        handoff_observed_velocity.uninstall()


if __name__ == "__main__":
    main()
