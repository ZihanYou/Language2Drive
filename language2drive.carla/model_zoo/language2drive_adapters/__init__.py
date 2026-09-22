"""Model-local Language2Drive adapters (no environment installation needed)."""

__all__ = ["Language2DriveTrajectoryAgent"]


def __getattr__(name):
    if name == "Language2DriveTrajectoryAgent":
        from .trajectory_agent import Language2DriveTrajectoryAgent

        return Language2DriveTrajectoryAgent
    raise AttributeError(name)
