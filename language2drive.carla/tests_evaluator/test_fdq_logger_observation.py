from __future__ import annotations

from types import SimpleNamespace

from leaderboard.utils.closed_loop_logger import ClosedLoopLogger


class _Vector(SimpleNamespace):
    pass


class _Transform:
    def __init__(self, x: float, y: float):
        self.location = _Vector(x=x, y=y, z=0.0)
        self.rotation = _Vector(pitch=0.0, yaw=0.0, roll=0.0)


class _Actor:
    def __init__(
        self, instance_id: int, actor_type: str, x: float, vx: float,
        *, y: float = 0.0, vy: float = 0.0,
    ):
        self.id = instance_id
        self.type_id = actor_type
        self._transform = _Transform(x, y)
        self._velocity = _Vector(x=vx, y=vy, z=0.0)
        self.bounding_box = SimpleNamespace(extent=_Vector(x=1.0, y=0.5, z=0.5))

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return self._velocity


def test_fdq_logger_captures_compact_ttc_and_route_progress(tmp_path):
    logger = ClosedLoopLogger(
        str(tmp_path),
        {"waypoints": [{"x": 0, "y": 0}, {"x": 100, "y": 0}]},
        workload="fdq",
    )
    ego = _Actor(1, "vehicle.ego", 25.0, 10.0)
    lead = _Actor(2, "vehicle.lead", 45.0, 0.0)
    observation = logger._collect_fdq_observation(
        ego, ego.get_transform(), ego.get_velocity(), [ego, lead]
    )
    assert observation["schema_version"] == "language2drive.fdq_observation.v1"
    assert observation["route_progress"] == 0.25
    assert observation["relevant_actor_count"] == 1
    assert observation["min_ttc_seconds"] is not None
    assert 1.7 < observation["min_ttc_seconds"] < 1.8
    assert observation["min_ttc_actor_instance_id"] == 2
    report = logger._capture_report()
    assert report["fdq_metric_observation"]["frame_samples"] == 1
    assert report["fdq_metric_observation"]["ttc_finite_samples"] == 1
    assert report["fdq_metric_observation"]["progress_supported"] is True


def test_fdq_ttc_does_not_treat_adjacent_lane_as_overlap(tmp_path):
    logger = ClosedLoopLogger(
        str(tmp_path),
        {"waypoints": [{"x": 0, "y": 0}, {"x": 100, "y": 0}]},
        workload="fdq",
    )
    ego = _Actor(1, "vehicle.ego", 25.0, 10.0)
    adjacent = _Actor(2, "vehicle.adjacent", 25.0, 10.0, y=4.0)
    observation = logger._collect_fdq_observation(
        ego, ego.get_transform(), ego.get_velocity(), [ego, adjacent]
    )
    assert observation["relevant_actor_count"] == 1
    assert observation["min_ttc_seconds"] is None


def test_fdq_ttc_excludes_distinct_carla_proxy_for_same_ego_id(tmp_path):
    logger = ClosedLoopLogger(
        str(tmp_path),
        {"waypoints": [{"x": 0, "y": 0}, {"x": 100, "y": 0}]},
        workload="fdq",
    )
    ego = _Actor(1, "vehicle.ego", 25.0, 10.0)
    proxy = _Actor(1, "vehicle.ego", 25.0, 10.0)
    observation = logger._collect_fdq_observation(
        ego, ego.get_transform(), ego.get_velocity(), [proxy]
    )
    assert observation["relevant_actor_count"] == 0
    assert observation["min_ttc_seconds"] is None
