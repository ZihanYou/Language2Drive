"""Bounded, per-call actor polygon memoization; official collision logic intact.

This is a standalone diagnostic adapter, not enabled in the runtime or formal
bridge. No file/model hashes, trajectory approximations, changed horizons,
out-of-range interpolation or cross-rollout caches are introduced.
"""
from functools import lru_cache
from importlib.metadata import version
from types import SimpleNamespace


def calculate_cached_open_loop(simulation_result, config, max_entries=8192):
    from eval.scorers.open_loop_collision import OpenLoopCollisionScorer

    if version("alpasim_eval") != "1.36.0":
        raise ValueError("review adapter against the installed evaluator before use")
    if type(max_entries) is not int or not 1 <= max_entries <= 8192:
        raise ValueError("require a bounded cache of 1 to 8192 actor polygons")
    trajectories = simulation_result.actor_trajectories

    @lru_cache(maxsize=max_entries)
    def polygon(actor_id, timestamp_us):
        return trajectories[actor_id].get_polygon_at_time(timestamp_us)

    class ActorView:
        def __init__(self, actor_id, trajectory):
            self.actor_id = actor_id
            self.trajectory = trajectory

        @property
        def time_range_us(self):
            return self.trajectory.time_range_us

        def get_polygon_at_time(self, timestamp_us):
            return polygon(self.actor_id, timestamp_us)

    view = SimpleNamespace(
        actor_trajectories={key: ActorView(key, value) for key, value in trajectories.items()},
        driver_responses=simulation_result.driver_responses,
    )
    try:
        metrics = OpenLoopCollisionScorer(config).calculate(view)
        info = polygon.cache_info()
        return metrics, {"scope": "exact_actor_polygon_reuse_not_qualification", "qualified": False,
                         "evaluator_version": "1.36.0", "official_scorer_logic_unchanged": True,
                         "cache_hits": info.hits, "cache_misses": info.misses,
                         "peak_bound_entries": info.maxsize, "final_entries": info.currsize,
                         "cross_call_cache": False, "runtime_enabled": False}
    finally:
        polygon.cache_clear()
