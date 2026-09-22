"""
Language2Drive Closed-Loop Evaluation Logger.

Writes per-frame JSONL data for computing instruction-following and
navigation metrics without modifying Bench2Drive leaderboard scoring.
"""
import json
import math
import os
import sys
import tempfile
import traceback
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

try:
    import carla
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    _CARLA_AVAILABLE = True
except ImportError:
    carla = None
    _CARLA_AVAILABLE = False


def _v3(v) -> Dict[str, float]:
    # 将carla.Vector3D转换为字典，保留4位小数, 然后存到jsonl中
    return {'x': round(float(v.x), 4), 'y': round(float(v.y), 4), 'z': round(float(v.z), 4)}


def _rot(r) -> Dict[str, float]:
    # 将carla.Rotation转换为字典，保留4位小数, 然后存到jsonl中
    return {'pitch': round(float(r.pitch), 4), 'yaw': round(float(r.yaw), 4), 'roll': round(float(r.roll), 4)}


def _dist3d(a, b) -> float:
    # 计算两个点之间的欧氏距离
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _speed(vector) -> float:
    return math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)


def _point_segment_distance(point, start, end) -> float:
    x, y = point
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    ratio = 0.0 if denominator == 0 else (
        (x - ax) * dx + (y - ay) * dy
    ) / denominator
    ratio = max(0.0, min(1.0, ratio))
    return math.hypot(x - (ax + ratio * dx), y - (ay + ratio * dy))


def _point_in_polygon(
    point: Tuple[float, float], polygon, tolerance: float = 0.0
) -> bool:
    if not isinstance(polygon, list) or len(polygon) < 3:
        return False
    x, y = point
    inside = False
    vertices = polygon[:-1] if polygon[0] == polygon[-1] else polygon
    if any(
        _point_segment_distance(point, vertex, vertices[(index + 1) % len(vertices)])
        <= max(1e-9, float(tolerance))
        for index, vertex in enumerate(vertices)
    ):
        return True
    j = len(vertices) - 1
    for i, vertex in enumerate(vertices):
        try:
            xi, yi = float(vertex[0]), float(vertex[1])
            xj, yj = float(vertices[j][0]), float(vertices[j][1])
        except (TypeError, ValueError, IndexError):
            return False
        if (yi > y) != (yj > y):
            intersection = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x <= intersection:
                inside = not inside
        j = i
    return inside


def _inside_gate(point: Tuple[float, float], definition: Mapping[str, Any]) -> bool:
    polygon = definition.get('polygon')
    if isinstance(polygon, list):
        return _point_in_polygon(point, polygon)
    center = definition.get('center')
    radius = definition.get('radius_m')
    if isinstance(center, Mapping) and radius is not None:
        try:
            return math.hypot(
                point[0] - float(center['x']), point[1] - float(center['y'])
            ) <= float(radius) + 1e-9
        except (KeyError, TypeError, ValueError):
            return False
    return False


def _angle_error_degrees(value: float, target: float) -> float:
    return abs((float(value) - float(target) + 180.0) % 360.0 - 180.0)


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


_AUDITED_EXPLICIT_PARKING_CONTROLLER_SOURCES = frozenset({
    'privileged_oracle:parking_staging_forward',
    'privileged_oracle:parking_staging_stop',
    'privileged_oracle:parking_reverse_path',
    'privileged_oracle:parking_dwell',
})


class ClosedLoopLogger:
    """
    Per-route closed-loop evaluation logger for Language2Drive benchmark.

    Writes:
      <save_dir>/run_meta.json and route_meta.json (v0 compatibility)
      <save_dir>/trace.jsonl and frames.jsonl (v0 compatibility)
      <save_dir>/events.jsonl

    All file I/O uses a list buffer flushed every FLUSH_EVERY frames, so the
    per-frame hot path only appends to a Python list.
    """

    FLUSH_EVERY: int = 50
    AGENT_ARTIFACT_NAMES = (
        'agent_plans.jsonl',
        'checkpoint_load_report.json',
        'metric_info.json',
        'front_frames.zip',
    )

    def __init__(
        self,
        save_dir: str,
        route_meta: Dict,
        frame_interval: int = 1,
        private_context: Optional[Mapping[str, Any]] = None,
        postprocessor=None,
        capture_provenance: Optional[Mapping[str, Any]] = None,
        private_runtime=None,
        agent_artifact_dir: Optional[str] = None,
        workload: str = 'core',
    ) -> None:
        self.save_dir = save_dir
        self.route_meta = route_meta
        self.frame_interval = int(frame_interval)
        if self.frame_interval < 1:
            raise ValueError("frame_interval must be >= 1")

        self._frame_buffer: List[str] = []
        self._event_buffer: List[str] = []
        self._frame_count: int = 0

        self._frames_fh = None
        self._trace_fh = None
        self._events_fh = None
        self._closed = False
        self._logger_error_count = 0
        self._private_context = dict(private_context or {})
        # Language2Drive v1 makes trace.jsonl authoritative.  The historical
        # frames.jsonl name remains a same-inode compatibility alias created
        # only after a successful close, avoiding a second multi-gigabyte
        # stream without changing legacy logger behavior.
        public_context = route_meta.get('language2drive_public_context', {})
        self._l2d_v1_trace = (
            self._private_context.get('runtime_required') is True
            or str(route_meta.get('simlingo_metadata', {}).get(
                'benchmark_type', ''
            )).startswith('language2drive_v1')
            or (
                public_context.get('schema_version')
                == 'language2drive.public_context.v1'
                and public_context.get('track') in {'standardized', 'native'}
            )
        )
        self._postprocessor = postprocessor
        self._postprocessed = False
        self._capture_provenance = dict(capture_provenance or {})
        self._private_runtime = private_runtime
        self._agent_artifact_dir = agent_artifact_dir
        self._workload = str(workload)
        self._fdq_metrics_enabled = self._workload == 'fdq'
        self._fdq_observation_samples = 0
        self._fdq_ttc_finite_samples = 0
        self._map_legal_current: Optional[bool] = None
        self._route_progress_points: List[Tuple[float, float]] = []
        self._route_progress_cumulative: List[float] = []
        self._route_progress_total = 0.0
        if self._fdq_metrics_enabled:
            self._index_route_progress()

        self._actor_bindings: Dict[str, int] = {}
        self._observed_actor_instances: Set[int] = set()
        self._previous_actor_instances: Set[int] = set()
        self._actor_set_initialized = False
        self._criterion_events_seen: Set[Tuple[Any, ...]] = set()
        self._criterion_event_support = {
            'collision': False,
            'off_road': False,
            'red_light': False,
            # Captured and reported for diagnostics. The frozen default
            # safe_legal scope remains collision/off_road/red_light unless a
            # contract explicitly adds stop_sign to forbidden_events.
            'stop_sign': False,
        }
        self._gate_inside: Dict[str, bool] = {}
        self._dwell_started: Dict[str, Optional[float]] = {}
        self._resolved_gate_ids: Set[str] = set()
        self._resolved_dwell_ids: Set[str] = set()
        self._resolved_actor_ids: Set[str] = set()
        self._lane_samples = 0
        self._traffic_light_samples = 0
        self._map_legality_samples = 0
        self._map_illegal_previous = False
        self._controller_source_samples = 0
        self._unknown_controller_source_samples = 0
        self._gate_definitions: Dict[str, Dict[str, Any]] = {}
        self._parking_definitions: Dict[str, Dict[str, Any]] = {}
        self._actor_specs: Dict[str, Dict[str, Any]] = {}
        self._required_actor_ids: Set[str] = set()
        self._index_private_capture()

        os.makedirs(save_dir, exist_ok=True)

    def _index_route_progress(self) -> None:
        """Index the public FDQ route polyline for same-route progress curves."""

        raw = self.route_meta.get('waypoints', [])
        points: List[Tuple[float, float]] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, Mapping):
                continue
            try:
                x, y = float(item['x']), float(item['y'])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                point = (x, y)
                if not points or point != points[-1]:
                    points.append(point)
        cumulative = [0.0]
        for first, second in zip(points, points[1:]):
            cumulative.append(
                cumulative[-1] + math.hypot(
                    second[0] - first[0], second[1] - first[1]
                )
            )
        self._route_progress_points = points
        self._route_progress_cumulative = cumulative
        self._route_progress_total = cumulative[-1] if cumulative else 0.0

    def _index_private_capture(self) -> None:
        requirements = self._private_context.get('requirements', {})
        self._required_actor_ids.update(requirements.get('actor_ids', []))
        route = self._private_context.get('route', {})
        setup = route.get('actor_setup', {}) if isinstance(route, dict) else {}
        for actor in setup.get('actors', []) if isinstance(setup, dict) else []:
            if isinstance(actor, dict) and actor.get('id'):
                identifier = str(actor['id'])
                self._actor_specs[identifier] = dict(actor)
        for target in self._private_context.get('capture_targets', []):
            if not isinstance(target, dict) or not target.get('id'):
                continue
            identifier = str(target['id'])
            gate_has_geometry = (
                isinstance(target.get('polygon'), list)
                or (
                    isinstance(target.get('center'), Mapping)
                    and target.get('radius_m') is not None
                )
            )
            if target.get('type') == 'gate' and gate_has_geometry:
                self._gate_definitions[identifier] = target
                self._resolved_gate_ids.add(identifier)
            if target.get('type') == 'parking_slot' and isinstance(target.get('polygon'), list):
                self._parking_definitions[identifier] = target
                self._resolved_dwell_ids.add(identifier)
    def start_route(self) -> None:
        """Call once before the scenario tick loop begins."""
        if self._closed:
            raise RuntimeError("cannot start a closed logger")
        if self._frames_fh is not None or self._trace_fh is not None:
            raise RuntimeError("logger route already started")
        # route_meta.json只在路线开始的时候写一次
        # 保存1.route_id 2.scenario_name 3. weather_id 4. save_name 5. town_name
        try:
            public_meta = dict(self.route_meta)
            if self._capture_provenance:
                public_meta['private_capture_provenance'] = self._capture_provenance
            for name in ('run_meta.json', 'route_meta.json'):
                with open(
                    os.path.join(self.save_dir, name),
                    'w',
                    encoding='utf-8',
                ) as fh:
                    json.dump(public_meta, fh, indent=2, ensure_ascii=False)

            frames_path = os.path.join(self.save_dir, 'frames.jsonl')
            if self._l2d_v1_trace:
                try:
                    os.unlink(frames_path)
                except FileNotFoundError:
                    pass
            self._trace_fh = open(
                os.path.join(self.save_dir, 'trace.jsonl'),
                'w',
                encoding='utf-8',
            )
            if self._l2d_v1_trace:
                # A non-None sentinel preserves the existing log_frame start
                # guard; bytes are written only through _trace_fh.
                self._frames_fh = self._trace_fh
            else:
                self._frames_fh = open(
                    frames_path,
                    'w',
                    encoding='utf-8',
                )
            self._events_fh = open(
                os.path.join(self.save_dir, 'events.jsonl'),
                'w',
                encoding='utf-8',
            )
        except Exception as exc:
            self._record_error('start_route', exc)
            raise

    def log_frame(
        self,
        frame,
        timestamp,
        ego_vehicle,
        ego_action,
        scenario=None,
        statistics_manager=None,
        controller_source: Optional[str] = None,
    ) -> None:
        """
        Called every tick (or every frame_interval ticks).
        Errors are recorded and surfaced on stderr rather than silently lost.
        """
        if self._frames_fh is None:
            return
        if frame % self.frame_interval != 0:
            return

        try:
            # 保存数据的主要方法，按照flush_every设置的频率写入frames.jsonl
            data = self._collect_frame(
                frame,
                timestamp,
                ego_vehicle,
                ego_action,
                scenario=scenario,
                controller_source=controller_source,
            )
            for event in data.get('events', []):
                self.log_event(event)
            self._frame_buffer.append(json.dumps(data))
            self._frame_count += 1
            if len(self._frame_buffer) >= self.FLUSH_EVERY:
                self._flush_frames()
        except Exception as exc:
            self._record_error('log_frame', exc)

    def log_event(self, event_dict: Dict) -> None:
        """Log a discrete event (infraction, key-decision-point, etc.)."""
        if self._events_fh is None:
            return
        try:
            self._event_buffer.append(json.dumps(event_dict))
            if len(self._event_buffer) >= 10:
                self._flush_events()
        except Exception as exc:
            self._record_error('log_event', exc)

    def close_route(self, summary: Optional[Dict] = None) -> None:
        """Flush all buffers, close file handles, and write optional summary."""
        if self._closed:
            return
        try:
            if self._private_runtime is not None:
                for event in self._private_runtime.drain_final_events():
                    self.log_event(event)
            self._adopt_agent_artifacts()
            self._flush_frames()
            self._flush_events()

            if self._trace_fh:
                self._trace_fh.close()
                self._trace_fh = None
            if self._frames_fh:
                if not self._l2d_v1_trace:
                    self._frames_fh.close()
                self._frames_fh = None
            if self._events_fh:
                self._events_fh.close()
                self._events_fh = None

            if self._l2d_v1_trace and self._logger_error_count == 0:
                try:
                    self._install_frames_hardlink()
                except Exception as exc:
                    self._record_error('publish_frames_hardlink', exc)

            out_summary = {
                'total_frames_logged': self._frame_count,
                'frame_interval': self.frame_interval,
                'logger_error_count': self._logger_error_count,
            }
            if summary:
                out_summary.update(summary)

            with open(
                os.path.join(self.save_dir, 'route_summary.json'),
                'w',
                encoding='utf-8',
            ) as fh:
                json.dump(out_summary, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            self._record_error('close_route', exc)
        finally:
            # Idempotence is important because normal stop and manager cleanup
            # can both attempt to close the same route.
            for handle_name in ('_trace_fh', '_frames_fh', '_events_fh'):
                handle = getattr(self, handle_name)
                if handle is not None:
                    try:
                        handle.close()
                    except Exception as exc:
                        self._record_error('close_handle', exc)
                    setattr(self, handle_name, None)
            self._closed = True

        capture_report = self._capture_report()
        try:
            with open(
                os.path.join(self.save_dir, 'capture_capabilities.json'),
                'w',
                encoding='utf-8',
            ) as fh:
                json.dump(capture_report, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            self._record_error('write_capture_capabilities', exc)

        postprocess_result = None
        if self._postprocessor is not None and not self._postprocessed:
            self._postprocessed = True
            try:
                postprocess_result = self._postprocessor(
                    self.save_dir, capture_report
                )
            except Exception as exc:
                self._record_error('postprocess_trace', exc)
        if postprocess_result is not None:
            try:
                summary_path = os.path.join(self.save_dir, 'route_summary.json')
                current_summary = {}
                if os.path.isfile(summary_path):
                    with open(summary_path, encoding='utf-8') as fh:
                        current_summary = json.load(fh)
                current_summary['contract_success'] = bool(
                    postprocess_result.get('success', False)
                )
                current_summary['capture_complete'] = bool(
                    postprocess_result.get('capture_complete', False)
                )
                with open(summary_path, 'w', encoding='utf-8') as fh:
                    json.dump(current_summary, fh, indent=2, ensure_ascii=False)
            except Exception as exc:
                self._record_error('update_route_summary', exc)

    def _capture_report(self) -> Dict[str, Any]:
        requirements = self._private_context.get('requirements', {})
        unresolved_gates = sorted(
            set(requirements.get('gate_ids', [])) - self._resolved_gate_ids
        )
        unresolved_actors = sorted(
            self._required_actor_ids - self._resolved_actor_ids
        )
        unresolved_dwells = sorted(
            set(requirements.get('dwell_ids', [])) - self._resolved_dwell_ids
        )
        report = {
            'schema_version': 'language2drive.trace_capture.v1',
            'private_context_loaded': bool(self._private_context),
            'only_contract_referenced_actors': True,
            'resolved_actor_ids': sorted(self._resolved_actor_ids),
            'unresolved_actor_ids': unresolved_actors,
            'resolved_gate_ids': sorted(self._resolved_gate_ids),
            'unresolved_gate_ids': unresolved_gates,
            'resolved_dwell_ids': sorted(self._resolved_dwell_ids),
            'unresolved_dwell_ids': unresolved_dwells,
            'criterion_event_support': dict(self._criterion_event_support),
            'lane_samples': self._lane_samples,
            'traffic_light_samples': self._traffic_light_samples,
            'map_legality_samples': self._map_legality_samples,
            'controller_source_samples': self._controller_source_samples,
            'unknown_controller_source_samples': self._unknown_controller_source_samples,
            'progress_supported': False,
            'passing_side_supported': False,
            'unsupported_fields': {
                'stage_transition': 'no authoritative scenario-stage hook',
                'generic_trigger': 'only versioned sidecar gate transitions are derived',
                'passing_side': 'no versioned geometric definition in current sidecar',
                'occupancy': 'reported only when a configured semantic actor is uniquely observed',
                'actor_spawn_initial_state': 'initial actors are observations, not claimed as spawn events',
            },
            'actor_lifecycle_support': 'world_actor_set_transitions_after_initial_sample',
            'gate_trigger_support': 'sidecar_polygon_or_circle_rising_edge',
            'logger_error_count': self._logger_error_count,
        }
        if self._fdq_metrics_enabled:
            report['fdq_metric_observation'] = {
                'schema_version': 'language2drive.fdq_observation.v1',
                'frame_samples': self._fdq_observation_samples,
                'ttc_finite_samples': self._fdq_ttc_finite_samples,
                'ttc_supported': True,
                'progress_supported': bool(
                    len(self._route_progress_points) >= 2
                    and self._route_progress_total > 0.0
                ),
                'map_legality_supported': self._map_legality_samples > 0,
                'relevant_actor_scope': [
                    'vehicle.*', 'walker.pedestrian.*',
                ],
                'observation_radius_m': 75.0,
                'ttc_method': (
                    'constant_velocity_ego_frame_swept_oriented_box_entry_time_v1'
                ),
            }
        if self._private_runtime is not None:
            report['evaluator_runtime'] = self._private_runtime.report()
        return report


    def _adopt_agent_artifacts(self) -> None:
        """Move opaque Standardized agent output into the private run folder."""

        if not self._agent_artifact_dir:
            return
        source_root = os.path.abspath(self._agent_artifact_dir)
        destination_root = os.path.abspath(self.save_dir)
        if source_root == destination_root:
            return
        for name in self.AGENT_ARTIFACT_NAMES:
            source = os.path.join(source_root, name)
            if os.path.isfile(source):
                os.replace(source, os.path.join(destination_root, name))
        plans_path = os.path.join(destination_root, 'agent_plans.jsonl')
        if not os.path.exists(plans_path):
            # An agent that never produced a plan is still a complete failed
            # rollout, represented by an explicit empty append-only stream.
            open(plans_path, 'a', encoding='utf-8').close()
        try:
            os.rmdir(source_root)
        except FileNotFoundError:
            pass
        except OSError:
            # Unknown files remain auditable in the opaque staging directory;
            # they are not silently copied into the canonical rollout.
            pass


    def _flush_frames(self) -> None:
        if self._frame_buffer:
            payload = '\n'.join(self._frame_buffer) + '\n'
            if self._trace_fh:
                self._trace_fh.write(payload)
            if self._frames_fh and not self._l2d_v1_trace:
                self._frames_fh.write(payload)
            self._frame_buffer.clear()


    def _install_frames_hardlink(self) -> None:
        """Atomically publish the v0 frames name as trace's same inode."""

        trace_path = os.path.join(self.save_dir, 'trace.jsonl')
        frames_path = os.path.join(self.save_dir, 'frames.jsonl')
        trace_stat = os.stat(trace_path)
        try:
            frames_stat = os.stat(frames_path)
        except FileNotFoundError:
            frames_stat = None
        if frames_stat is not None and (
            trace_stat.st_dev == frames_stat.st_dev
            and trace_stat.st_ino == frames_stat.st_ino
        ):
            return

        descriptor, temporary_path = tempfile.mkstemp(
            prefix='.frames.jsonl.link.', dir=self.save_dir
        )
        os.close(descriptor)
        os.unlink(temporary_path)
        try:
            os.link(trace_path, temporary_path)
            linked = os.stat(temporary_path)
            if (
                linked.st_dev != trace_stat.st_dev
                or linked.st_ino != trace_stat.st_ino
                or linked.st_size != trace_stat.st_size
            ):
                raise OSError('frames compatibility link identity mismatch')
            os.replace(temporary_path, frames_path)
            published = os.stat(frames_path)
            if (
                published.st_dev != trace_stat.st_dev
                or published.st_ino != trace_stat.st_ino
                or published.st_size != trace_stat.st_size
            ):
                raise OSError('published frames compatibility link mismatch')
        finally:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


    def _flush_events(self) -> None:
        if self._events_fh and self._event_buffer:
            self._events_fh.write('\n'.join(self._event_buffer) + '\n')
            self._event_buffer.clear()

    def _record_error(self, operation: str, exc: Exception) -> None:
        """Persist logger failures without recursively depending on open handles."""
        self._logger_error_count += 1
        if self._private_runtime is not None:
            self._private_runtime.record_capture_failure(operation, exc)
        payload = {
            'operation': operation,
            'error_type': type(exc).__name__,
            'message': str(exc),
            'traceback': traceback.format_exc(),
        }
        try:
            with open(
                os.path.join(self.save_dir, 'logger_errors.jsonl'),
                'a',
                encoding='utf-8',
            ) as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + '\n')
        except Exception as write_exc:
            print(
                '[Language2Drive logger] failed to persist logger error: {}'.format(
                    write_exc
                ),
                file=sys.stderr,
                flush=True,
            )
        print(
            '[Language2Drive logger] {} failed: {}'.format(operation, exc),
            file=sys.stderr,
            flush=True,
        )


    def _road_position(self, location) -> Optional[Dict[str, Any]]:
        if not _CARLA_AVAILABLE:
            return None
        try:
            waypoint = CarlaDataProvider.get_map().get_waypoint(location)
        except Exception:
            return None
        if waypoint is None:
            return None
        value = {
            'road_id': int(waypoint.road_id),
            'section_id': int(waypoint.section_id),
            'lane_id': int(waypoint.lane_id),
            's': round(float(waypoint.s), 4),
            'is_junction': bool(getattr(waypoint, 'is_junction', False)),
        }
        town = str(self.route_meta.get('town_name', '')).split('/')[-1]
        value['lane_key'] = '{}:{}:{}:{}'.format(
            town, value['road_id'], value['section_id'], value['lane_id']
        )
        return value

    @staticmethod
    def _bounding_box(actor) -> Dict[str, Any]:
        box = getattr(actor, 'bounding_box', None)
        if box is None:
            return {}
        return {
            'location': _v3(box.location),
            'rotation': _rot(box.rotation),
            'extent': _v3(box.extent),
        }

    @staticmethod
    def _footprint(transform, actor) -> List[List[float]]:
        box = getattr(actor, 'bounding_box', None)
        if box is None:
            return []
        yaw = math.radians(float(transform.rotation.yaw))
        cosine, sine = math.cos(yaw), math.sin(yaw)
        center_x = float(transform.location.x)
        center_y = float(transform.location.y)
        offset = getattr(box, 'location', None)
        if offset is not None:
            center_x += float(offset.x) * cosine - float(offset.y) * sine
            center_y += float(offset.x) * sine + float(offset.y) * cosine
        length, width = float(box.extent.x), float(box.extent.y)
        output = []
        for longitudinal, lateral in (
            (length, width), (length, -width), (-length, -width), (-length, width)
        ):
            output.append([
                round(center_x + longitudinal * cosine - lateral * sine, 4),
                round(center_y + longitudinal * sine + lateral * cosine, 4),
            ])
        return output

    def _world_actors(self) -> List[Any]:
        if not _CARLA_AVAILABLE:
            return []
        try:
            return list(CarlaDataProvider.get_world().get_actors())
        except Exception:
            return []

    def _resolve_actor(self, identifier: str, actors: List[Any], ego_vehicle):
        bound_instance = self._actor_bindings.get(identifier)
        if bound_instance is not None:
            match = next(
                (actor for actor in actors if int(actor.id) == bound_instance), None
            )
            if match is not None:
                return match

        def attributes(actor):
            value = getattr(actor, 'attributes', {})
            return value if isinstance(value, Mapping) else {}

        exact = [
            actor for actor in actors
            if actor is not ego_vehicle and str(attributes(actor).get('role_name', ''))
            in {identifier, 'l2d:{}'.format(identifier)}
        ]
        exact.extend(
            actor for actor in actors
            if actor is not ego_vehicle
            and str(attributes(actor).get('l2d_id', '')) == identifier
            and actor not in exact
        )
        spec = self._actor_specs.get(identifier, {})
        if not exact and spec:
            expected_instance = spec.get('instance_id')
            if expected_instance is not None:
                exact = [
                    actor for actor in actors
                    if str(getattr(actor, 'id', '')) == str(expected_instance)
                ]
        if not exact and spec.get('blueprint'):
            candidates = [
                actor for actor in actors
                if actor is not ego_vehicle
                and getattr(actor, 'type_id', None) == spec['blueprint']
            ]
            expected_color = spec.get('color')
            if expected_color is not None:
                candidates = [
                    actor for actor in candidates
                    if str(attributes(actor).get('color', '')) == str(expected_color)
                ]
            if len(candidates) == 1:
                exact = candidates
        if len(exact) != 1:
            return None
        actor = exact[0]
        self._actor_bindings[identifier] = int(actor.id)
        self._resolved_actor_ids.add(identifier)
        return actor

    def _actor_snapshot(
        self, identifier: str, actor, ego_transform, ego_speed: float
    ) -> Dict[str, Any]:
        transform = actor.get_transform()
        velocity = actor.get_velocity()
        acceleration = actor.get_acceleration()
        dx = float(transform.location.x - ego_transform.location.x)
        dy = float(transform.location.y - ego_transform.location.y)
        dz = float(transform.location.z - ego_transform.location.z)
        forward = ego_transform.get_forward_vector()
        right = ego_transform.get_right_vector()
        longitudinal = dx * float(forward.x) + dy * float(forward.y)
        lateral = dx * float(right.x) + dy * float(right.y)
        attributes = getattr(actor, 'attributes', {})
        attributes = attributes if isinstance(attributes, Mapping) else {}
        lights = None
        try:
            lights = str(actor.get_light_state())
        except (AttributeError, RuntimeError):
            pass
        road = self._road_position(transform.location)
        value = {
            'semantic_id': identifier,
            'instance_id': int(actor.id),
            'type': getattr(actor, 'type_id', None),
            'color': attributes.get('color'),
            'role_name': attributes.get('role_name'),
            'transform': {
                'location': _v3(transform.location),
                'rotation': _rot(transform.rotation),
            },
            'velocity': _v3(velocity),
            'speed_mps': round(_speed(velocity), 4),
            'acceleration': _v3(acceleration),
            'bounding_box': self._bounding_box(actor),
            'lights': lights,
            'relative_distance_m': round(math.sqrt(dx * dx + dy * dy + dz * dz), 4),
            'longitudinal_distance_m': round(longitudinal, 4),
            'lateral_distance_m': round(lateral, 4),
        }
        if ego_speed > 1e-3 and longitudinal >= 0:
            value['headway_seconds'] = round(longitudinal / ego_speed, 4)
        if road is not None:
            value['road_lane_s'] = road
        return value

    def _criterion_events(
        self, scenario, frame: int, sim_time: float
    ) -> List[Dict[str, Any]]:
        if scenario is None or not hasattr(scenario, 'get_criteria'):
            return []
        try:
            criteria = scenario.get_criteria()
        except Exception:
            return []
        output = []
        for criterion in criteria:
            criterion_type = type(criterion).__name__
            if 'Collision' in criterion_type:
                self._criterion_event_support['collision'] = True
            if any(name in criterion_type for name in ('OutsideRoute', 'OutsideLane', 'OffRoad')):
                self._criterion_event_support['off_road'] = True
            if 'RunningRedLight' in criterion_type:
                self._criterion_event_support['red_light'] = True
            if 'RunningStop' in criterion_type:
                self._criterion_event_support['stop_sign'] = True
            for event in getattr(criterion, 'events', []) or []:
                event_type = event.get_type()
                raw_name = getattr(event_type, 'name', str(event_type))
                event_frame = int(event.get_frame())
                key = (criterion_type, raw_name, event_frame, event.get_message())
                if key in self._criterion_events_seen:
                    continue
                self._criterion_events_seen.add(key)
                normalized = {
                    'COLLISION_STATIC': 'collision',
                    'COLLISION_VEHICLE': 'collision',
                    'COLLISION_PEDESTRIAN': 'collision',
                    'TRAFFIC_LIGHT_INFRACTION': 'red_light',
                    'STOP_INFRACTION': 'stop_sign',
                    'OUTSIDE_LANE_INFRACTION': 'off_road',
                    'OUTSIDE_ROUTE_LANES_INFRACTION': 'off_road',
                    'ON_SIDEWALK_INFRACTION': 'off_road',
                }.get(raw_name, raw_name.casefold())
                output.append({
                    'type': normalized,
                    'frame': event_frame,
                    'observed_at_frame': int(frame),
                    'sim_time': sim_time,
                    'source': 'scenario_criterion',
                    'source_event_type': raw_name,
                    'criterion': criterion_type,
                    'message': event.get_message(),
                    'details': _json_safe(event.get_dict()),
                })
        return output

    def _map_legality_event(
        self,
        ego_vehicle,
        transform,
        sim_time: float,
        frame: int,
        controller_source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Derive road legality from CARLA's lane map, independent of XML route.

        This deliberately does not use ``InRouteTest`` because the Standardized
        XML contains only a public coarse envelope rather than the hidden legal
        path.  A failed/unsupported map lookup produces no event and remains an
        explicit unsupported capability instead of fabricating an infraction.
        """

        if not _CARLA_AVAILABLE or carla is None:
            return []
        try:
            world_map = CarlaDataProvider.get_map()
        except (AttributeError, RuntimeError, ValueError):
            return []
        if world_map is None:
            return []
        location = transform.location
        candidates = []
        try:
            for lane_name in ('Driving', 'Parking'):
                lane_type = getattr(carla.LaneType, lane_name, None)
                if lane_type is None:
                    continue
                waypoint = world_map.get_waypoint(
                    location, project_to_road=False, lane_type=lane_type
                )
                if waypoint is not None:
                    candidates.append(waypoint)
        except (AttributeError, RuntimeError, TypeError):
            return []
        self._map_legality_samples += 1
        self._criterion_event_support['off_road'] = True

        in_confirmed_slot = any(
            _point_in_polygon(
                (float(location.x), float(location.y)),
                target.get('polygon'),
                min(0.2, max(0.0, float(target.get('tolerance_m', 0.2)))),
            )
            for target in self._parking_definitions.values()
        )
        legal = bool(candidates) or in_confirmed_slot
        reason = 'outside_driving_and_parking_lanes'
        audited_explicit_parking_stage = (
            bool(self._parking_definitions)
            and controller_source in _AUDITED_EXPLICIT_PARKING_CONTROLLER_SOURCES
        )
        if candidates:
            driving = next(
                (
                    waypoint
                    for waypoint in candidates
                    if str(getattr(waypoint, 'lane_type', '')).casefold().endswith('driving')
                ),
                candidates[0],
            )
            if not bool(getattr(driving, 'is_junction', False)):
                lane_yaw = float(driving.transform.rotation.yaw)
                ego_yaw = float(transform.rotation.yaw)
                heading_error = _angle_error_degrees(ego_yaw, lane_yaw)
                if heading_error > 120.0 and not audited_explicit_parking_stage:
                    legal = False
                    reason = 'wrong_direction'

        events = []
        illegal = not legal
        self._map_legal_current = legal
        if illegal and not self._map_illegal_previous:
            events.append({
                'type': 'off_road',
                'frame': int(frame),
                'sim_time': sim_time,
                'source': 'carla_map_lane_legality',
                'reason': reason,
                'location': _v3(location),
            })
        self._map_illegal_previous = illegal
        return events

    def _traffic_light(self, ego_vehicle) -> Dict[str, Any]:
        result = {'at_light': False, 'state': None, 'instance_id': None}
        try:
            result['at_light'] = bool(ego_vehicle.is_at_traffic_light())
            light = ego_vehicle.get_traffic_light()
            if light is not None:
                result['instance_id'] = int(light.id)
                result['state'] = str(light.get_state())
                transform = light.get_transform()
                result['transform'] = {
                    'location': _v3(transform.location),
                    'rotation': _rot(transform.rotation),
                }
            self._traffic_light_samples += 1
        except (AttributeError, RuntimeError):
            result['supported'] = False
        return result

    def _route_progress(self, location) -> Optional[float]:
        if len(self._route_progress_points) < 2 or self._route_progress_total <= 0.0:
            return None
        point = (float(location.x), float(location.y))
        best_distance = float('inf')
        best_progress = 0.0
        for index, (start, end) in enumerate(zip(
            self._route_progress_points, self._route_progress_points[1:]
        )):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length_squared = dx * dx + dy * dy
            if length_squared <= 0.0:
                continue
            ratio = max(0.0, min(1.0, (
                (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
            ) / length_squared))
            projected = (start[0] + ratio * dx, start[1] + ratio * dy)
            distance = math.hypot(point[0] - projected[0], point[1] - projected[1])
            progress = (
                self._route_progress_cumulative[index]
                + ratio * math.sqrt(length_squared)
            ) / self._route_progress_total
            if distance < best_distance:
                best_distance, best_progress = distance, progress
        return round(max(0.0, min(1.0, best_progress)), 6)

    @staticmethod
    def _collision_half_extents(actor) -> Tuple[float, float]:
        box = getattr(actor, 'bounding_box', None)
        extent = getattr(box, 'extent', None)
        if extent is None:
            return 0.5, 0.5
        try:
            return max(0.1, float(extent.x)), max(0.1, float(extent.y))
        except (TypeError, ValueError):
            return 0.5, 0.5

    @staticmethod
    def _swept_axis_interval(
        position: float, velocity: float, half_extent: float
    ) -> Optional[Tuple[float, float]]:
        if abs(velocity) <= 1e-6:
            return (0.0, float('inf')) if abs(position) <= half_extent else None
        first = (-half_extent - position) / velocity
        second = (half_extent - position) / velocity
        return min(first, second), max(first, second)

    def _collect_fdq_observation(
        self, ego_vehicle, ego_transform, ego_velocity, actors: List[Any]
    ) -> Dict[str, Any]:
        """Capture compact, evaluator-authoritative inputs for PSM/RLS/GPE."""

        ego_location = ego_transform.location
        ego_longitudinal, ego_lateral = self._collision_half_extents(ego_vehicle)
        ego_yaw = math.radians(float(ego_transform.rotation.yaw))
        cosine, sine = math.cos(ego_yaw), math.sin(ego_yaw)
        relevant_count = 0
        minimum_ttc: Optional[float] = None
        minimum_actor_id: Optional[int] = None
        minimum_actor_type: Optional[str] = None
        ego_instance_id = int(ego_vehicle.id)
        for actor in actors:
            if actor is ego_vehicle or int(getattr(actor, 'id', -1)) == ego_instance_id:
                continue
            actor_type = str(getattr(actor, 'type_id', ''))
            if not (
                actor_type.startswith('vehicle.')
                or actor_type.startswith('walker.pedestrian.')
            ):
                continue
            try:
                transform = actor.get_transform()
                velocity = actor.get_velocity()
                world_px = float(transform.location.x - ego_location.x)
                world_py = float(transform.location.y - ego_location.y)
                distance = math.hypot(world_px, world_py)
                if distance > 75.0:
                    continue
                relevant_count += 1
                world_vx = float(velocity.x - ego_velocity.x)
                world_vy = float(velocity.y - ego_velocity.y)
                px = world_px * cosine + world_py * sine
                py = -world_px * sine + world_py * cosine
                vx = world_vx * cosine + world_vy * sine
                vy = -world_vx * sine + world_vy * cosine
                actor_longitudinal, actor_lateral = self._collision_half_extents(actor)
                relative_yaw = math.radians(
                    float(transform.rotation.yaw) - float(ego_transform.rotation.yaw)
                )
                actor_cosine, actor_sine = (
                    abs(math.cos(relative_yaw)), abs(math.sin(relative_yaw))
                )
                projected_longitudinal = (
                    actor_cosine * actor_longitudinal
                    + actor_sine * actor_lateral
                )
                projected_lateral = (
                    actor_sine * actor_longitudinal
                    + actor_cosine * actor_lateral
                )
                longitudinal_interval = self._swept_axis_interval(
                    px, vx, ego_longitudinal + projected_longitudinal + 0.2
                )
                lateral_interval = self._swept_axis_interval(
                    py, vy, ego_lateral + projected_lateral + 0.2
                )
                if longitudinal_interval is None or lateral_interval is None:
                    continue
                entry = max(
                    0.0, longitudinal_interval[0], lateral_interval[0]
                )
                exit_time = min(
                    longitudinal_interval[1], lateral_interval[1]
                )
                if entry > exit_time or exit_time < 0.0:
                    continue
                ttc = entry
                if minimum_ttc is None or ttc < minimum_ttc:
                    minimum_ttc = ttc
                    minimum_actor_id = int(actor.id)
                    minimum_actor_type = actor_type
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
        self._fdq_observation_samples += 1
        if minimum_ttc is not None:
            self._fdq_ttc_finite_samples += 1
        return {
            'schema_version': 'language2drive.fdq_observation.v1',
            'route_progress': self._route_progress(ego_location),
            'relevant_actor_count': relevant_count,
            'min_ttc_seconds': (
                None if minimum_ttc is None else round(minimum_ttc, 4)
            ),
            'min_ttc_actor_instance_id': minimum_actor_id,
            'min_ttc_actor_type': minimum_actor_type,
            'map_legal': self._map_legal_current,
        }

    def _collect_private_states(
        self,
        data: Dict[str, Any],
        ego_vehicle,
        ego_transform,
        ego_speed: float,
        sim_time: float,
        actors: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        actors = self._world_actors() if actors is None else actors
        actor_rows = []
        current_instances: Set[int] = set()
        occupancy: Dict[str, Dict[str, Any]] = {}
        actor_order_values = []
        headways = {}
        for identifier in sorted(self._required_actor_ids):
            actor = self._resolve_actor(identifier, actors, ego_vehicle)
            if actor is None:
                continue
            instance_id = int(actor.id)
            current_instances.add(instance_id)
            snapshot = self._actor_snapshot(
                identifier, actor, ego_transform, ego_speed
            )
            actor_rows.append(snapshot)
            if (
                self._actor_set_initialized
                and instance_id not in self._previous_actor_instances
            ):
                events.append({
                    'type': 'actor_spawn',
                    'actor_id': identifier,
                    'instance_id': instance_id,
                    'sim_time': sim_time,
                    'source': 'world_actor_observation',
                })
            longitudinal = snapshot['longitudinal_distance_m']
            if longitudinal > 1.0:
                order = 'ego_behind:{}'.format(identifier)
            elif longitudinal < -1.0:
                order = 'ego_ahead:{}'.format(identifier)
            else:
                order = None
            if order is not None:
                actor_order_values.append(order)
                events.append({
                    'type': 'actor_order',
                    'id': order,
                    'actor_id': identifier,
                    'sim_time': sim_time,
                    'source': 'ego_frame_longitudinal_projection',
                })
            if snapshot.get('headway_seconds') is not None:
                headways[identifier] = snapshot['headway_seconds']
            spec = self._actor_specs.get(identifier, {})
            if spec.get('slot'):
                slot = str(spec['slot'])
                occupancy[slot] = True
                data.setdefault('occupancy_details', {})[slot] = {
                    'occupied': True,
                    'actor_id': identifier,
                    'instance_id': instance_id,
                    'source': 'configured_semantic_actor_observed',
                }
                events.append({
                    'type': 'actor_order',
                    'id': 'occupied:{}'.format(slot),
                    'actor_id': identifier,
                    'sim_time': sim_time,
                    'source': 'configured_semantic_actor_observed',
                })
        reverse_bindings = {value: key for key, value in self._actor_bindings.items()}
        if self._actor_set_initialized:
            for instance_id in sorted(self._previous_actor_instances - current_instances):
                events.append({
                    'type': 'actor_despawn',
                    'actor_id': reverse_bindings.get(instance_id),
                    'instance_id': instance_id,
                    'sim_time': sim_time,
                    'source': 'world_actor_set_transition',
                })
        self._previous_actor_instances = current_instances
        self._actor_set_initialized = True
        self._observed_actor_instances.update(current_instances)
        data['actors'] = actor_rows
        data['headways'] = headways
        data['actor_order'] = actor_order_values[0] if len(actor_order_values) == 1 else actor_order_values
        data['occupancy'] = occupancy

        point = (float(ego_transform.location.x), float(ego_transform.location.y))
        trigger_states = {}
        for identifier, target in self._gate_definitions.items():
            inside = _inside_gate(point, target)
            trigger_states[identifier] = inside
            if inside and not self._gate_inside.get(identifier, False):
                events.append({
                    'type': 'gate',
                    'gate_id': identifier,
                    'id': identifier,
                    'sim_time': sim_time,
                    'source': 'sidecar_gate_geometry_transition',
                })
                events.append({
                    'type': 'trigger',
                    'trigger_id': identifier,
                    'id': identifier,
                    'sim_time': sim_time,
                    'source': 'sidecar_gate_geometry_transition',
                })
            self._gate_inside[identifier] = inside
        data['trigger_states'] = trigger_states

        footprint = data['ego'].get('footprint', [])
        dwell = {}
        for identifier, target in self._parking_definitions.items():
            tolerance = min(
                0.2, max(0.0, float(target.get('tolerance_m', 0.2)))
            )
            inside = bool(footprint) and all(
                _point_in_polygon(
                    (corner[0], corner[1]), target['polygon'], tolerance
                )
                for corner in footprint
            )
            target_heading = target.get(
                'heading_deg', target.get('target_heading_deg')
            )
            if target_heading is not None:
                headings = (
                    target_heading
                    if isinstance(target_heading, list)
                    else [target_heading]
                )
                heading_tolerance = float(
                    target.get('heading_tolerance_deg', 15.0)
                )
                ego_heading = float(ego_transform.rotation.yaw)
                inside = inside and any(
                    _angle_error_degrees(ego_heading, heading) <= heading_tolerance
                    for heading in headings
                )
            maximum_speed = float(target.get('speed_max_mps', 0.2))
            active = inside and ego_speed < maximum_speed
            if active and self._dwell_started.get(identifier) is None:
                self._dwell_started[identifier] = sim_time
            if not active:
                self._dwell_started[identifier] = None
            start = self._dwell_started.get(identifier)
            dwell[identifier] = 0.0 if start is None else round(sim_time - start, 4)
        data['dwell'] = dwell
        return events

    def _collect_frame(
        self,
        frame: int,
        timestamp,
        ego_vehicle,
        ego_action,
        *,
        scenario=None,
        controller_source: Optional[str] = None,
    ) -> Dict:
        """Record authoritative world state and criterion events for one tick."""
        data: Dict[str, Any] = {
            'frame': frame,
            'sim_time': round(float(timestamp.elapsed_seconds), 4),
        }
        public_context = self.route_meta.get(
            'language2drive_public_context', {}
        )
        data['instruction'] = public_context.get(
            'instruction', self.route_meta.get('instruction')
        )
        data['instruction_version'] = int(
            public_context.get('instruction_version', 0)
        )

        # 获取ego的位子信息，包含location和rotation
        transform = ego_vehicle.get_transform()
        # 获取ego的速度 (m/s) 世界坐标系下的三个分量
        vel = ego_vehicle.get_velocity()
        # 获取ego的加速度 (m/s^2) 世界坐标系下的三个分量
        acc = ego_vehicle.get_acceleration()
        # 获取ego的角速度 (rad/s或者deg/s，取决于carla版本，需要确认一下) 世界坐标系下的三个分量 TODO: 确认carla版本对应的是哪个单位
        ang = ego_vehicle.get_angular_velocity()
        # 标量速度表示，由vel这个向量速度计算得到
        speed = _speed(vel)
        road = self._road_position(transform.location)
        footprint = self._footprint(transform, ego_vehicle)

        # 保存ego的状态信息
        data['ego'] = {
            'location': _v3(transform.location),
            'rotation': _rot(transform.rotation),
            'transform': {
                'location': _v3(transform.location),
                'rotation': _rot(transform.rotation),
            },
            'velocity': _v3(vel),
            'speed': round(speed, 4),
            'speed_mps': round(speed, 4),
            'acceleration': _v3(acc),
            'angular_velocity': _v3(ang),
            'forward_vector': _v3(transform.get_forward_vector()),
            'right_vector': _v3(transform.get_right_vector()),
            'bounding_box': self._bounding_box(ego_vehicle),
            'footprint': footprint,
        }
        if road is not None:
            data['ego']['road_lane_s'] = road
            data['ego']['road_id'] = road['road_id']
            data['ego']['section_id'] = road['section_id']
            data['ego']['lane_id'] = road['lane_id']
            data['ego']['s'] = road['s']
            data['ego']['lane_relation'] = road['lane_key']
            self._lane_samples += 1

        # 保存ego的控制信息，包括throttle、steer、brake、reverse、hand_brake、gear
        data['control'] = {
            'throttle': round(float(ego_action.throttle), 4),
            'steer': round(float(ego_action.steer), 4),
            'brake': round(float(ego_action.brake), 4),
            'reverse': bool(ego_action.reverse),
            'hand_brake': bool(ego_action.hand_brake),
            'gear': int(getattr(ego_action, 'gear', 0)),
        }

        source = str(controller_source or 'unknown')
        data['controller'] = {'source': source}
        data['controller_source'] = source
        self._controller_source_samples += 1
        if source == 'unknown':
            self._unknown_controller_source_samples += 1
        data['traffic_light'] = self._traffic_light(ego_vehicle)
        actors = self._world_actors()
        events = self._collect_private_states(
            data, ego_vehicle, transform, speed, data['sim_time'], actors
        )
        events.extend(
            self._criterion_events(scenario, frame, data['sim_time'])
        )
        events.extend(
            self._map_legality_event(
                ego_vehicle,
                transform,
                data['sim_time'],
                frame,
                controller_source=source,
            )
        )
        if self._fdq_metrics_enabled:
            data['fdq_observation'] = self._collect_fdq_observation(
                ego_vehicle, transform, vel, actors
            )
            data['fdq_observation']['map_legal'] = self._map_legal_current
        data['events'] = events
        if self._private_runtime is not None:
            self._private_runtime.observe_frame(data)

        return data
