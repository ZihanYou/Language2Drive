"""Bench2Drive autonomous agent using the common trajectory service.

Controller: pure pursuit, wheelbase 2.8m; no route/target-point input.
"""
import json
import math
import uuid
import carla
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from team_code.language2drive_context import PublicContextState
from l2d_benchmark.adapter import Request, SPECS
from l2d_benchmark.service import Client, image_payload


def get_entry_point():
    return 'UnifiedAgent'


class UnifiedAgent(AutonomousAgent):
    def set_language2drive_context(self, context):
        state = getattr(self, '_context_state', None) or PublicContextState()
        self._context_state = state
        parsed = state.bind(context)
        if parsed.track != 'standardized':
            raise ValueError('Unified bridge requires standardized, language-only evaluation')
        self._global_plan = self._global_plan_world_coord = None
        return parsed.to_dict()

    def set_global_plan(self, *args):
        raise ValueError('Unified bridge never consumes global routes')

    def setup(self, path_to_conf_file, route_index=None):
        self.track = Track.SENSORS
        self.cfg = json.loads(open(path_to_conf_file).read())
        self.client = Client(self.cfg.get('endpoint', 'http://127.0.0.1:8090'))
        self.session = uuid.uuid4().hex
        self.client.reset(self.session)
        self.history = []

    def sensors(self):
        result = []
        for name, yaw in [('front', 0), ('left', -60), ('right', 60)]:
            result.append(dict(type='sensor.camera.rgb', x=0, y=0, z=1.6, roll=0, pitch=0, yaw=yaw,
                               width=960, height=540, fov=100, id='rgb_' + name))
        result.append(dict(type='sensor.speedometer', reading_frequency=20, id='speed'))
        return result

    def run_step(self, input_data, timestamp):
        context = self._context_state.context
        if context is None:
            raise ValueError('Evaluator did not bind the public instruction')
        # CARLA camera BGRA -> RGB. Camera positions are vehicle origin, not rear axle.
        cameras = {name: [{'timestamp_s': float(timestamp), 'image': image_payload(input_data['rgb_' + name][1][:, :, :3][:, :, ::-1])}]
                   for name in ('front', 'left', 'right')}
        speed = float(input_data['speed'][1]['speed'])
        request = Request('carla', context.instruction,
                          {'cameras': cameras, 'speed_m_s': speed, 'timestamp_s': float(timestamp),
                           'camera_rig': self.sensors(), 'vehicle_origin_to_rear_axle_m': self.cfg.get('rear_axle_offset_m', -1.4)}, SPECS['carla'], int(self.cfg.get('model_seed', 0)))
        xy = self.client.predict(request, self.session)['xy_m']
        lookahead = max(2.0, speed * .5)
        target = next((p for p in xy if math.hypot(*p) >= lookahead), xy[-1])
        distance2 = max(target[0]**2 + target[1]**2, .01)
        # Positive model Y is left; positive CARLA steer is right.
        steering_angle = -math.atan2(2 * 2.8 * target[1], distance2)
        steer = max(-1., min(1., steering_angle / .7))
        desired = math.hypot(*xy[0]) / SPECS['carla'].timestamps_s[0]
        delta = desired - speed
        return carla.VehicleControl(steer=steer, throttle=max(0., min(.75, delta * .3)), brake=max(0., min(1., -delta * .5)))

    def destroy(self):
        if hasattr(self, 'client'):
            self.client.call('close', {'session': self.session})
