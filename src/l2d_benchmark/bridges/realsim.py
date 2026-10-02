"""Pinned AlpaSim trajectory plugin, with isolated recurrent sessions."""
import os
import numpy as np
from alpasim_driver.models.base import BaseTrajectoryModel, ModelPrediction
from alpasim_utils.geometry import Trajectory, pose_from_grpc
from l2d_benchmark.adapter import Request, SPECS
from l2d_benchmark.service import Client, image_payload


class UnifiedTrajectoryModel(BaseTrajectoryModel):
    @classmethod
    def from_config(cls, model_cfg, device, camera_ids, context_length, output_frequency_hz):
        if output_frequency_hz != 10:
            raise ValueError('Unified RealSim output frequency must be 10Hz')
        return cls(camera_ids, context_length)

    def __init__(self, camera_ids, context_length):
        self._camera_ids = list(camera_ids)
        self._context_length = context_length
        self.client = Client(os.environ.get('L2D_MODEL_ENDPOINT', 'http://127.0.0.1:8090'))
        self.sessions = set()

    @property
    def camera_ids(self):
        return self._camera_ids

    @property
    def context_length(self):
        return self._context_length

    @property
    def output_frequency_hz(self):
        return 10

    def _encode_command(self, command):
        return 0

    def predict(self, prediction_input):
        session = prediction_input.l2d_session_uuid
        if session not in self.sessions:
            self.client.reset(session)
            self.sessions.add(session)
        timestamp = int(prediction_input.l2d_timestamp_us)
        cameras = {name: [{'timestamp_us': int(t), 'image': image_payload(im)} for t, im in frames]
                   for name, frames in prediction_input.camera_images.items()}
        poses = sorted(prediction_input.ego_pose_history or [], key=lambda p: p.timestamp_us)
        if not poses or not poses[0].timestamp_us <= timestamp <= poses[-1].timestamp_us:
            raise ValueError('No ego pose covers the trajectory anchor; refusing unanchored output')
        trajectory = self.client.predict(Request('realsim', prediction_input.language_instruction,
            {'cameras': cameras, 'speed_m_s': float(prediction_input.speed), 'acceleration_m_s2': float(prediction_input.acceleration), 'timestamp_us': timestamp, 'camera_rectification': prediction_input.l2d_camera_calibration},
            SPECS['realsim'], int(prediction_input.inference_seed)), session)
        xy = np.asarray(trajectory['xy_m'])
        delta = np.diff(np.vstack([np.zeros((1, 2)), xy]), axis=0)
        prediction = ModelPrediction.from_planar(xy, np.arctan2(delta[:, 1], delta[:, 0]))
        pose = Trajectory.from_poses(np.asarray([p.timestamp_us for p in poses], dtype=np.uint64), [pose_from_grpc(p.pose) for p in poses]).interpolate_pose(timestamp)
        prediction.model_t0_us = timestamp
        prediction.pose_local_to_rig_t0 = pose
        prediction.waypoint_timestamps_us = timestamp + np.arange(1, len(xy)+1, dtype=np.uint64) * 100000
        return prediction

    def close(self):
        for session in self.sessions:
            self.client.call('close', {'session': session})
        self.sessions.clear()
