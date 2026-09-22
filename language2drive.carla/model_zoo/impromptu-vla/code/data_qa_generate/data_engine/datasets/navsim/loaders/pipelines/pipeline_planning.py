from pathlib import Path
import re
import traceback
import lzma, pickle
from typing import List
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import numpy.typing as npt
import data_engine.datasets.navsim.loaders.navsim as navsim
import sys
sys.modules['navsim'] = navsim

from data_engine.datasets.navsim.loaders.navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex

import os
from hydra.utils import instantiate
from data_engine.datasets.navsim.loaders.navsim.common.dataloader import MetricCacheLoader
from data_engine.datasets.nuscenes.loaders.pipelines.pipeline_planning import *
from data_engine.datasets.navsim.loaders.pipelines.pipeline_blueprint import PromptNavsimBlueprint

from data_engine.datasets.navsim.loaders.navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from data_engine.datasets.navsim.loaders.navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from data_engine.datasets.navsim.loaders.navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from data_engine.datasets.navsim.loaders.navsim.planning.metric_caching.metric_cache import MetricCache
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from navsim.common.dataclasses import PDMResults, Trajectory
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
)
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array



FUT_TIMESTEPS = 8

class PromptNavsimPlanning(PromptNavsimBlueprint):

    def __init__(self, navsim=None, mode="dist-dtheta", **kwargs):
        super().__init__(navsim=navsim, container_out_key="planning", cache_response_filename=None, need_helper=False, **kwargs)
        
        self.mode = mode
        assert self.mode in ["dist-dtheta", "dist-theta", "polar", "dx-dy", "dist-curvature", "x-y"]
        
        self.simulator = None
        self.planning_metrics = {}
        
    def extract_scene_trajs(self, container_out, container_in):
        global_ego_poses = []
        for k in range(len(container_in['frame_data'])):
            global_ego_pose_k = container_in['frame_data'][k]['ego_status'].ego_pose
            global_ego_poses.append(global_ego_pose_k)
        
        global_ego_pose_now = global_ego_poses[3]
        
        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_pose_now), np.array(global_ego_poses, dtype=np.float64)
        )
        local_ego_poses_xy = local_ego_poses[:, :2]
        x, y = local_ego_poses_xy[:, 0], local_ego_poses_xy[:, 1]
        x, y = -y, x  # to lidar coordinate frame
        
        return np.concatenate([x[:, None], y[:, None]], axis=1), 3

    def format_output(self, helper_ret, container_out, container_in):
        assert len(helper_ret) == 0
        out_string = ""
        out_string += "<PLANNING>"
        
        ego_poses, this_sample_idx = self.extract_scene_trajs(container_out, container_in)
        
        ego_poses_future = ego_poses[this_sample_idx:]

        if self.mode == "dist-dtheta":
            # in current bev plane, what is the change of direction and distance traveled?
            ego_dthetas_future = compute_traj_to_dthetas(ego_poses_future)
            ego_velocity_future = compute_traj_to_velocity(ego_poses_future)
            ego_dthetas_future, ego_velocity_future = np.round(ego_dthetas_future, 0), np.round(ego_velocity_future, 2)
            
            x, y = ego_velocity_future, ego_dthetas_future

            out_string += "Predicted future movement details for the next 5 seconds (sampled at 0.5-second intervals), including distance traveled (in meters) and change in direction (in degrees). A value of 0 for direction indicates moving straight, while positive values represent left turns (with the angle increasing during the initial phase and decreasing back to 0 as the turn completes). The output is formatted as [displacement, theta]: "
            
        elif self.mode == "dist-theta":
            # in current BEV map (xoy plane), which direction does the ego vehicle go?
            ego_thetas_future = compute_traj_to_thetas(ego_poses_future)
            ego_velocity_future = compute_traj_to_velocity(ego_poses_future)
            
            ego_thetas_future, ego_velocity_future = np.round(ego_thetas_future, 0), np.round(ego_velocity_future, 2)
            x, y = ego_velocity_future, ego_thetas_future
            out_string += "Predicted future movement details for the next 5 seconds (sampled at 0.5-second intervals), including distance traveled (in meters) and direction (in degrees). A value of 0 for direction indicates moving straight, while positive values represent left turns. The output is formatted as [displacement, theta]: "
            
        elif self.mode == "polar":
            ego_pose = ego_poses_future[0]
            polar_distance, polar_angle = compute_traj_to_polars(ego_poses_future, ego_pose)
            
            polar_distance, polar_angle = np.round(polar_distance, 2), np.round(polar_angle, 0)
            x, y = polar_distance, polar_angle
            out_string += "Predicted future movement details for the next 5 seconds (sampled at 0.5-second intervals), including distance to current location (in meters) and polar direction (in degrees). A value of 0 for direction indicates moving straight, while positive values represent leftwards. The output is formatted as [distance, polar angle]: "
            
        elif self.mode == "dx-dy":
            x, y = compute_traj_to_dx_dy(ego_poses_future)
            x, y = np.round(x, 2), np.round(y, 2)
            out_string += "Predicted future movement details for the next 5 seconds (sampled at 0.5-second intervals), including displacement in x and y directions (in meters). Positive y means forward direction while positive x means rightwards. The output is formatted as [dx, dy]: "
        
        elif self.mode == "dist-curvature":
            dist = compute_traj_to_velocity(ego_poses_future)
            curvature = compute_traj_to_curvature(ego_poses_future) * 100
            x, y = np.round(dist, 2), np.round(curvature, 2)
            out_string += "Predicted future movement details for the next 5 seconds (sampled at 0.5-second intervals), including distance traveled (in meters) and curvature. The output is formatted as [distance, curvature]: "
        
        elif self.mode == "x-y":
            rightward, forward = ego_poses_future[:, 0], ego_poses_future[:, 1]
            x, y = forward, -rightward
            x, y = np.round(x, 2), np.round(y, 2)
            out_string += "Predicted future movement details for the next 5 seconds (sampled at 0.5-second intervals), including BEV location in x and y directions (in meters). Positive x means forward direction while positive y means leftwards. The output is formatted as [x, y]: "
            
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented.")
        
        for i in range(1, 11):
            xi = x[i]
            yi = y[i]
            if xi == -0.0: xi = 0.0
            if yi == -0.0: yi = 0.0
            out_string += f"[{xi}, {yi}]"
            if i < 10:
                out_string += ", "
        
        
        out_string += "</PLANNING>"
        return out_string

    def evaluation_reset(self):
        cfg_path = "data_engine/datasets/navsim/loaders/navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"
        cfg = OmegaConf.load(cfg_path)
        # cfg: DictConfig = args[0]["cfg"]
        
        prefix = "data_engine.datasets.navsim.loaders."
        
        cfg.simulator._target_ = prefix + cfg.simulator._target_
        self.simulator: PDMSimulator = instantiate(cfg.simulator)
        
        cfg.scorer._target_ = prefix + cfg.scorer._target_
        cfg.scorer.config._target_ =  prefix + cfg.scorer.config._target_
        self.scorer: PDMScorer = instantiate(cfg.scorer)
        assert (
            self.simulator.proposal_sampling == self.scorer.proposal_sampling
        ), "Simulator and scorer proposal sampling has to be identical"
        
        
        # cfg.metric_cache_path_trainval = "data_engine/data_storage/cached_responses/navsim_trainval_metric_cache"
        cfg.metric_cache_path_test = "data_engine/data_storage/cached_responses/navsim_test_metric_cache"
        self.metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path_test))
        
        # DONE! please use self.metric_cache_loader[token] to get the metric cache for a token
        self.planning_metrics = {}
    

    def transform_trajectory(self, pred_trajectory: Trajectory, initial_ego_state: EgoState) -> InterpolatedTrajectory:
        """
        Transform trajectory in global frame and return as InterpolatedTrajectory
        :param pred_trajectory: trajectory dataclass in ego frame
        :param initial_ego_state: nuPlan's ego state object
        :return: nuPlan's InterpolatedTrajectory
        """

        future_sampling = pred_trajectory.trajectory_sampling
        timesteps = _get_fixed_timesteps(initial_ego_state, future_sampling.time_horizon, future_sampling.interval_length)

        relative_poses = np.array(pred_trajectory.poses, dtype=np.float64)
        relative_states = [StateSE2.deserialize(pose) for pose in relative_poses]
        absolute_states = relative_to_absolute_poses(initial_ego_state.rear_axle, relative_states)

        # NOTE: velocity and acceleration ignored by LQR + bicycle model
        agent_states = [
            _se2_vel_acc_to_ego_state(
                state,
                [0.0, 0.0],
                [0.0, 0.0],
                timestep,
                initial_ego_state.car_footprint.vehicle_parameters,
            )
            for state, timestep in zip(absolute_states, timesteps)
        ]

        # NOTE: maybe make addition of initial_ego_state optional
        return InterpolatedTrajectory([initial_ego_state] + agent_states)

    def get_trajectory_as_array(
        self,
        trajectory: InterpolatedTrajectory,
        future_sampling: TrajectorySampling,
        start_time: TimePoint,
    ) -> npt.NDArray[np.float64]:
        """
        Interpolated trajectory and return as numpy array
        :param trajectory: nuPlan's InterpolatedTrajectory object
        :param future_sampling: Sampling parameters for interpolation
        :param start_time: TimePoint object of start
        :return: Array of interpolated trajectory states.
        """

        times_s = np.arange(
            0.0,
            future_sampling.time_horizon + future_sampling.interval_length,
            future_sampling.interval_length,
        )
        times_s += start_time.time_s
        times_us = [int(time_s * 1e6) for time_s in times_s]
        times_us = np.clip(times_us, trajectory.start_time.time_us, trajectory.end_time.time_us)
        time_points = [TimePoint(time_us) for time_us in times_us]

        trajectory_ego_states: List[EgoState] = trajectory.get_state_at_times(time_points)

        return ego_states_to_state_array(trajectory_ego_states)

    def evaluation_pdms(self, traj, container_in):
        if self.simulator is None:
            self.evaluation_reset()
        
        this_token = self.get_token(container_in)
        
        metric_cache_path = self.metric_cache_loader.metric_cache_paths[this_token]
        
        #! 0. prepare input 
        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache: MetricCache = pickle.load(f)
        model_trajectory = traj
        future_sampling = self.simulator.proposal_sampling
        simulator = self.simulator
        scorer = self.scorer
        
        #! 0.5. model_trajectory  from [1, FUT_TIMESTEPS, 2] to [FUT_TIMESTEPS, 3] by padding 0
        model_trajectory = torch.cat([model_trajectory, torch.zeros(1, FUT_TIMESTEPS, 1)], dim=2)[0]
        model_trajectory = model_trajectory.detach().cpu().numpy()
        model_trajectory = Trajectory(model_trajectory)
        

        #! 1. get states
        initial_ego_state = metric_cache.ego_state
        pdm_trajectory = metric_cache.trajectory
        pred_trajectory = self.transform_trajectory(model_trajectory, initial_ego_state)

        pdm_states, pred_states = (
            self.get_trajectory_as_array(pdm_trajectory, future_sampling, initial_ego_state.time_point),
            self.get_trajectory_as_array(pred_trajectory, future_sampling, initial_ego_state.time_point),
        )

        #! 2. pred_states -> pred_mod_states (compensate for heading, i.e. dim 2)
        # 取出前 3 维
        poses_3d = pred_states[:, :3].copy()

        # 计算前两维的均值
        mean_first_dim = np.mean(poses_3d[:, 0])
        mean_second_dim = np.mean(poses_3d[:, 1])

        # 对前两维减去它们各自的均值
        poses_3d[:, 0] -= mean_first_dim
        poses_3d[:, 1] -= mean_second_dim
        
        
        new_d2 = np.zeros(len(poses_3d))

        # 计算中间点的 heading
        for i in range(1, len(poses_3d) - 1):
            x1, y1 = poses_3d[i-1][:2]
            x2, y2 = poses_3d[i+1][:2]
            new_d2[i] = np.arctan2(y2 - y1, x2 - x1)

        # 外插第一个点的 heading
        x1, y1 = poses_3d[0][:2]
        x2, y2 = poses_3d[1][:2]
        new_d2[0] = 2 * new_d2[1] - new_d2[2]

        # 外插最后一个点的 heading
        x1, y1 = poses_3d[-2][:2]
        x2, y2 = poses_3d[-1][:2]
        new_d2[-1] = 2 * new_d2[-2] - new_d2[-3]

        pred_modified_states = np.copy(pred_states)
        pred_modified_states[:, 2] = new_d2

        trajectory_states = np.concatenate([pdm_states[None, ...], pred_modified_states[None, ...]], axis=0)  # 0 for GT, 1 for PRED
        
        #! 3. simulate states!
        simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)
        
        #! 4. compute scores
        scores = scorer.score_proposals(
            simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
        )
        
        #! 5. save scores
        pred_idx = 1

        no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx]
        drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx]

        ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_idx]
        time_to_collision_within_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx]
        comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx]
        driving_direction_compliance = scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, pred_idx]
        score = scores[pred_idx]
        
        self.planning_metrics[this_token] = {
            "no_at_fault_collisions": no_at_fault_collisions,
            "drivable_area_compliance": drivable_area_compliance,
            "ego_progress": ego_progress,
            "time_to_collision_within_bound": time_to_collision_within_bound,
            "comfort": comfort,
            "driving_direction_compliance": driving_direction_compliance,
            "score": score,
        }
        
        # todo: add visualization with condition
        
        
    
    def evaluation_update(self, pred, container_out, container_in, use_gt_str=False):
        
        trajs_str = pred['predict']
        
        # parse the matches into a list of tuples
        def trajs_str_to_trajs(trajs_str):
            try:
                trajs_str = trajs_str[trajs_str.find("<PLANNING>") + len("<PLANNING>"):trajs_str.find("</PLANNING>")]
                
                # use regex to extract the trajectory points
                pattern = r'\[([^\]]+)\]'
                matches = re.findall(pattern, trajs_str)

                if 'displacement, theta' in matches:
                    matches.remove('displacement, theta')
                    
                if 'dx, dy' in matches:
                    matches.remove('dx, dy')
                    
                if 'distance, curvature' in matches:
                    matches.remove('distance, curvature')
                    
                if 'distance, polar angle' in matches:
                    matches.remove('distance, polar angle')
                
                if 'x, y' in matches:
                    matches.remove('x, y')
                
                traj_reps = []
                for match in matches:
                    point = tuple(map(float, match.split(',')))
                    traj_reps.append(point)
                traj_reps = torch.tensor(np.array(traj_reps))[:FUT_TIMESTEPS, :2]
                # traj_array = np.array(traj_reps)

                # # 如果是一维，例如 [x1, y1, x2, y2, ...]，就 reshape
                # if traj_array.ndim == 1:
                #     traj_array = traj_array.reshape(-1, 2)

                # traj_reps = torch.tensor(traj_array[:FUT_TIMESTEPS, :2], dtype=torch.float32)

                if self.mode == "dist-dtheta":
                    # if there are 'displacement and theta in matches, drop them
                    vel, dtheta = traj_reps[:, 0], traj_reps[:, 1]
                    trajs = compute_vel_dtheta_to_traj(vel, dtheta,)  # after cumsum, [FUT_TIMESTEPS, 2]
                elif self.mode == "dist-theta":
                    dist, theta = traj_reps[:, 0], traj_reps[:, 1]
                    trajs = compute_vel_theta_to_traj(dist, theta)
                    
                elif self.mode == "dist-curvature":
                    dist, curvature = traj_reps[:, 0], traj_reps[:, 1] / 100
                    trajs = compute_vel_curvature_to_traj(dist, curvature)
                
                elif self.mode == "polar":
                    dist, angle = traj_reps[:, 0], traj_reps[:, 1]
                    trajs = compute_polar_to_traj(dist, angle)
                    
                elif self.mode == "dx-dy":
                    dx, dy = traj_reps[:, 0], traj_reps[:, 1]
                    trajs = compute_dx_dy_to_traj(dx, dy)
                    
                elif self.mode == "x-y":
                    forward, leftward = traj_reps[:, 0], traj_reps[:, 1]
                    trajs = torch.concatenate([-leftward[..., None], forward[..., None]], dim=-1)  # [k, 2]
                
                else:
                    raise NotImplementedError(f"Mode {self.mode} not implemented.")
                
                # if < FUT_TIMESTEPS points, pad last point
                if trajs.shape[0] < FUT_TIMESTEPS:
                    pad = trajs[-1:, :].repeat(FUT_TIMESTEPS - trajs.shape[0], 1)
                    trajs = torch.cat([trajs, pad], dim=0)
                
                trajs = trajs[None, ...]  # [1, FUT_TIMESTEPS, 2]
                trajs = torch.tensor(trajs)
                
            except:
                trajs = torch.zeros(1, FUT_TIMESTEPS, 2)
                traceback.print_exc()
                print(f"Error parsing trajectory: {trajs_str}")
            return trajs
        
        trajs = trajs_str_to_trajs(trajs_str)
        
        
        x, y = trajs[0, :, 0], trajs[0, :, 1]
        new_x = y
        new_y = -x
        trajs[0, :, 0] = new_x
        trajs[0, :, 1] = new_y

        self.evaluation_pdms(trajs, container_in)        

        
        

    
    def evaluation_compute(self, results):
        # aggregate all scores
        all_scores = [v["score"] for k, v in self.planning_metrics.items()]
        all_no_at_fault_collisions = [v["no_at_fault_collisions"] for k, v in self.planning_metrics.items()]
        all_drivable_area_compliance = [v["drivable_area_compliance"] for k, v in self.planning_metrics.items()]
        all_ego_progress = [v["ego_progress"] for k, v in self.planning_metrics.items()]
        all_time_to_collision_within_bound = [v["time_to_collision_within_bound"] for k, v in self.planning_metrics.items()]
        all_comfort = [v["comfort"] for k, v in self.planning_metrics.items()]
        all_driving_direction_compliance = [v["driving_direction_compliance"] for k, v in self.planning_metrics.items()]
        
        # compute avg, 20%, 40%, 60%, 80%, 95%, 99%, 99.9% percentiles, for all these values
        keys = ["score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance"]
        all_values = [all_scores, all_no_at_fault_collisions, all_drivable_area_compliance, all_ego_progress, all_time_to_collision_within_bound, all_comfort, all_driving_direction_compliance]
        
        def calculate_stats(values):
            values = np.array(values)
            avg = np.mean(values)
            percentiles = np.percentile(values, [20, 40, 60, 80, 95, 99, 99.9])
            return avg.item(), percentiles.tolist()
        
        stats = {k: calculate_stats(v) for k, v in zip(keys, all_values)}
        results["planning"] = stats

if __name__ == "__main__":
    prompt_stage = PromptNavsimPlanning(None)
    string = "<PLANNING>Predicted future movement details for the next 5 seconds (sampled at 0.5-second intervals), including distance traveled (in meters) and change in direction (in degrees). A value of 0 for direction indicates moving straight, while positive values represent left turns (with the angle increasing during the initial phase and decreasing back to 0 as the turn completes). The output is formatted as [displacement, theta]: [3.01, 2.0], [2.89, 8.0], [2.83, 12.0], [2.89, 15.0], [2.94, 15.0], [2.94, 11.0], [2.89, 8.0], [2.98, 7.0], [3.13, 4.0], [3.3, 1.0]</PLANNING>"
    
    cfg_scene = DictConfig({"scene_token": "5798a6e25f2553e4"})
    prompt_stage.evaluation_update({"predict": string}, {}, {"scene_metadata": cfg_scene})

    # In [4]: container_in['scene_metadata']
    # Out[4]: SceneMetadata(log_name='2021.09.29.19.02.14_veh-28_03198_03360', scene_token='41e04ef8335e588d', map_name='us-ma-boston', initial_token='7da6ba784b8b5ff0', num_history_frames=4, num_future_frames=10)