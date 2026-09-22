import os
import numpy as np
from data_engine.datasets.navsim.loaders.pipelines.pipeline_blueprint import PromptNavsimBlueprint
from data_engine.datasets.navsim.loaders.pipelines.pipeline_planning import *

class PromptNavsimEgoStatus(PromptNavsimPlanning):
    def __init__(self, navsim=None, mode="dist-dtheta", **kwargs):
        super().__init__(navsim=navsim, mode=mode, **kwargs)
    
    def __call__(self, container_out, container_in):
        this_id = container_in['scene_metadata'].scene_token
        
        ego_poses, this_sample_idx = self.extract_scene_trajs(container_out, container_in)
        
        mode_prompt = ""
        
        if self.mode == "dist-dtheta":
            ego_dthetas = compute_traj_to_dthetas(ego_poses)
            ego_velocity = compute_traj_to_velocity(ego_poses)
            ego_dthetas, ego_velocity = np.round(ego_dthetas, 0), np.round(ego_velocity, 2)
            x, y = ego_velocity, ego_dthetas
            mode_prompt = "This includes the distance traveled during each 0.5-second interval (in meters) and the change in direction (in degrees). A direction value of 0 indicates moving straight, while positive values represent left turns (with the angle increasing during the initial phase of the turn and decreasing back to 0 as the turn concludes). The data is presented in the format [distance, direction]:"
        elif self.mode == "dist-theta":
            ego_velocity = compute_traj_to_velocity(ego_poses)
            ego_thetas = compute_traj_to_thetas(ego_poses)
            x, y = ego_velocity, ego_thetas
            x, y = np.round(x, 2), np.round(y, 0)
            mode_prompt = "This includes the distance traveled during each 0.5-second interval (in meters) and the direction (in degrees). A direction value of 0 indicates moving straight, while positive values represent left turns. The data is presented in the format [distance, direction]:"
        elif self.mode == "dist-curvature":
            ego_velocity = compute_traj_to_velocity(ego_poses)
            ego_curvatures = compute_traj_to_curvature(ego_poses)
            x, y = ego_velocity, ego_curvatures * 100
            x, y = np.round(x, 2), np.round(y, 2)
            mode_prompt = "This includes the distance traveled during each 0.5-second interval (in meters) and the curvature. A curvature value of 0 indicates moving straight. The data is presented in the format [distance, curvature]:"
        elif self.mode == "polar":
            ego_pose = ego_poses[this_sample_idx]
            polar_distance, polar_angle = compute_traj_to_polars(ego_poses, ego_pose)
            polar_distance, polar_angle = np.round(polar_distance, 2), np.round(polar_angle, 0)
            x, y = polar_distance, polar_angle
            mode_prompt = "This includes the distance and angle of the ego vehicle relative to the current position. The data is presented in the format [polar distance, polar angle]:"
        elif self.mode == "dx-dy":
            dx, dy = compute_traj_to_dx_dy(ego_poses)
            dx, dy = np.round(dx, 2), np.round(dy, 2)
            x, y = dx, dy
            mode_prompt = "This includes the change in x and y coordinates of the ego vehicle during each 0.5-second interval (in meters). Positive y means forward direction while positive x means rightwards. The data is presented in the format [dx, dy]:"
            
        elif self.mode == "x-y":
            rightward, forward = ego_poses[:, 0], ego_poses[:, 1]
            x, y = forward, -rightward
            x, y = np.round(x, 2), np.round(y, 2)
            
            mode_prompt = "This includes the x and y coordinates of the ego vehicle. Positive x means forward direction while positive y means leftwards. The data is presented in the format [x, y]:"
        
        if self.mode != "x-y":
            new_x = x[1: this_sample_idx+1]
            new_y = y[1: this_sample_idx+1]
            x, y = new_x, new_y
        else:
            new_x = x[0: this_sample_idx + 1]
            new_y = y[0: this_sample_idx + 1]
            x, y = new_x, new_y
        
        available_time = 0.5 * len(x)
        if self.mode == "x-y":
            available_time = 0.5 * len(x) - 0.5
        available_time = round(available_time, 1)
        
        meta_ego_status_all = container_in['ego_status'][:this_sample_idx + 1]
        
        ego_status_string = ""
        if len(x) > 0:
            ego_status_string += f"Provided are the previous ego vehicle statuses recorded over the last {available_time} seconds (at 0.5-second intervals). {mode_prompt}\n"

            if meta_ego_status_all:
                "We also include [acc_x, acc_y, vel_x, vel_y] signals in previous 3 seconds, where acc_x and acc_y are the acceleration in the x and y directions (m/s^2), vel_x and vel_y are the velocity of ego vehicle (m/s). The definiton is the same as the coordinates, where positive x means forward direction while positive y means leftwards."
            
            def convert_status(status) -> str:
                velocity = status.ego_velocity
                acc = status.ego_acceleration
                acc_x, acc_y = acc[0].item(), acc[1].item()
                vel_x, vel_y = velocity[0].item(), velocity[1].item()
                acc_x, acc_y, vel_x, vel_y = round(acc_x, 2), round(acc_y, 2), round(vel_x, 2), round(vel_y, 2)
                return f"Acceleration: X {acc_x}, Y {acc_y} m/s^2, Velocity: X {vel_x}, Y {vel_y} m/s"

            
            # ego_status_string += f"(t-3s) [{previous_ego_velocity[0]}, {previous_ego_dthetas[0]}], (t-2.5s) [{previous_ego_velocity[1]}, {previous_ego_dthetas[1]}], (t-2s) [{previous_ego_velocity[2]}, {previous_ego_dthetas[2]}], (t-1.5s) [{previous_ego_velocity[3]}, {previous_ego_dthetas[3]}], (t-1s) [{previous_ego_velocity[4]}, {previous_ego_dthetas[4]}], (t-0.5s) [{previous_ego_velocity[5]}, {previous_ego_dthetas[5]}]\n"
            for i in range(len(x)):
                ego_status_string += f"(t-{available_time - i*0.5}s) [{x[i]}, {y[i]}], {convert_status(meta_ego_status_all[i])}"
                if i != len(x) - 1:
                    ego_status_string += ", "
                else:
                    ego_status_string += "\n"
        
        container_out['messages'][0]['content'] += ego_status_string
        return container_out

    def evaluation_update(self, pred, container_out, container_in, use_gt_str=False):
        pass

    def evaluation_compute(self, results):
        pass