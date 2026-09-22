import os
import numpy as np
from data_engine.datasets.nuscenes.loaders.pipelines.pipeline_blueprint import PromptNuScenesBlueprint
from data_engine.datasets.nuscenes.loaders.pipelines.pipeline_planning import *


class PromptNuScenesEgoStatus(PromptNuScenesPlanning):
    def __init__(self, nuscenes=None, mode="dist-dtheta", **kwargs):
        super().__init__(nuscenes=nuscenes, mode=mode, **kwargs)
    
    def __call__(self, container_out, container_in):
        this_id = container_in['img_metas'].data['token']
        
        ego_poses, this_sample_idx, all_ego_status = self.extract_scene_trajs(container_out, container_in)
        
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
            x = x[max(0, this_sample_idx-5): this_sample_idx+1]
            y = y[max(0, this_sample_idx-5): this_sample_idx+1]
            if all_ego_status is not None:
                all_ego_status = all_ego_status[max(0, this_sample_idx-5): this_sample_idx+1]
        else:
            # fu_x = x[this_sample_idx: this_sample_idx+10]
            # fu_y = y[this_sample_idx: this_sample_idx+10]
            x = x[max(0, this_sample_idx-6): this_sample_idx + 1]
            y = y[max(0, this_sample_idx-6): this_sample_idx + 1]
            if all_ego_status is not None:
                all_ego_status = all_ego_status[max(0, this_sample_idx-6): this_sample_idx + 1]
        # if len(previous_ego_dthetas) < 6:
        #     previous_ego_dthetas = np.pad(previous_ego_dthetas, ((6-len(previous_ego_dthetas), 0),), mode='edge')
        #     previous_ego_velocity = np.pad(previous_ego_velocity, ((6-len(previous_ego_velocity), 0),), mode='edge')
            
        available_time = 0.5 * len(x)
        if self.mode == "x-y":
            available_time = 0.5 * len(x) - 0.5
        # round to 1 decimal place
        available_time = round(available_time, 1)

        # ego_status = container_in['ego_status'].data
        # ego_status.extend(pose_data["accel"]) # acceleration in ego vehicle frame, m/s/s, 3 elements
        # ego_status.extend(pose_data["rotation_rate"]) # angular velocity in ego vehicle frame, rad/s, 3 elements
        # ego_status.extend(pose_data["vel"]) # velocity in ego vehicle frame, m/s, 3 elements
        # ego_status.append(steer_data["value"]) # steering angle, positive: left turn, negative: right turn, 1 element
        
        # ego_status_string += "You are given with ego vehicle status:\n"
        # ego_status_string += f"Acceleration: {self.format_data(ego_status[:3])} m/s^2\n"
        # ego_status_string += f"Angular velocity: {self.format_data(ego_status[3:6])} rad/s\n"
        # ego_status_string += f"Velocity: {self.format_data(ego_status[6:9])} m/s\n"
        # ego_status_string += f"Steering angle: {ego_status[9]} (positive: left turn, negative: right turn)\n"
        
        ego_status_string = ""
        if len(x) > 0:
            ego_status_string += f"Provided are the previous ego vehicle status recorded over the last {available_time} seconds (at 0.5-second intervals). {mode_prompt}" 
            
            if all_ego_status is not None:
                "We also include [acc_x, acc_y, vel, steering_angle] signals in previous 3 seconds, where acc_x and acc_y are the acceleration in the x and y directions (m/s^2), vel are the velocity of ego vehicle (m/s), and steering_angle is the steering angle of the ego vehicle (left turn is positive, right turn is negative). Note the difference between the steering angles and y coordinates is that steering angles provide an immediate indication of the direction in which the vehicle is steering. It's important to accumulate these angles over time to understand the overall direction. For example, if there are multiple right steers followed by a sudden left steer, the actual path is still turning right. In contrast, the y coordinates reflect the resultant position over time, showing the actual path taken by the vehicle.\n"
            

            def convert_status(status) -> str:
                acc_x, acc_y = round(status[0], 2), round(status[1], 2)
                vel_x, vel_y = round(status[6], 2), round(status[7], 2)
                steering_angle = round(status[9], 2)
                # return f"Acceleration: X {acc_x}, Y {acc_y} m/s^2, Velocity: X {vel_x}, Y {vel_y} m/s"
            # , Steering angle: {steering_angle} (positive: left turn, negative: right turn)"
                return f"Acceleration: X {acc_x}, Y {acc_y} m/s^2, Velocity: X {vel_x}, Steering angle: {steering_angle} (positive: left turn, negative: right turn)"

            # ego_status_string += f"(t-3s) [{previous_ego_velocity[0]}, {previous_ego_dthetas[0]}], (t-2.5s) [{previous_ego_velocity[1]}, {previous_ego_dthetas[1]}], (t-2s) [{previous_ego_velocity[2]}, {previous_ego_dthetas[2]}], (t-1.5s) [{previous_ego_velocity[3]}, {previous_ego_dthetas[3]}], (t-1s) [{previous_ego_velocity[4]}, {previous_ego_dthetas[4]}], (t-0.5s) [{previous_ego_velocity[5]}, {previous_ego_dthetas[5]}]\n"
            for i in range(len(x)):
                xi, yi = x[i], y[i]
                if xi == -0.0:
                    xi = 0.0
                if yi == -0.0:
                    yi = 0.0
                if all_ego_status is not None:
                    ego_status_i = all_ego_status[i]
                    ego_status_string += f"(t-{available_time - i*0.5}s) [{xi}, {yi}], {convert_status(ego_status_i)}"
                        
                else:
                    ego_status_i = None
                    
                    ego_status_string += f"(t-{available_time - i*0.5}s) [{xi}, {yi}]"
                
                
                if i != len(x) - 1:
                    ego_status_string += ", "
                else:
                    ego_status_string += "\n"
        
        
        container_out['messages'][0]['content'] += ego_status_string
        return container_out
        # return ego_status_string, ", ".join([f"[{x:.2f}, {y:.2f}]" for x, y in zip(fu_x, fu_y)]), len(x), len(fu_x), (all_ego_status is not None)


    def evaluation_update(self, pred, container_out, container_in, use_gt_str=False):
        pass

    def evaluation_compute(self, results):
        pass