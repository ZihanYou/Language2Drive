import re
import traceback
import torch
import numpy as np

from data_engine.datasets.nuscenes.loaders.pipelines.pipeline_blueprint import PromptNuScenesBlueprint
from data_engine.datasets.nuscenes.loaders.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import PlanningMetric


def compute_traj_to_velocity(traj):
    # traj: [N, 2]
    # unit: m/0.5s
    velocity = np.zeros(len(traj))
    # Compute velocity at each point (excluding the first)
    for i in range(1, len(traj)):
        x1, y1 = traj[i - 1][0], traj[i - 1][1]
        x2, y2 = traj[i][0], traj[i][1]

        # Compute velocity
        velocity[i] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # Set the first velocity to the second
    velocity[0] = velocity[1]
    return velocity

def compute_traj_to_dthetas(traj):
    # traj: [N, 2]
    # displacement: [N]
    # Return: [N]
    # dtheta indicates the change in direction in degrees
    thetas = np.zeros(len(traj))
    for i in range(1, len(traj)):
        x1, y1 = traj[i - 1][0], traj[i - 1][1]
        x2, y2 = traj[i][0], traj[i][1]

        # Compute theta
        if np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) < 0.2:
            thetas[i] = 0
        else:
            thetas[i] = np.arctan2(y2 - y1, x2 - x1) / np.pi * 180 - 90

    thetas[0] = 0
    
    dthetas = np.zeros(len(traj))
    for i in range(1, len(traj)):
        dthetas[i] = thetas[i] - thetas[i-1]

    for i in range(1, len(traj)):
        if dthetas[i] > 180:
            dthetas[i] -= 360
        if dthetas[i] < -180:
            dthetas[i] += 360
    
    return dthetas

def compute_traj_to_thetas(traj):
    # traj: [N, 2]
    # displacement: [N]
    # Return: [N]
    # theta indicates the absolute direction in degrees
    thetas = np.zeros(len(traj))
    for i in range(1, len(traj)):
        x1, y1 = traj[i - 1][0], traj[i - 1][1]
        x2, y2 = traj[i][0], traj[i][1]

        # Compute theta
        if np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) < 0.2:
            thetas[i] = 0
        else:
            thetas[i] = np.arctan2(y2 - y1, x2 - x1) / np.pi * 180 - 90

    thetas[0] = thetas[1]
    return thetas

def compute_traj_to_polars(traj, ego_pose):
    polar_distance, polar_angle = np.zeros(len(traj)), np.zeros(len(traj))
    diff_pose = traj - ego_pose[None, ...]
    polar_distance = np.linalg.norm(diff_pose, axis=1)
    polar_angle = np.arctan2(diff_pose[:, 1], diff_pose[:, 0]) / np.pi * 180 - 90
    for i in range(len(traj)):
        if polar_angle[i] > 180:
            polar_angle[i] -= 360
        if polar_angle[i] < -180:
            polar_angle[i] += 360
    # if distance < 0.2, set angle to 0
    for i in range(len(traj)):
        if polar_distance[i] < 0.2:
            polar_angle[i] = 0
    return polar_distance, polar_angle        

def compute_traj_to_dx_dy(traj):
    dx, dy = traj[1:, 0] - traj[:-1, 0], traj[1:, 1] - traj[:-1, 1]
    x, y = np.concatenate([np.zeros(1), dx]), np.concatenate([np.zeros(1), dy]) 
    return x, y

def compute_traj_to_curvature(traj):
    curvature = np.zeros(len(traj))
    
    x = traj[:, 0]
    y = traj[:, 1]
    
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # 计算曲率
    for i in range(len(traj)):
        denominator = (dx[i]**2 + dy[i]**2)**1.5
        if denominator != 0:
            curvature[i] = (dx[i] * ddy[i] - dy[i] * ddx[i]) / denominator
        else:
            curvature[i] = 0
    
    return curvature

def compute_traj_to_curvature_linear(traj):
    # traj: [N, 2]
    curvature = np.zeros(len(traj))
    # Compute curvature at each point (excluding the first and last)
    for i in range(1, len(traj) - 1):
        x1, y1 = traj[i - 1][0], traj[i - 1][1]
        x2, y2 = traj[i][0], traj[i][1]
        x3, y3 = traj[i + 1][0], traj[i + 1][1]

        # Compute side lengths
        L1 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        L2 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        L3 = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

        # Compute triangle area
        area = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        # Compute curvature
        if L1 > 0 and L2 > 0 and L3 > 0:  # Avoid division by zero
            curvature[i] = 4 * area / (L1 * L2 * L3)

    curvature[0] = curvature[1]  # Set the first curvature to the second
    curvature[-1] = curvature[-2]  # Set the last curvature to the second-to-last
    return curvature

def compute_traj_to_curvature_spline(traj):
    from scipy.interpolate import CubicSpline
    t = np.arange(len(traj))
    x = traj[:, 0]
    y = traj[:, 1]
    
    # Fit cubic splines
    spline_x = CubicSpline(t, x)
    spline_y = CubicSpline(t, y)
    
    # Compute derivatives
    dx = spline_x.derivative(1)(t)
    dy = spline_y.derivative(1)(t)
    ddx = spline_x.derivative(2)(t)
    ddy = spline_y.derivative(2)(t)
    
    # Compute curvature
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-5)**1.5
    return curvature

def compute_vel_curvature_to_traj(velocity, curvature):
    # velocity: [N] displacements in every 0.5s
    # curvature: [N]
    # Return: [N, 2]
    
    traj = torch.zeros((len(velocity), 2), device=velocity.device)
    
    t = torch.arange(len(velocity), device=velocity.device, dtype=velocity.dtype)
    thetas = torch.cumsum(curvature * velocity * 0.5, dim=0)
    
    v_x = velocity * torch.cos(thetas + torch.pi / 2)
    v_y = velocity * torch.sin(thetas + torch.pi / 2)
    
    x = torch.cumsum(v_x * 0.5, dim=0)
    y = torch.cumsum(v_y * 0.5, dim=0)
    
    traj[:, 0] = x
    traj[:, 1] = y
    return traj

def compute_vel_dtheta_to_traj(velocity, dthetas):
    # velocity: [N] displacements in every 0.5s
    # dthetas: [N]
    # Return: [N-1, 2]
    
    # left pad zero for velocity and dthetas
    velocity = torch.cat([torch.zeros(1, device=velocity.device), velocity])
    dthetas = torch.cat([torch.zeros(1, device=dthetas.device), dthetas])
    
    dthetas = dthetas / 180 * torch.pi
    thetas = torch.cumsum(dthetas, dim=0)
    
    traj = torch.zeros((len(velocity), 2), device=velocity.device)
    
    for i in range(1, len(velocity)):
        traj[i][0] = traj[i-1][0] + velocity[i] * torch.cos(thetas[i] + torch.pi / 2)
        traj[i][1] = traj[i-1][1] + velocity[i] * torch.sin(thetas[i] + torch.pi / 2)
    
    return traj[1:]

def compute_vel_theta_to_traj(velocity, thetas):
    # left pad 0
    velocity = torch.cat([torch.zeros(1, device=velocity.device), velocity])
    thetas = torch.cat([torch.zeros(1, device=thetas.device), thetas])
    thetas = thetas / 180 * torch.pi
    
    traj = torch.zeros((len(velocity), 2), device=velocity.device)
    for i in range(1, len(velocity)):
        traj[i][0] = traj[i-1][0] + velocity[i] * torch.cos(thetas[i] + torch.pi / 2)
        traj[i][1] = traj[i-1][1] + velocity[i] * torch.sin(thetas[i] + torch.pi / 2)
    return traj[1:]

def compute_dx_dy_to_traj(dx, dy):
    # left pad 0
    dx = torch.cat([torch.zeros(1, device=dx.device), dx])
    dy = torch.cat([torch.zeros(1, device=dy.device), dy])
    
    traj = torch.zeros((len(dx), 2), device=dx.device)
    for i in range(1, len(dx)):
        traj[i][0] = traj[i-1][0] + dx[i]
        traj[i][1] = traj[i-1][1] + dy[i]
    return traj[1:]

def compute_polar_to_traj(polar_distance, polar_angle):
    # polar_distance: [N]
    # polar_angle: [N]
    # Return: [N, 2]
    
    # left pad 0 for polar_distance and polar_angle
    polar_distance = torch.cat([torch.zeros(1, device=polar_distance.device), polar_distance])
    polar_angle = torch.cat([torch.zeros(1, device=polar_angle.device), polar_angle])
    
    polar_angle = polar_angle / 180 * torch.pi
    
    traj = torch.zeros((len(polar_distance), 2), device=polar_distance.device)
    for i in range(1, len(polar_distance)):
        traj[i][0] = polar_distance[i] * torch.cos(polar_angle[i] + torch.pi / 2)
        traj[i][1] = polar_distance[i] * torch.sin(polar_angle[i] + torch.pi / 2)
    return traj[1:]


def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i

class PromptNuScenesPlanning(PromptNuScenesBlueprint):
    def __init__(self, nuscenes, mode="dist-dtheta", viz_every_eval=-1, viz_path=None, can_bus=None, **kwargs):
        super().__init__(nuscenes=nuscenes, cache_filename=None, container_out_key="planning", need_helper=False, **kwargs)
        self.planning_metrics = PlanningMetric()
        self.nuscenes = nuscenes
        self.mode = mode
        self.can_bus = can_bus
        
        self.viz_every_eval = viz_every_eval
        self.viz_path = viz_path
        
        # IN BEV SPACE:
        #          y
        #          ^
        #          |
        #  +theta  |  -theta
        #          |
        #          |
        # ---------o--------> x,  except for x-y mode (x forward, y leftward)
        
        assert self.mode in ["dist-dtheta", "dist-theta", "polar", "dx-dy", "dist-curvature", "x-y"]

        

    def extract_scene_trajs(self, container_out, container_in):
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion

        # 0. read local cache
        # if keys already in container_out, return the cached values
        if 'this_sample_idx' in container_out['buffer_container']:
            return container_out['buffer_container']['ego_all_trajs'], container_out['buffer_container']['this_sample_idx'], container_out['buffer_container']['all_ego_status']

        # 1. scene samples
        this_sample = self.nuscenes.get('sample', container_in['img_metas'].data['token'])
        this_sample_idx = 0  # store the index of this_sample in the scene_samples
        scene_samples = []
        scene_samples.append(this_sample)
        while this_sample['next'] != '':
            this_sample = self.nuscenes.get('sample', this_sample['next'])
            scene_samples.append(this_sample)
        this_sample = self.nuscenes.get('sample', container_in['img_metas'].data['token'])
        while this_sample['prev'] != '':
            this_sample = self.nuscenes.get('sample', this_sample['prev'])
            scene_samples.insert(0, this_sample)
            this_sample_idx += 1

        # 2. this_sample
        this_sample = self.nuscenes.get('sample', container_in['img_metas'].data['token'])
        lidar_token = this_sample['data']['LIDAR_TOP']
        sd_rec = self.nuscenes.get('sample_data', lidar_token)
        cs_record = self.nuscenes.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = self.nuscenes.get('ego_pose', sd_rec['ego_pose_token'])

        # 2.5 get acc
        this_sample = self.nuscenes.get('sample', container_in['img_metas'].data['token'])
        this_scene_token = this_sample['scene_token']
        this_scene = self.nuscenes.get('scene', this_scene_token)
        try:
            pose_msgs = self.can_bus.get_messages(this_scene['name'],'pose')
            steer_msgs = self.can_bus.get_messages(this_scene['name'], 'steeranglefeedback')
            pose_uts = [msg['utime'] for msg in pose_msgs]
            steer_uts = [msg['utime'] for msg in steer_msgs]
            
            all_ego_status = []
            all_utimes = [x['timestamp'] for x in scene_samples]
            for ref_utime in all_utimes:
                ego_status = []
                pose_index = locate_message(pose_uts, ref_utime)
                pose_data = pose_msgs[pose_index]
                steer_index = locate_message(steer_uts, ref_utime)
                steer_data = steer_msgs[steer_index]
                ego_status.extend(pose_data["accel"]) # acceleration in ego vehicle frame, m/s/s
                ego_status.extend(pose_data["rotation_rate"]) # angular velocity in ego vehicle frame, rad/s
                ego_status.extend(pose_data["vel"]) # velocity in ego vehicle frame, m/s
                ego_status.append(steer_data["value"]) # steering angle, positive: left turn, negative: right turn
                all_ego_status.append(ego_status)
        except:
            print(f"Error getting CAN bus data for {this_scene['name']}")
            all_ego_status = None

        # 3. normalize
        ego_all_trajs = np.zeros((len(scene_samples), 3))
        for i in range(len(scene_samples)):
            lidar_sample_data_i = self.nuscenes.get('sample_data', scene_samples[i]['data']['LIDAR_TOP'])
            pose_record_i = self.nuscenes.get('ego_pose', lidar_sample_data_i['ego_pose_token'])
            cs_record_i = self.nuscenes.get('calibrated_sensor', lidar_sample_data_i['calibrated_sensor_token'])
            ego2global_i = transform_matrix(pose_record_i["translation"], Quaternion(pose_record_i["rotation"]), inverse=False)
            sensor2ego_i = transform_matrix(cs_record_i["translation"], Quaternion(cs_record_i["rotation"]), inverse=False)
            pose_mat_i = ego2global_i.dot(sensor2ego_i)
            ego_all_trajs[i] = pose_mat_i[:3, 3]
        ego_all_trajs = ego_all_trajs - np.array(pose_record['translation'])
        rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
        ego_all_trajs = np.dot(rot_mat, ego_all_trajs.T).T
        ego_all_trajs = ego_all_trajs - np.array(cs_record['translation'])
        rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
        ego_all_trajs = np.dot(rot_mat, ego_all_trajs.T).T
        ego_all_trajs = ego_all_trajs[:, :2]
        ego_all_trajs = np.round(ego_all_trajs, 3) # round to 3 decimal places
        
        
        for key, element in zip(['ego_all_trajs', 'all_ego_status'], [ego_all_trajs, all_ego_status]):
            if key not in container_out['buffer_container']:
                container_out['buffer_container'][key] = element
        container_out['buffer_container']['this_sample_idx'] = this_sample_idx
        return ego_all_trajs, this_sample_idx, all_ego_status
    
    def format_output(self, helper_ret, container_out, container_in):
        assert len(helper_ret) == 0
        
        out_string = ""
        out_string += "<PLANNING>"
        out_ignore_flag = False
        
        # 4. compute curvature and velocity
        ego_poses, this_sample_idx, _ = self.extract_scene_trajs(container_out, container_in)
        ego_poses_future = ego_poses[this_sample_idx:this_sample_idx+7]
        if len(ego_poses_future) < 7:
            ego_poses_future = np.pad(ego_poses_future, ((0, 7-len(ego_poses_future)), (0, 0)), 'edge')
            out_ignore_flag = True   # no enough future poses. DROP this sample.

        if self.mode == "dist-dtheta":
            # in current bev plane, what is the change of direction and distance traveled?
            ego_dthetas_future = compute_traj_to_dthetas(ego_poses_future)
            ego_velocity_future = compute_traj_to_velocity(ego_poses_future)
            ego_dthetas_future, ego_velocity_future = np.round(ego_dthetas_future, 0), np.round(ego_velocity_future, 2)
            
            x, y = ego_velocity_future, ego_dthetas_future

            out_string += "Predicted future movement details for the next 3 seconds (sampled at 0.5-second intervals), including distance traveled (in meters) and change in direction (in degrees). A value of 0 for direction indicates moving straight, while positive values represent left turns (with the angle increasing during the initial phase and decreasing back to 0 as the turn completes). The output is formatted as [displacement, theta]: "
            
        elif self.mode == "dist-theta":
            # in current BEV map (xoy plane), which direction does the ego vehicle go?
            ego_thetas_future = compute_traj_to_thetas(ego_poses_future)
            ego_velocity_future = compute_traj_to_velocity(ego_poses_future)
            
            ego_thetas_future, ego_velocity_future = np.round(ego_thetas_future, 0), np.round(ego_velocity_future, 2)
            x, y = ego_velocity_future, ego_thetas_future
            out_string += "Predicted future movement details for the next 3 seconds (sampled at 0.5-second intervals), including distance traveled (in meters) and direction (in degrees). A value of 0 for direction indicates moving straight, while positive values represent left turns. The output is formatted as [displacement, theta]: "
            
        elif self.mode == "polar":
            ego_pose = ego_poses_future[0]
            polar_distance, polar_angle = compute_traj_to_polars(ego_poses_future, ego_pose)
            
            polar_distance, polar_angle = np.round(polar_distance, 2), np.round(polar_angle, 0)
            x, y = polar_distance, polar_angle
            out_string += "Predicted future movement details for the next 3 seconds (sampled at 0.5-second intervals), including distance to current location (in meters) and polar direction (in degrees). A value of 0 for direction indicates moving straight, while positive values represent leftwards. The output is formatted as [distance, polar angle]: "
            
        elif self.mode == "dx-dy":
            x, y = compute_traj_to_dx_dy(ego_poses_future)
            x, y = np.round(x, 2), np.round(y, 2)
            out_string += "Predicted future movement details for the next 3 seconds (sampled at 0.5-second intervals), including displacement in x and y directions (in meters). Positive y means forward direction while positive x means rightwards. The output is formatted as [dx, dy]: "
        
        elif self.mode == "dist-curvature":
            dist = compute_traj_to_velocity(ego_poses_future)
            curvature = compute_traj_to_curvature(ego_poses_future) * 100
            x, y = np.round(dist, 2), np.round(curvature, 2)
            out_string += "Predicted future movement details for the next 3 seconds (sampled at 0.5-second intervals), including distance traveled (in meters) and curvature. The output is formatted as [distance, curvature]: "
        
        elif self.mode == "x-y":
            rightward, forward = ego_poses_future[:, 0], ego_poses_future[:, 1]
            x, y = forward, -rightward
            x, y = np.round(x, 2), np.round(y, 2)
            out_string += "Predicted future movement details for the next 3 seconds (sampled at 0.5-second intervals), including BEV location in x and y directions (in meters). Positive x means forward direction while positive y means leftwards. The output is formatted as [x, y]: "
        
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented.")

        
        for i in range(1, 7):
            xi = x[i]
            yi = y[i]
            if xi == -0.0: xi = 0.0
            if yi == -0.0: yi = 0.0
            out_string += f"[{xi}, {yi}]"
            if i < 6:
                out_string += ", "
        out_string += "</PLANNING>"
        
        if out_ignore_flag:
            out_string = "%DROP%"
        
        return out_string
        

    def evaluation_reset(self):
        self.planning_metrics.reset()
        
    def evaluation_update(self, pred, container_out, container_in, use_gt_str=False):
        # call self.planning_metrics.update(trajs, gt_trajs, gt_trajs_mask, fut_boxes)
        
        # read from pred['predict], and parse the trajectories.
        # 'predict': '<PLANNING>Trajectory points for the next 3 seconds: [-0.02, 5.22], [-0.09, 5.14], [-0.19, 5.04], [-0.31, 4.94], [-0.44, 4.76], [-0.64, 4.61]</PLANNING>'
        
        trajs_str = pred['predict']
        gt_str = ""
        if use_gt_str:
            gt_str = pred['label']
        
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
                traj_reps = torch.tensor(np.array(traj_reps))[:6, :2]

                if self.mode == "dist-dtheta":
                    # if there are 'displacement and theta in matches, drop them
                    vel, dtheta = traj_reps[:, 0], traj_reps[:, 1]
                    trajs = compute_vel_dtheta_to_traj(vel, dtheta,)  # after cumsum, [6, 2]
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
                
                elif self.mode == 'x-y':
                    forward, leftward = traj_reps[:, 0], traj_reps[:, 1]
                    trajs = torch.concatenate([-leftward[..., None], forward[..., None]], dim=-1)  # [6, 2]
                
                else:
                    raise NotImplementedError(f"Mode {self.mode} not implemented.")
                
                # if < 6 points, pad last point
                if trajs.shape[0] < 6:
                    pad = trajs[-1:, :].repeat(6 - trajs.shape[0], 1)
                    trajs = torch.cat([trajs, pad], dim=0)
                
                trajs = trajs[None, ...]  # [1, 6, 2]
                trajs = torch.tensor(trajs)
                
            except:
                trajs = torch.zeros(1, 6, 2)
                traceback.print_exc()
                print(f"Error parsing trajectory: {trajs_str}")
            return trajs
        
        trajs = trajs_str_to_trajs(trajs_str)
        # get the ground truth trajectories
        gt_trajs = container_in['gt_ego_fut_trajs'].data[None, ...]  # [1, 6, 2]
        sdc_planning = gt_trajs.cumsum(dim=-2).unsqueeze(1)  # [1, 1, 6, 2]
        
        gt_trajs_mask = container_in['gt_ego_fut_masks'].data[None, ...]  # [1, 6]
        sdc_planning_mask = gt_trajs_mask.unsqueeze(-1).repeat(1, 1, 2).unsqueeze(1)  # [1, 1, 6, 2]
        
        
        if use_gt_str:
            print("GT string is used, this will lead to a false collision rate.")
            gt_trajs = trajs_str_to_trajs(gt_str)  # [1, 6, 2]
            sdc_planning = gt_trajs.cumsum(dim=-2).unsqueeze(1)  # [1, 1, 6, 2]
            
            gt_trajs_mask = torch.ones(1, 6)  # [1, 6]
            sdc_planning_mask = gt_trajs_mask.unsqueeze(-1).repeat(1, 1, 2).unsqueeze(1)  # [1, 1, 6, 2]
        
        
        if not sdc_planning_mask.all(): ## for incomplete gt, we do not count this sample
            return

        fut_boxes = container_in['fut_boxes'].copy()
        for i in range(len(fut_boxes)):
            fut_boxes[i] = torch.tensor(fut_boxes[i])
            if fut_boxes[i].ndim == 2:
                fut_boxes[i] = fut_boxes[i][None, ...]
        
        if use_gt_str:
            fut_boxes = [torch.zeros(1, 0, 7)] * 6
        
        self.planning_metrics.update(trajs, sdc_planning[0, :, :6, :2], sdc_planning_mask[0,:, :6, :2], fut_boxes)


        if self.viz_every_eval > 0:
            
            this_idx = container_in['idx']
            
            if this_idx % self.viz_every_eval == 0:
                
                this_idx = f"{this_idx:06d}"
            
                planning_metrics_new = PlanningMetric()
                planning_metrics_new.update(trajs, sdc_planning[0, :, :6, :2], sdc_planning_mask[0,:, :6, :2], fut_boxes)
                final_dict = planning_metrics_new.compute()
                final_dict['l2_1s'] = final_dict['L2'][0:1].mean()
                final_dict['l2_2s'] = final_dict['L2'][0:3].mean()
                final_dict['l2_3s'] = final_dict['L2'][0:6].mean()
                final_dict['l2_avg'] = (final_dict['l2_1s'] + final_dict['l2_2s'] + final_dict['l2_3s']) / 3
                
                this_idx = this_idx + f"_{final_dict['l2_avg'].item():.4f}"
            
                query_str = container_out['messages'][0]['content']
                prediction_str = pred['predict']
                gt_str = container_out['messages'][1]['content']
                
                
                output_dir = f"{self.viz_path}/{this_idx}"
                
                import os
                # write query_str to `self.viz_path/{idx}/query.txt`
                os.makedirs(f"{output_dir}", exist_ok=True)
                text_content = ""
                with open(f"{output_dir}/all_in_one.txt", 'w') as f:
                    text_content += f"== Metrics ==\n"
                    text_content += f"{str(final_dict)}\n"
                    text_content += f"== Query ==\n"
                    text_content += f"{query_str}\n"
                    text_content += f"== Ground Truth ==\n"
                    text_content += f"{gt_str}\n"
                    text_content += f"== Prediction ==\n"
                    text_content += f"{prediction_str}\n"
                    f.write(text_content)

                    
                from PIL import Image
                image_paths = container_out['images']
                target_size = (576, 384)
                output_dir_imgs = os.path.join(output_dir, "images")
                output_img_names = []
                
                os.makedirs(output_dir_imgs, exist_ok=True)

                # Process each image
                for image_path in image_paths:
                    # Extract the camera status from the path
                    camera_status = image_path.split('/')[-2]
                    
                    # Open the image
                    with Image.open(image_path) as img:
                        # Resize the image
                        img_resized = img.resize(target_size, Image.Resampling.NEAREST)
                        
                        # Create the output file name
                        output_file_name = f"{camera_status}.jpg"
                        output_file_path = os.path.join(output_dir_imgs, output_file_name)
                        output_img_names.append(output_file_name)
                        
                        # Save the resized image
                        img_resized.save(output_file_path, "JPEG")

                # Create the HTML content
                html_file_path = os.path.join(output_dir, "all_in_one.html")

                # html escape text_content
                text_content = text_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

                html_content = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>All In One</title>
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            margin: 20px;
                        }}
                        h2 {{
                            color: #333;
                        }}
                        p {{
                            font-family: monospace;
                        }}
                        .section {{
                            margin-bottom: 20px;
                        }}
                        .images {{
                            display: flex;
                            flex-wrap: wrap;
                        }}
                        .images img {{
                            margin: 5px;
                            max-width: 200px;
                        }}
                    </style>
                </head>
                <body>
                    <div class="section">
                        <h2>Metrics</h2>
                        <p>{text_content.split("== Metrics ==")[1].split("== Query ==")[0].strip()}</p>
                    </div>
                    <div class="section">
                        <h2>Query</h2>
                        <p>{text_content.split("== Query ==")[1].split("== Ground Truth ==")[0].strip()}</p>
                    </div>
                    <div class="section">
                        <h2>Ground Truth</h2>
                        <p>{text_content.split("== Ground Truth ==")[1].split("== Prediction ==")[0].strip()}</p>
                    </div>
                    <div class="section">
                        <h2>Prediction</h2>
                        <p>{text_content.split("== Prediction ==")[1].strip()}</p>
                    </div>
                    <div class="section">
                        <h2>Images</h2>
                        <div class="images">
                """

                # Add the images to the HTML content
                for image_file in output_img_names:
                    image_path = os.path.join(output_dir_imgs, image_file)
                    html_content += f'            <img src="images/{image_file}" alt="{image_file}">\n'

                # Close the HTML tags
                html_content += """
                        </div>
                    </div>
                </body>
                </html>
                """

                # Write the HTML content to a file
                with open(html_file_path, 'w') as f:
                    f.write(html_content)


                

    
    def evaluation_compute(self, results):
        final_dict = self.planning_metrics.compute()
        final_dict = {
            k: v.tolist() if isinstance(v, torch.Tensor) else v
            for k, v in final_dict.items()
        }
        self.evaluation_reset()
        
        from prettytable import PrettyTable
        planning_tab = PrettyTable()
        metric_dict = {}
        planning_tab.field_names = [
        "metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s", "avg"]
        for key in final_dict.keys():
            value = final_dict[key]
            new_values = []
            for i in range(len(value)):
                new_values.append(np.array(value[:i+1]).mean())
            value = new_values
            avg = [value[1], value[3], value[5]]
            avg = sum(avg) / len(avg)
            value.append(avg)
            metric_dict[key] = avg
            row_value = []
            row_value.append(key)
            for i in range(len(value)):
                if 'col' in key:
                    row_value.append('%.3f' % float(value[i]*100) + '%')
                else:
                    row_value.append('%.4f' % float(value[i]))
            planning_tab.add_row(row_value)
        
        final_dict['table'] = planning_tab.get_string()
        results["planning"] = final_dict
        return results



if __name__ == "__main__":
    prompt_stage = PromptNuScenesPlanning(None)
    string = "<PLANNING>Predicted future movement details for the next 3 seconds (sampled at 0.5-second intervals), including distance traveled (in meters) and change in direction (in degrees). A value of 0 for direction indicates moving straight, while positive values represent left turns (with the angle increasing during the initial phase and decreasing back to 0 as the turn completes). The output is formatted as [displacement, theta]: [2.44, -1.0], [2.26, 0.0], [1.97, 0.0], [1.81, 0.0], [1.78, -1.0], [1.79, 0.0]</PLANNING>"
    prompt_stage.evaluation_update({"predict": string},{}, {"gt_ego_fut_trajs": torch.zeros(1, 6, 2), "gt_ego_fut_masks": torch.ones(1, 6), "fut_boxes": []})