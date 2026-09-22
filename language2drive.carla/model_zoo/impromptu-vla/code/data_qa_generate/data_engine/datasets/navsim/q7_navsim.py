import os
import json
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from pathlib import Path
import sys
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.parent.parent
sys.path.append(f"{project_root}/data_qa_generate/")
from data_engine.datasets.navsim.dataset_navsim import VLMNavsim
qa_root = project_root / "data_qa_results" / "navsim"
os.makedirs(qa_root, exist_ok=True)
# Configuration constants
class Config:
    NUM_FUT = 4
    NUM_FUT_NAVI = 12
    VEL_NAVI_THRESH = 4.0
    VEL_DIFF_THRESH = 3.0
    VAL_STOP = 2.0
    LAT_THRESH = 2
    ANGLE_THRESH = 30.0
    ANGLE_THRESH_NAVI = 8.0  # abused
    DATA_FPS = 2
    TARGET_FPS = 2
    
    # Navigation command thresholds
    NAVI_DIS_FORWARD_THRESH = 1000.0
    NAVI_DIS_THRESH = 10.0
    
    QA_ROOT =qa_root
    BASE_PATH =project_root

def get_scene_data(dataset: VLMNavsim) -> Dict[str, Dict[str, Any]]:
    """
    Organize data by scenario, including trajectory, velocity, and acceleration information
    Return structure: 
    {
        "scene_token_1": {
            "metadata": {
                "log_name": str,
                "scene_token": str,
                "map_name": str,
                "num_frames": int
            },
            "frames": [
                {
                    "timestamp": float,
                    "position": [x, y, z],
                    "rotation": [qx, qy, qz, qw],
                    "velocity": [vx, vy, vz],
                    "acceleration": [ax, ay, az],
                    "images": List[str]  # 图像路径列表
                },
                ...
            ]
        },
        ...
    }
    """
    scene_data = defaultdict(dict)
    
    for idx in tqdm(range(len(dataset)), desc="Processing scenes"):
        container = dataset.get_container_in(idx)
        scene_metadata = container["scene_metadata"]
        frame_data = container["frame_data"]
        
        scene_token = scene_metadata.scene_token

        if scene_token not in scene_data:
            scene_data[scene_token] = {
                "metadata": {
                    "log_name": scene_metadata.log_name,
                    "scene_token": scene_token,
                    "map_name": scene_metadata.map_name,
                    "num_frames": 0
                },
                "frames": []
            }
        
        for frame in frame_data:
            # 提取信息
            ego_status = frame["ego_status"]
            frame_info = {
                "timestamp": frame["timestamp"],
                "position": ego_status.ego_pose[:3].tolist(),
                "rotation": ego_status.ego_pose[3:].tolist(),
                "velocity": ego_status.ego_velocity.tolist(),
                "acceleration": ego_status.ego_acceleration.tolist(),
                "images": [
                    os.path.join(Config.BASE_PATH, cam["image_path"]) 
                    for cam in frame["cameras"].values() 
                    if "image_path" in cam
                ]
            }
            
            scene_data[scene_token]["frames"].append(frame_info)
            scene_data[scene_token]["metadata"]["num_frames"] += 1
        
    return dict(scene_data)

def get_scene_data(dataset: VLMNavsim) -> Dict[str, Dict[str, Any]]:
  
    scene_data = defaultdict(dict)
    
    for idx in tqdm(range(len(dataset)), desc="Processing scenes"):
        container = dataset.get_container_in(idx)
        scene_metadata = container["scene_metadata"]
        frame_data = container["frame_data"]
        
        scene_token = scene_metadata.scene_token
        
        if scene_token not in scene_data:
            scene_data[scene_token] = {
                "metadata": {
                    "log_name": scene_metadata.log_name,
                    "scene_token": scene_token,
                    "map_name": scene_metadata.map_name,
                    "num_frames": 0
                },
                "frames": []
            }
        
    
        for frame in frame_data:
            ego_status = frame["ego_status"]
            
          
            quat = ego_status.ego_pose[3:]
            yaw = Rotation.from_quat(quat).as_euler('zyx')[0]
            
            frame_info = {
                "timestamp": frame["timestamp"],
                "position": ego_status.ego_pose[:3].tolist(),
                "rotation": quat.tolist(),
                "yaw": float(yaw),  # 存储计算后的航向角
                "velocity": ego_status.ego_velocity.tolist(),
                "acceleration": ego_status.ego_acceleration.tolist(),
                "images": [
                    os.path.join(Config.BASE_PATH, cam["image_path"]) 
                    for cam in frame["cameras"].values() 
                    if "image_path" in cam
                ]
            }
            
            scene_data[scene_token]["frames"].append(frame_info)
            scene_data[scene_token]["metadata"]["num_frames"] += 1
    
   
    return {
        k: v for k, v in dict(scene_data).items() 
        if v["metadata"]["num_frames"] >= Config.MIN_SCENE_FRAMES
    }

def calculate_angle(x, y):
    """Calculate angle in degrees from x and y coordinates"""
    angle = math.degrees(math.atan2(x, -y))
    return angle if angle >= 0 else angle + 360

def point_to_line_distance(points):
    """Calculate minimum distance from last point to the line formed by first two points"""
    if len(points) < 3:
        raise ValueError("At least three points required")
    
    points = np.asarray(points)
    (x1, y1), (x2, y2), (xn, yn) = points[0], points[1], points[-1]
    
    if np.allclose([x1, y1], [x2, y2]):
        return np.sqrt((xn - x1)**2 + (yn - y1)**2)  # Fallback to point-to-point distance
    
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    
    numerator = np.abs(A * xn + B * yn + C)
    denominator = np.sqrt(A**2 + B**2)
    return float(numerator / denominator) if denominator > 1e-10 else 0.0

def point_to_line_projection_distance(points):
    """Calculate projection distance of last point onto the line formed by first two points"""
    if len(points) < 3:
        raise ValueError("At least three points required")
    
    x1, y1 = points[0]
    x2, y2 = points[1]
    xn, yn = points[-1]
    dx, dy = x2 - x1, y2 - y1
    vx, vy = xn - x1, yn - y1
    
    line_length = (dx ** 2 + dy ** 2) ** 0.5
    if line_length == 0 or np.isnan(line_length):
        return 0

    return (vx * dx + vy * dy) / line_length

def global_to_ego(global_x, global_y, ego_x, ego_y, ego_yaw_rad):
    """Convert global coordinates to ego-centric coordinates"""
    delta_x = global_x - ego_x
    delta_y = global_y - ego_y
    cos_theta = math.cos(ego_yaw_rad)
    sin_theta = math.sin(ego_yaw_rad)
    return (
        delta_x * cos_theta + delta_y * sin_theta,
        -delta_x * sin_theta + delta_y * cos_theta
    )

def generate_q7_for_scene(images, scene_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
   
    qas = []
    
 
    enu_positions = [frame["position"] for frame in scene_frames]
    velocities = [frame["velocity"] for frame in scene_frames]
    accelerations = [frame["acceleration"] for frame in scene_frames]
    
    # 为每个有效帧生成Q7数据
    for i in range(len(scene_frames)):
        if i < 3 or i + 10 >= len(scene_frames):
            continue  
            
        current_ego_x, current_ego_y, current_yaw = enu_positions[i][0], enu_positions[i][1], enu_positions[i][2]
            
        
        status_lines = []
        for time_offset, idx in [(-1.5, i-3), (-1.0, i-2), (-0.5, i-1)]:
            if idx < 0:
                continue
                
         
            global_x, global_y = enu_positions[idx][0], enu_positions[idx][1]
            ego_x_new, ego_y_new = global_to_ego(
                global_x, global_y,
                current_ego_x, current_ego_y,
                current_yaw
            )
            
            acc_x, acc_y = accelerations[idx][0], accelerations[idx][1]
            vel_x, vel_y = velocities[idx][0], velocities[idx][1]
            
            status_lines.append(
                f"(t{time_offset:+}s) [{ego_x_new:.2f}, {ego_y_new:.2f}], "
                f"Acceleration: X {acc_x:.2f}, Y {acc_y:.2f} m/s^2, "
                f"Velocity: X {vel_x:.2f}, Y {vel_y:.2f} m/s"
            )
        
        status_lines_future = []
        for idx in range(i+1, i+11):
            if idx >= len(enu_positions):
                break
                
            global_x, global_y = enu_positions[idx][0], enu_positions[idx][1]
            ego_x_new, ego_y_new = global_to_ego(
                global_x, global_y,
                current_ego_x, current_ego_y,
                current_yaw
            )
            status_lines_future.append(f"[{ego_x_new:.2f}, {ego_y_new:.2f}]")
        

        if len(status_lines) != 3 or len(status_lines_future) != 10:
            continue
      
        question = (
            "You are an autonomous driving agent. You have access to multi-view camera images of a vehicle: "
            "(1) front view (which you should focus on with the most attention) <image>, "
            "(2) front right view <image>, and (3) front left view <image>. "
            "Your task is to do your best to predict future waypoints for the vehicle over the next 10 timesteps, "
            "given the vehicle's intent inferred from the images.\n\n"
            "Provided are the previous ego vehicle statuses recorded over the last 1.5 seconds "
            "(at 0.5-second intervals). This includes the x and y coordinates of the ego vehicle. "
            "Positive x means forward direction while positive y means leftwards.\n"f"{', '.join(status_lines)}\n"
        )
        
        answer = (
            'Predicted future movement details for the next 5 seconds (sampled at 0.5-second intervals), '
            'including BEV location in x and y directions (in meters). Positive x means forward direction '
            'while positive y means leftwards. The output is formatted as [x, y]: \n'
            f"{', '.join(status_lines_future)}\n"
        )
        
     
        qas.append({
            "images": [images[i][0], images[i][2], images[i][1]],  # 前视、右前、左前
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })
    
    return qas

def generate_q7_data(mode,images,dataset: VLMNavsim, output_dir: str):

    scene_data = get_scene_data(dataset)

    all_q7_data = []
    cnt_frame=0
    for scene_token, data in tqdm(scene_data.items(), desc="Generating Q7 data per scene"):
        num_frames_this_scene = scene_data[scene_token]["metadata"]["num_frames"]
        scene_q7 = generate_q7_for_scene(images[cnt_frame:cnt_frame+num_frames_this_scene],data["frames"])
        cnt_frame+=num_frames_this_scene
        all_q7_data.extend(scene_q7)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"q7_{mode}.json")
    with open(output_path, "w") as f:
        json.dump(all_q7_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved Q7 data for {len(all_q7_data)} samples to {output_path}")

if __name__ == "__main__":
    for mode in ["test", "train"]:
    # for mode in ["test"]:
        dataset = VLMNavsim(mode=mode)
        images=[]
        for sid in tqdm(range(len(dataset)), desc=f"Processing {mode} samples"):
            
            images.append([
                os.path.join(Config.BASE_PATH, cam["image_path"]) 
                for cam in dataset.get_container_in(sid)["frame_data"][3]["cameras"].values() 
                if "image_path" in cam
            ])
        generate_q7_data(mode,images,dataset, Config.QA_ROOT)