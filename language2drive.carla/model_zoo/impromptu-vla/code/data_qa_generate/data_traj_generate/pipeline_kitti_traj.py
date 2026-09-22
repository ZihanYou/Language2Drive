import os
import numpy as np
import json
from pathlib import Path
import sys
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
sys_path=f"{project_root}/data_qa_generate/data_traj_generate"
if sys_path not in sys.path:
    sys.path.append(sys_path)
from kitti_utils import load_oxts_packets_and_poses, rotz
# print(project_root)
data_dir = project_root / "data_raw" / "kitti" 
raw_hz=10
step_by_hz=raw_hz//2

class PromptKittiPlanning:
    def __init__(self,input_path,output_path,max_previous_samples,max_next_samples):
        self.input_path = input_path
        self.output_path = output_path
        self.max_previous_samples =max_previous_samples  
        self.max_next_samples =max_next_samples    
        

    def process(self):
        for input_dir, output_file in zip(self.input_path, self.output_path):
            print(f"正在处理目录: {input_dir}")
            txt_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')])
            if not txt_files:
                print(f"目录 {input_dir} 下没有找到任何txt文件，跳过处理。")
                continue

            self.input_files = txt_files
            data = self.read_input_data()
            grouped_data = self.group_by_scene(data)
            processed = self.process_groups(grouped_data)
            self.save_output_traj(processed, output_file)

    def read_input_data(self):
        oxts_data_list = []  
        for file_path in self.input_files:
            oxts_data = load_oxts_packets_and_poses([file_path]) 
            if not oxts_data:
                continue
            oxts_data_list.append(oxts_data)  

        return oxts_data_list


    def group_by_scene(self, data):
        
        return data 

    def process_groups(self, grouped_data):
        all_entries = []
        for scene_idx, scene_data in enumerate(grouped_data):
            indices = [i for i in range(0, len(scene_data), step_by_hz)]
            subsscene_data = [scene_data[i] for i in indices]
            scene_entries = self.extract_scene_trajs(subsscene_data, scene_idx)
            all_entries.extend(scene_entries)
        return all_entries

    def extract_scene_trajs(self, scene_data, scene_id):
        outputs = []
        enu_positions = [data.T_w_imu[:3, 3] for data in scene_data]
        N = len(enu_positions)

        for i in range(0, len(scene_data)):
            data = scene_data[i]
            R_sample = data.T_w_imu[:3, :3]
            yaw = np.arctan2(R_sample[1, 0], R_sample[0, 0])
            yaw = self.normalize_angle_neg_pi_to_pi(yaw)
            R_ego = rotz(np.pi / 2 - yaw)

            lower_bound = max(0, i - self.max_previous_samples)
            upper_bound = min(N, i + self.max_next_samples + 1)
            traj_indices = list(range(lower_bound, upper_bound))

            traj_sample = []
            current_ego_pos = None
            
            ego_frame_points = []
            for idx in traj_indices:
                ego_point = R_ego.dot(enu_positions[idx])
                ego_frame_points.append(ego_point)
                if idx == i:
                    current_ego_pos = ego_point
            
            for ego_point in ego_frame_points:
                relative_pos = ego_point[:2] - current_ego_pos[:2]
                traj_sample.append(relative_pos.tolist())
                
            traj_sample = self.clean_trajectory(traj_sample, self.max_next_samples, self.max_previous_samples)

            rotation_angles = {
                'roll': data.packet.roll,
                'pitch': data.packet.pitch,
                'yaw': yaw
            }

            outputs.append({
                'scene_id': scene_id,
                'frame_id': i*step_by_hz,
                'rotation_angles': rotation_angles,
                'trajectory': traj_sample
            })
        return outputs

    def clean_trajectory(self,trajectory, max_next_samples, max_previous_samples):   
        trajectory = [list(point) for point in trajectory] 
        trajectory = [[point[1], -point[0]] for point in trajectory]
        if len(trajectory) > 1:
            if max_next_samples == 0:    
                if trajectory[-1] == [0.0, -0.0]:
                    trajectory = trajectory[:-1]             
            if max_previous_samples == 0:        
                if trajectory[0] == [0.0, -0.0]:
                    trajectory = trajectory[1:]
        trajectory = np.array(trajectory)
        trajectory = np.round(trajectory, 2).tolist()
        return trajectory

    def save_output_traj(self, processed, output_file):
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, 'w') as f:
            for entry in processed:
                f.write(json.dumps(entry) + '\n')
            print("output:",output_file)

    def normalize_angle_neg_pi_to_pi(self, angle):
        import math
        angle = angle % (2 * math.pi)  # 先归一化到 [0, 2π]
        if angle > math.pi:
            angle -= 2 * math.pi
        return angle

if __name__ == "__main__":
    kitti_planning = PromptKittiPlanning(
        input_path=[
            f'{data_dir}/data_tracking_oxts/training/oxts',
            f'{data_dir}/data_tracking_oxts/testing/oxts'
        ],
        output_path=[
            f"{project_root}/data_qa_generate/data_traj_generate/data_traj_results/kitti/traj_kitti_train_future10.jsonl",
            f"{project_root}/data_qa_generate/data_traj_generate/data_traj_results/kitti/traj_kitti_test_future10.jsonl"
        ],
        max_previous_samples=0,
        max_next_samples=10
    )
    kitti_planning.process()

    kitti_planning = PromptKittiPlanning(
        input_path=[
            f'{data_dir}/data_tracking_oxts/training/oxts',
            f'{data_dir}/data_tracking_oxts/testing/oxts'
        ],
        output_path=[
            f"{project_root}/data_qa_generate/data_traj_generate/data_traj_results/kitti/traj_kitti_train_last3.jsonl",
            f"{project_root}/data_qa_generate/data_traj_generate/data_traj_results/kitti/traj_kitti_test_last3.jsonl"
        ],
        max_previous_samples=3,
        max_next_samples=0
    )
    kitti_planning.process()