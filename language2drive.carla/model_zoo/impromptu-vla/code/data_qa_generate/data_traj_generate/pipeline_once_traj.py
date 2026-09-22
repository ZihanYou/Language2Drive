import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as R
scene_sum_num=581
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
data_dir = project_root / "data_raw"
modes = ['test', 'val', 'train']

class OnceProcessor:
    def __init__(self, base_path, output_file, max_previous_samples=0, max_next_samples=10):
        self.base_path = base_path
        self.output_file = output_file
        self.max_previous_samples = max_previous_samples
        self.max_next_samples = max_next_samples

    def process(self):
        data = self.read_input_data()
        grouped_data = self.group_by_scene(data)
        processed_data = self.process_groups(grouped_data)
        self.save_output(processed_data)

    def read_input_data(self):
        data = []
        for sequence_num in range(scene_sum_num):
            folder_name = f"{sequence_num:06d}"
            file_path = os.path.join(self.base_path, folder_name, f"{folder_name}.json")
            try:
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                    file_data['scene_id'] = folder_name 
                    data.append(file_data)
            except FileNotFoundError:
                print(f"File {file_path} not found. Skipping.")
                continue
            except json.JSONDecodeError as e:
                print(f"JSON decode error in {file_path}: {e}")
                continue
        return data

    def group_by_scene(self, data):
        return [{'scene_id': entry['scene_id'], 'frames': entry.get('frames', [])} for entry in data]

    def process_groups(self, grouped_data):
        all_processed_scenes = []
        for group in grouped_data:
            scene_traj = self.extract_scene_trajs(group['frames'], group['scene_id'])
            all_processed_scenes.append(scene_traj)
        return all_processed_scenes

    def extract_scene_trajs(self, frames, scene_id):
        scene_traj = []
        for i, frame in enumerate(frames):
            traj = self.get_trajectory(frames, i,scene_id)
            scene_traj.append({
                "scene_id": scene_id,  # 添加 scene_id
                "frame_id": frame["frame_id"],
                "trajectory": traj,
            })
        return scene_traj
# The "pose" data in frames is defined relative to the lidar coordinate system of the scene's first frame (initial frame).
# The lidar coordinate system is located at the center of the lidar sensor, with:
# - positive x-axis pointing left
# - positive y-axis pointing backward 
# - positive z-axis pointing upward
# Therefore, first transform to the ego vehicle's coordinate system, then simply invert both x and y (axis conversion)

    def get_df_trajectory(self, frames, index,scene_id):
        start_idx = max(0, index - self.max_previous_samples-1)
        end_idx = min(len(frames), index + self.max_next_samples + 1)

        ref_pose = frames[index]['pose']
        ref_translation = np.array(ref_pose[4:6]) 

        quat = ref_pose[:4]
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-6: 
            # print(f"Warning: Invalid quaternion (norm={quat_norm}) at frame {index}.scene{scene_id}. Using raw trajectory.")
            use_rotation = False
        else:
            use_rotation = True
            ref_rotation = R.from_quat(quat).as_matrix()[:2, :2]  

        trajectory = []
        for i in range(start_idx, end_idx):
            pose = frames[i]['pose']
            trans = np.array(pose[4:6])  
            if use_rotation:
                local_pos = ref_rotation.T @ (trans - ref_translation)  
                local_pos = -local_pos  
            else:
                local_pos = trans  
            trajectory.append(local_pos.tolist())

        if len(trajectory) >= 2:
            translations = np.array(trajectory)
            disp_trajectory = np.diff(translations, axis=0)  
            if start_idx == 0:
                disp_trajectory = np.vstack(([0, 0], disp_trajectory))  
        else:
            disp_trajectory = np.array([[0, 0]])  

        return disp_trajectory.tolist()

    def get_trajectory(self, frames, index,scene_id):
        start_idx = max(0, index - self.max_previous_samples)
        end_idx = min(len(frames), index + self.max_next_samples + 1)

        ref_pose = frames[index]['pose']
        ref_translation = np.array(ref_pose[4:6])

        quat = ref_pose[:4]
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-6: 
            # print(f"Warning: Invalid quaternion (norm={quat_norm}) at frame {index}.scene{scene_id}. Using raw trajectory.")
            use_rotation = False
        else:
            use_rotation = True
            ref_rotation = R.from_quat(quat).as_matrix()[:2, :2] 

        trajectory = []
        for i in range(start_idx, end_idx):
            pose = frames[i]['pose']
            trans = np.array(pose[4:6]) 
            if use_rotation:
                local_pos = ref_rotation.T @ (trans - ref_translation)  
                local_pos = -local_pos  
            else:
                local_pos = trans  
            # trajectory.append(local_pos.tolist())
            # (x, y) -> (y, -x)
            new_x = local_pos[1]  
            new_y = -local_pos[0]  
            trajectory.append([new_x, new_y])  
        trajectory=self.clean_trajectory(trajectory, self.max_next_samples, self.max_previous_samples)
        return trajectory

    def clean_trajectory(self,trajectory, max_next_samples, max_previous_samples):   
        trajectory = [list(point) for point in trajectory] 
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
       

    def compute_diff_trajectory(self, traj):
        return [[traj[i+1][0] - traj[i][0], traj[i+1][1] - traj[i][1]] for i in range(len(traj) - 1)]

    def save_output(self, processed_data):
        output_dir = os.path.dirname(self.output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for entry in processed_data:
                    for item in entry:
                        json_line = json.dumps(item, ensure_ascii=False)
                        f.write(json_line + '\n') 
            print(f"Processed data saved to {self.output_file}")
        except Exception as e:
            print(f"Error saving output file: {e}")


if __name__ == "__main__":
    base_path = f"{data_dir}/ONCE/3D_infos/untar/data/"
    processor = OnceProcessor(
        base_path, 
        output_file=f"{project_root}/data_qa_generate/data_traj_generate/data_traj_results/once/traj_once_future10.jsonl",
        max_previous_samples=0,max_next_samples=10)
    processor.process()

    processor = OnceProcessor(
        base_path, 
        output_file= f"{project_root}/data_qa_generate/data_traj_generate/data_traj_results/once/traj_once_last3.jsonl",
        max_previous_samples=3,max_next_samples=0)
    processor.process()
