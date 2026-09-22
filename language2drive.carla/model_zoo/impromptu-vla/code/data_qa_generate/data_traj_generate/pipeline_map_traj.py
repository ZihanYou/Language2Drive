import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
# print(project_root)
data_dir = project_root / "data_raw" / "mapillary_sls" / "train_val"
output_file_last =Path( project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "map"/"traj_mapl_query_last3.jsonl")
output_file_last1 =Path( project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "map"/"traj_mapl_database_last3.jsonl")
output_file_future = Path(project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "map"/"traj_mapl_query_future10.jsonl")
output_file_future1 = Path(project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "map"/"traj_mapl_database_future10.jsonl")
output_file_last.parent.mkdir(parents=True, exist_ok=True)
output_file_future.parent.mkdir(parents=True, exist_ok=True)

class MapillarySLSProcessor:
    def __init__(self, max_previous_samples=0, max_next_samples=10, output_file=None,type_dir="query"):
        self.base_dir = Path(data_dir)
        self.output_file = output_file
        self.type_dir = type_dir
        self.max_previous_samples = max_previous_samples 
        self.max_next_samples = max_next_samples   
        self.rotation_cache = {}

    def process(self):
        all_frames = []
        city_dirs = sorted([d for d in self.base_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        
        for city_dir in city_dirs:
            
            sequences = self.process_city(city_dir)
            all_frames.extend(sequences)
         
        
        self.save_results(all_frames)
        print(f"处理完成，总帧数：{len(all_frames)}")

    def process_city(self, city_dir):
        frames = []
        print(f"正在处理城市: {city_dir.name}")
        try:
            post_df = pd.read_csv(city_dir / self.type_dir / "postprocessed.csv")
            seq_df = pd.read_csv(city_dir / self.type_dir / "seq_info.csv")
            merged_df = pd.merge(post_df, seq_df, on='key')
        except Exception as e:
            print(f"加载数据失败: {str(e)}.导致无法得到这个城市的轨迹数据")
            return frames
        for seq_key, seq_group in merged_df.groupby('sequence_key'):
            sorted_group = seq_group.sort_values('frame_number')
            sub_scenes = []
            current_scene = []
            prev_frame = None
            
            for _, row in sorted_group.iterrows():
                current_frame = row['frame_number']
                if prev_frame is None:
                    current_scene.append(row)
                else:
                    if current_frame - prev_frame != 1:
                        sub_scenes.append(current_scene)
                        current_scene = [row]
                    else:
                        current_scene.append(row)
                prev_frame = current_frame
            
            if current_scene:
                sub_scenes.append(current_scene)

            for sub_scene in sub_scenes:
                if not sub_scene:
                    continue

                start_frame = sub_scene[0]['frame_number']
                scene_id = f"{seq_key}-{start_frame}"

                sub_scene_df = pd.DataFrame(sub_scene)
                traj_raw = sub_scene_df[['easting', 'northing']].values   
                try:
                    self.compute_traj_to_thetas(traj_raw)
                    # print(self.rotation_cache)
                except Exception as e:
                    print(f"预计算旋转矩阵失败: {str(e)}")
                    continue
                sub_scene = pd.DataFrame(sub_scene)
                for idx, (_, row) in enumerate(sub_scene.iterrows()):
                    try:
                        frame_data = self.process_frame(row, sub_scene, idx, seq_key, scene_id,city_dir)
                        if frame_data:  
                            frames.append(frame_data)
                    except Exception as e:
                        print(f"处理帧失败: {str(e)}")
                        continue
        
        return frames

    def process_frame(self, row, seq_group, idx, seq_key,scene_id,city_dir):
        traj, diff_traj ,theta= self.get_trajectory(seq_group, idx, seq_key)
        traj = np.array(traj)
        traj = np.round(traj, 2).tolist()

        diff_traj = np.array(diff_traj)
        diff_traj = np.round(diff_traj, 2).tolist()
        image_path = str(city_dir / self.type_dir / "images" / (row['key'] + '.jpg'))
        
        return {
            "scene_id": scene_id,
            "frame_id": idx,
            "image_path": image_path,  
            "trajectory": traj
        }
        

    def compute_traj_to_thetas(self, traj):
        traj = np.array(traj)  # shape: (N, 2)
        N = len(traj)
        if N < 2:
            thetas = np.zeros(N)
            self.rotation_cache = thetas
            return thetas
        diffs = np.diff(traj, axis=0)  # shape: (N-1, 2)
        dists = np.linalg.norm(diffs, axis=1)
        # Calculate the angle (Note: atan2 returns radians, convert to degrees and subtract 90 degrees)
        # angles = np.degrees(np.arctan2(diffs[:,1], diffs[:,0])) - 90
        angles = np.degrees(np.arctan2(diffs[:,0], diffs[:,1])) # Parameters ordered as (x, y) follow ENU convention (0°=North, 90°=East).

        angles[dists < 0.2] = 0
        thetas = np.empty(N)
        thetas[0] = angles[0]
        thetas[1:] = angles
        self.rotation_cache = thetas
        return thetas

    def get_trajectory(self, seq_group, center_idx, seq_key):
        theta = self.rotation_cache[center_idx]
        heading_radians = np.deg2rad(theta)  

        R = np.array([
            [np.cos(heading_radians), -np.sin(heading_radians)],
            [np.sin(heading_radians), np.cos(heading_radians)]
        ])

        origin = seq_group.iloc[center_idx][['easting', 'northing']].values

        start = max(0, center_idx - self.max_previous_samples)
        end = min(len(seq_group), center_idx + self.max_next_samples + 1)

        points = seq_group.iloc[start:end][['easting', 'northing']].values  # shape: (window, 2)
        transformed = (R @ (points - origin).T).T  # shape: (window, 2)
        trajectory = transformed.tolist()

        traj_array = np.array(trajectory)
        if len(traj_array) < 2:
            diff_traj = [[0.0, 0.0]]
        else:
            diff = np.diff(traj_array, axis=0)
            if start == 0:
                diff_traj = np.vstack(([0.0, 0.0], diff)).tolist()
            else:
                diff_traj = diff.tolist()

        trajectory=self.clean_trajectory(transformed, self.max_next_samples, self.max_previous_samples)
        return trajectory, diff_traj,theta

    def clean_trajectory(self,transformed, max_next_samples, max_previous_samples):   
        transformed[:, 0], transformed[:, 1] = transformed[:, 1], -transformed[:, 0]
        trajectory = transformed.tolist()      
        if len(trajectory) > 1:
            if max_next_samples == 0:        
                if trajectory[-1] == [0.0, -0.0]:
                    trajectory = trajectory[:-1]             
            if max_previous_samples == 0:        
                if trajectory[0] == [0.0, -0.0]:
                    trajectory = trajectory[1:]
        return trajectory

    def save_results(self, data):
        with open(self.output_file, 'w') as f:
            for entry in data:
                json.dump(entry, f)
                f.write('\n')
        print(self.output_file)

if __name__ == "__main__":
    # First run: 3 history samples
    processor_history = MapillarySLSProcessor(
        max_previous_samples=3,
        max_next_samples=0,
        output_file=output_file_last,
        type_dir="query"
    )
    processor_history.process()

    processor_history = MapillarySLSProcessor(
    max_previous_samples=3,
    max_next_samples=0,
    output_file=output_file_last1,
    type_dir="database"
    )
    processor_history.process()
    
    # Second run: 10 future samples
    processor_future = MapillarySLSProcessor(
        max_previous_samples=0,
        max_next_samples=10,
        output_file=output_file_future,
        type_dir="query"
    )
    processor_future.process()

    processor_future = MapillarySLSProcessor(
    max_previous_samples=0,
    max_next_samples=10,
    output_file=output_file_future1,
    type_dir="database"
    )
    processor_future.process()