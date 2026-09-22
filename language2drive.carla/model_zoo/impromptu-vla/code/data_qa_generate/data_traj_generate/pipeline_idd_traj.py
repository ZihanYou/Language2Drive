import os
import json
import pandas as pd
import numpy as np
from pyproj import Transformer
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
# print(project_root)
data_dir = project_root / "data_raw" / "idd_multimodal" / "primary"
output_file_last =Path( project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "idd"/"traj_idd_last3.jsonl")
output_file_future = Path(project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "idd"/"traj_idd_future10.jsonl")
output_file_last.parent.mkdir(parents=True, exist_ok=True)
output_file_future.parent.mkdir(parents=True, exist_ok=True)

def convert_latlon_to_utm(lons, lats, epsg_target="EPSG:32644"):
    transformer = Transformer.from_crs("EPSG:4326", epsg_target, always_xy=True)
    eastings, northings = transformer.transform(lons, lats)
    return eastings, northings

class IDDTrajectoryProcessor:
    def __init__(self, max_previous_samples=0, max_next_samples=10, output_file=None):
        self.data_dir = data_dir
        self.sub_dirs = ['d0', 'd1', 'd2']
        self.dataset_types = ['train', 'val']
        self.output_file = output_file
        self.max_previous_samples = max_previous_samples
        self.max_next_samples = max_next_samples
        self.results = []
        self.rotation_cache = None

    def process(self):
        for sub_dir in self.sub_dirs:
            for dataset_type in self.dataset_types:
                file_path = self.data_dir / sub_dir / f"{dataset_type}.csv"
                if not file_path.exists():
                    print(f"文件 {file_path} 不存在，跳过。")
                    continue
                print(f"正在处理文件：{file_path}")
                df = pd.read_csv(file_path)

                lons = df['longitude'].values
                lats = df['latitude'].values
                eastings, northings = convert_latlon_to_utm(lons, lats)
                df['easting'] = eastings
                df['northing'] = northings

                easting_ref, northing_ref = eastings[0], northings[0]
                df['x_rel'] = df['easting'] - easting_ref
                df['y_rel'] = df['northing'] - northing_ref

                indices = []
                current = 0
                add_eight = True
                while current < len(df):
                    indices.append(current)
                    step = 7 if add_eight else 8
                    next_current = current + step
                    if next_current >= len(df):
                        break
                    current = next_current
                    add_eight = not add_eight

            
                df_subset = df.loc[indices].reset_index(drop=True)  
                traj = df_subset[['easting', 'northing']].values
                self.compute_traj_to_thetas(traj)  

                for i in range(len(df_subset)):
                    original_idx = indices[i]  
                    try:
                        traj, diff_traj = self.get_trajectory(df_subset, i)

                        frame_data = {
                            "scene_id": f"{sub_dir}_{dataset_type}",
                            "frame_id": original_idx, 
                            "image_idx": str(df.loc[original_idx, "image_idx"]),
                            "timestamp": str(df.loc[original_idx, "timestamp"]),
                            "trajectory": traj
                        }
                        self.results.append(frame_data)
                    except Exception as e:
                        print(f"处理 {sub_dir}_{dataset_type} 中帧 {original_idx} 时出错：{str(e)}")
                        continue

        self.save_results(self.results)
        print(f"处理完成，总帧数：{len(self.results)}")

    def compute_traj_to_thetas(self, traj):
    
        traj = np.array(traj)
        N = len(traj)
        if N < 2:
            thetas = np.zeros(N)
            self.rotation_cache = thetas
            return thetas
        diffs = np.diff(traj, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        angles = np.degrees(np.arctan2(diffs[:, 0], diffs[:, 1]))  # 参数顺序改为 (x, y),这样的话，是ENU，北0度，东90度
        angles[dists < 0.2] = 0
        thetas = np.empty(N)
        thetas[0] = angles[0]
        thetas[1:] = angles
        self.rotation_cache = thetas
        return thetas

    def get_trajectory(self, df, center_idx):
        theta = self.rotation_cache[center_idx]
        heading_radians = np.deg2rad(theta)
        # Construct the rotation matrix to transform global UTM coordinates to the vehicle coordinate system (vehicle's forward direction is the positive y-axis)  
        R = np.array([
            [np.cos(heading_radians), -np.sin(heading_radians)],
            [np.sin(heading_radians), np.cos(heading_radians)]
        ])
        origin = df[['easting', 'northing']].iloc[center_idx].values

        start = max(0, center_idx - self.max_previous_samples)
        end = min(len(df), center_idx + self.max_next_samples+1)
        points = df[['easting', 'northing']].iloc[start:end].values
        transformed = (R @ (points - origin).T).T
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
        return trajectory, diff_traj

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
        trajectory = np.array(trajectory)
        trajectory = np.round(trajectory, 2).tolist()
        return trajectory

    def save_results(self, data):
        with open(self.output_file, 'w') as f:
            for entry in data:
                json.dump(entry, f)
                f.write('\n')
            print("output:",self.output_file)


if __name__ == "__main__":
    # First run: 3 history samples
    processor_history = IDDTrajectoryProcessor(
        max_previous_samples=3,
        max_next_samples=0,
        output_file=output_file_last
    )
    processor_history.process()
    
    # Second run: 10 future samples
    processor_future = IDDTrajectoryProcessor(
        max_previous_samples=0,
        max_next_samples=10,
        output_file=output_file_future
    )
    processor_future.process()