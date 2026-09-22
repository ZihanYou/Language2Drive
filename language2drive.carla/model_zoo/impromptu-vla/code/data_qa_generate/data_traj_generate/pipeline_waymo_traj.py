import os
import sys
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
# print(project_root)
data_dir_train = project_root / "data_raw" / "waymo"/"training"
data_dir_val = project_root / "data_raw" / "waymo"/"validation"
processed_dir = project_root / "data_raw" / "waymo_processed"
processed_dir_train = processed_dir / "training"
processed_dir_val = processed_dir / "validation"
output_file_train =Path( project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "waymo"/"traj_waymo_train.jsonl")
output_file_val= Path(project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "waymo"/"traj_waymo_val.jsonl")
output_file_train.parent.mkdir(parents=True, exist_ok=True)
output_file_val.parent.mkdir(parents=True, exist_ok=True)
raw_hz=10
step_by_hz=raw_hz//2

class PipelineProcessor:
    def __init__(self, input_paths, output_paths, processed_dir, max_prev=5, max_next=6, workers=4):

        if len(output_paths) != 1:
            raise ValueError("Output paths must contain exactly one file")
            
        self.input_paths = input_paths
        self.output_path = output_paths[0]
        self.max_prev = max_prev
        self.max_next = max_next
        self.workers = workers
        self.processed_dir = processed_dir
        
    def _parse_frame(self, data):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        return frame

    def _format_coordinates(self, coord_list):
        return [round(coord, 2) for coord in coord_list]

    def _process_single_file(self, input_path):
        dataset = tf.data.TFRecordDataset(input_path, compression_type='')
        all_frames = [self._parse_frame(data) for data in dataset]
        
        step_by_hz = 5
        indices = [i for i in range(0, len(all_frames), step_by_hz)]
        subsscene_data = [all_frames[i] for i in indices]
        
        seq = input_path.split("/")[-1].split(".")[0]
        with open(f"{self.processed_dir}/{seq}/dynamic_objects_pose.json", "r") as f:
            dynamic_info = json.load(f)

        raw_poses = [np.array(frame.pose.transform).reshape(4, 4).astype(np.float32) for frame in subsscene_data]
        base_pose = raw_poses[0][:, 3]
        np.save(f"{self.processed_dir}/{seq}/raw_poses.npy",raw_poses)
        np.save(f"{self.processed_dir}/{seq}/base_pose.npy",base_pose)
        re_poses = [np.linalg.inv(raw_pose) for raw_pose in raw_poses]
        np.save(f"{self.processed_dir}/{seq}/re_poses.npy",re_poses)
        sequence_data = {
            "sequence_id": seq,
            "num_frames": len(raw_poses),
            "relative_poses": {}
        }
        
        for dcls, cinfo in dynamic_info.items():
            for dobj in cinfo:
                strat_idx = dobj["segments"][0]["start_frame"]
                num_frame = dobj["segments"][0]["n_frames"]
                rposes = []
                for trans in dobj["segments"][0]["data"]["transform"]:
                    npose = np.array(trans, dtype=np.float32)
                    wpose = npose
                    wpose[:3] += base_pose[:3]
                    
                    for frame_idx in range(strat_idx, strat_idx+num_frame):
                        rpose = re_poses[frame_idx] @ wpose
                        rposes.append(rpose.tolist()[:2])
                dobj["segments"][0]["data"]["transform"] = rposes
                        
        with open(f"{self.processed_dir}/{seq}/dynamic_objects_pose_processed.json", "w") as f:
            json.dump(dynamic_info, f)
        

        for j in range(len(raw_poses)):
            sequence_data["relative_poses"][str(j)] = {}
            for i in range(max(0, j-3), j):
                rel_pose = re_poses[j] @ raw_poses[i]
                sequence_data["relative_poses"][str(j)][f"prev_{j-i}"] = self._format_coordinates(rel_pose[:2, 3].tolist())

            for i in range(j+1, min(len(raw_poses), j+11)):
                rel_pose = re_poses[j] @ raw_poses[i]
                sequence_data["relative_poses"][str(j)][f"next_{i-j}"] = self._format_coordinates(rel_pose[:2, 3].tolist())
        return sequence_data


    def execute(self):
        with open(self.output_path, 'w') as f:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = [executor.submit(self._process_single_file, path) for path in self.input_paths]
                for future in tqdm(futures, desc="Processing files"):
                    json.dump(future.result(), f)
                    f.write('\n') 

    def _convert_to_json_serializable(self, data):
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in data]
        elif isinstance(data, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in data.items()}
        return data




if __name__ == "__main__":
    configs = [
        {
            'data_split': 'validation',
            'input_dir': data_dir_val,
            'output_path': output_file_val,
            'processed_dir': processed_dir_val
        },

        {
            'data_split': 'training',
            'input_dir': data_dir_train,
            'output_path': output_file_train,
            'processed_dir': processed_dir_train
        }
    ]

    for config in configs:
        print(f"Processing {config['data_split']} data from {config['input_dir']}")
        input_paths = sorted(glob(os.path.join(config['input_dir'], "*.tfrecord")))
        processor = PipelineProcessor(
            input_paths=input_paths,
            output_paths=[config['output_path']],
            processed_dir=config['processed_dir'],
            max_prev=0,
            max_next=9,
            workers=32
        )
        processor.execute()
        print(f"Finished processing {config['data_split']} data")