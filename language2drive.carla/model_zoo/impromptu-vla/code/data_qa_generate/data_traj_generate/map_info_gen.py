import os
import json
import pandas as pd
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
root = project_root / "data_raw" / "mapillary_sls" / "train_val"
output_file =Path( project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "map"/"map_info.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
results = {}
for city in sorted(os.listdir(root)):
    cresults = {}
    for d in ["database", "query"]:
        dresults = {}
        csv_path = f"{root}/{city}/{d}/seq_info.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for seq_key, seq_group in df.groupby('sequence_key'):
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
                
                scene_res = []
                for scene in sub_scenes:
                    start_frame = scene[0]['frame_number']
                    n_frame = len(scene)
                    scene_res.append({"start_frame": start_frame, "n_frame": n_frame, "keys": [s['key'] for s in scene]})
                dresults[seq_key] = scene_res
            cresults[d] = dresults
    results[city] = cresults
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)
                