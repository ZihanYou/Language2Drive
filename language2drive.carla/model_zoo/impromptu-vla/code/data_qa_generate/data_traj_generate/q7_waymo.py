import csv
import pandas as pd
import os
import json
import pdb
from tqdm import tqdm
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
root =project_root / "data_raw" / "waymo"

user_p = "You are an autonomous driving agent. You have access to multi-view camera images of a vehicle: (1) front view (which you should focus on with the most attention) <image>, (2) front right view <image>, and (3) front left view <image>. Your task is to do your best to predict future waypoints for the vehicle over the next 10 timesteps, given the vehicle's intent inferred from the images. Provided are the previous ego vehicle statuses recorded over the last 1.5 seconds (at 0.5-second intervals). This includes the x and y coordinates of the ego vehicle. Positive x means forward direction while positive y means leftwards. "
    
assistant_p = "Predicted future movement details for the next 5 seconds (sampled at 0.5-second intervals), including BEV location in x and y directions (in meters). Positive x means forward direction while positive y means leftwards. The output is formatted as [x, y]: "
sps = [
    "training",
    "validation"
]
output_file_train =Path( project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "waymo"/"traj_waymo_train.jsonl")
output_file_val= Path(project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "waymo"/"traj_waymo_val.jsonl")
files = [
    output_file_train,output_file_val
]

def avs(traj, traj_f):
    vx0 = (traj[-2][0] - traj[-3][0]) / 0.5
    vx1 = (traj[-1][0] - traj[-2][0]) / 0.5
    vx2 = (0 - traj[-1][0]) / 0.5
    vy0 = (traj[-2][1] - traj[-3][1]) / 0.5
    vy1 = (traj[-1][1] - traj[-2][1]) / 0.5
    vy2 = (0 - traj[-1][1]) / 0.5
    vxc = traj_f[0][0] / 0.5
    vyc = traj_f[0][1] / 0.5
    vxf = (traj_f[1][0] - traj_f[0][0]) / 0.5
    vyf = (traj_f[1][1] - traj_f[0][1]) / 0.5
    ax0 = (vx1 - vx0) / 0.5
    ax1 = (vx2 - vx1) / 0.5
    ax2 = (vxc - vx2) / 0.5
    ay0 = (vy1 - vy0) / 0.5
    ay1 = (vy2 - vy1) / 0.5
    ay2 = (vyc - vy2) / 0.5
    axc = (vxf - vxc) / 0.5
    ayc = (vyf - vyc) / 0.5
    
    va_prompt = f"The data is presented in the format [x, y]:\n(t-1.5s) [{traj[-3][0]:.2f}, {traj[-3][1]:.2f}], Acceleration: X {ax0:.2f}, Y {ay0:.2f} m/s^2, Velocity: X {vx0:.2f}, Y {vy0:.2f} m/s, (t-1.0s) [{traj[-2][0]:.2f}, {traj[-2][1]:.2f}], Acceleration: X {ax1:.2f}, Y {ay1:.2f} m/s^2, Velocity: X {vx1:.2f}, Y {vy1:.2f} m/s, (t-0.5s) [{traj[-1][0]:.2f}, {traj[-1][1]:.2f}], Acceleration: X {ax2:.2f}, Y {ay2:.2f} m/s^2, Velocity: X {vx2:.2f}, Y {vy2:.2f} m/s, (t-0.0s) [0.0, 0.0], Acceleration: X {axc:.2f}, Y {ayc:.2f} m/s^2, Velocity: X {vxc:.2f}, Y {vyc:.2f} m/s\n"
    
    return va_prompt

def fps(traj_f):
    as_prompt = ", ".join([f"[{traj_f[i][0]:.2f}, {traj_f[i][1]:.2f}]" for i in range(10)])
    return as_prompt

def get_original_path(symlink_path):
    try:
        original_path = os.readlink(symlink_path)
        return original_path
    except OSError as e:
        # print(f"Error: {e}")
        return "/None-Exists"


for file, sp in zip(files, sps):
    results = []
    datas = []
    with open(file, "r") as f:
        for line in f.readlines():
            datas.append(json.loads(line))
            
    for d in tqdm(datas):
        for frame_id, rep in d["relative_poses"].items():
            if "next_10" in rep and "prev_3" in rep:
                images = [f"{root}/{sp}/{d['sequence_id']}/images/{view}/{str(int(frame_id)*5).zfill(8)}.jpg" for view in ["camera_FRONT", "camera_FRONT_RIGHT", "camera_FRONT_LEFT"]]
                prev = [rep[k] for k in ["prev_3", "prev_2", "prev_1"]]
                futr = [rep[k] for k in [f"next_{i}" for i in range(1, 11)]]
                results.append({"images": images, "messages": [{"role": "user", "content": user_p + avs(prev, futr)}, {"role": "assistant", "content": assistant_p + fps(futr)}]})
    
    print(len(results))
    if sp=="validation":
        sp="val"
    elif sp=="training":
        sp="train"
    output_path = f"{project_root}/data_qa_results/waymo/q7_{sp}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)
            
            
            