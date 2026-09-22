import csv
import pandas as pd
import os
import json
import pdb
from tqdm import tqdm
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
root = f"{project_root}/data_raw/ONCE/3D_images/data"
output_file_last= f"{project_root}/data_qa_generate/data_traj_generate/data_traj_results/once/traj_once_last3.jsonl"
output_file_future= f"{project_root}/data_qa_generate/data_traj_generate/data_traj_results/once/traj_once_future10.jsonl"
user_p = ". Your task is to do your best to predict future waypoints for the vehicle over the next 10 timesteps, given the vehicle's intent inferred from the images. Provided are the previous ego vehicle statuses recorded over the last 1.5 seconds (at 0.5-second intervals). This includes the x and y coordinates of the ego vehicle. Positive x means forward direction while positive y means leftwards. "

assistant_p = "Predicted future movement details for the next 5 seconds (sampled at 0.5-second intervals), including BEV location in x and y directions (in meters). Positive x means forward direction while positive y means leftwards. The output is formatted as [x, y]: "

pasts = [
    output_file_last
]

futures = [
    output_file_future
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
    as_prompt = ", ".join(
        [f"[{traj_f[i][0]:.2f}, {traj_f[i][1]:.2f}]" for i in range(10)])
    return as_prompt


views = {
    "cam01": "front left view image <image>",     # cam01
    "cam02": "front right image <image>",         # cam02
    "cam03": "front center image <image>",        # cam03
    "cam04": "rear left image <image>",           # cam04
    "cam05": "side left front image <image>",     # cam05
    "cam06": "side left center image <image>",    # cam06
    "cam07": "side right front image <image>",    # cam07
    "cam08": "side right center image <image>",   # cam08
    "cam09": "rear center image <image>"          # cam09
}


for pt, ft in zip(pasts, futures):
    results = []
    pts = []
    fts = []
    with open(pt, "r") as f:
        for line in f.readlines():
            pts.append(json.loads(line))

    with open(ft, "r") as f:
        for line in f.readlines():
            fts.append(json.loads(line))

    assert len(pts) == len(fts)
    frames = {}
    for p in pts:
        if p['scene_id'] in frames:
            frames[p['scene_id']].append(p['frame_id'])
        else:
            frames[p['scene_id']] = [p['frame_id']]

    Pts = {}
    for p in pts:
        if p['scene_id'] in Pts:
            Pts[p['scene_id']].append(p['trajectory'])
        else:
            Pts[p['scene_id']] = [p['trajectory']]

    Fts = {}
    for f in fts:
        if f['scene_id'] in Fts:
            Fts[f['scene_id']].append(f['trajectory'])
        else:
            Fts[f['scene_id']] = [f['trajectory']]

    for k in tqdm(Pts.keys()):
        if not os.path.exists(f"{root}/{k}"):
            continue
        cams = sorted(os.listdir(f"{root}/{k}"))
        if "cam03" in cams:
            images = [
                [f"{root}/{k}/{c}/{i}.jpg" for c in cams if c in ["cam03"]] for i in frames[k]]
            cam_p = "You are an autonomous driving agent. You have vehicle's " + \
                ", ".join([views[c]
                          for c in cams if c in ["cam03"]])
            for i in range(3, len(Pts[k]) - 10):
                assert len(Pts[k][i]) >= 3
                assert len(Fts[k][i]) >= 10
                results.append({"images": images[i], "messages": [{"role": "user", "content": cam_p + user_p + avs(
                    Pts[k][i], Fts[k][i])}, {"role": "assistant", "content": assistant_p + fps(Fts[k][i])}]})

    print(len(results))
    output_path = f"{project_root}/data_qa_results/once/q7.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)
