import os
import json
from tqdm import tqdm
import numpy as np
import pdb
import math
import sys
import pandas as pd
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
data_root = project_root / "data_raw" / "mapillary_sls" / "train_val"
qa_root = project_root / "data_qa_results" / "map"
os.makedirs(qa_root, exist_ok=True)

def cal_angel(x, y):
    angle = math.degrees(math.atan2(x, -y))
    return angle if angle >= 0 else angle + 360

def point_to_line_distance(points):
    if len(points) < 3:
        raise ValueError("至少需要三个点")
    

    points = np.asarray(points)
    (x1, y1), (x2, y2), (xn, yn) = points[0], points[1], points[-1]
    
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    numerator = np.abs(A * xn + B * yn + C)
    denominator = np.sqrt(A**2 + B**2)
    distance = numerator / denominator if denominator > 1e-10 else 0.0
    
    return float(distance)
def point_to_line_projection_distance(points):
    if len(points) < 3:
        raise ValueError("至少需要三个点")
    
    x1, y1 = points[0]
    x2, y2 = points[1]
    xn, yn = points[-1]
    dx = x2 - x1
    dy = y2 - y1
    vx = xn - x1
    vy = yn - y1
    line_length = (dx ** 2 + dy ** 2) ** 0.5
    if line_length == 0 or np.isnan(line_length):
        return 0

    projection_length = (vx * dx + vy * dy) / line_length
    return projection_length
def cal_angel(dx, dy):
    return np.degrees(np.arctan2(dy, dx))


def get_plan(raw_poses, num_fut = 4, num_fut_navi = 12,vel_navi_thresh=4.0,vel_diff_thresh=3.0, val_stop=2.0, lat_thresh=1.0, angle_thresh=5.0,angle_thresh_navi=8.0, data_fps = 2, target_fps = 2):
    if raw_poses is None or len(raw_poses) == 0:
        return [], [], []
    
    interval = int(data_fps / target_fps)
    interval = max(interval, 1)
    raw_xy = np.array([pose for pose in raw_poses])[::interval]
    

    if len(raw_xy) <= 1:
        speeds = np.zeros(len(raw_xy))
    else:
        xy_diffs = np.diff(raw_xy, axis=0)
        distances = np.sqrt(np.sum(xy_diffs**2, axis=1))
        speeds = distances * target_fps
        speeds = np.append(speeds, speeds[-1]) if len(speeds) > 0 else np.zeros(1)
    

    speed_plans = []
    if len(speeds) > num_fut:
        speeds_diff = speeds[num_fut:] - speeds[:-num_fut]
        for i, speed_diff in enumerate(speeds_diff):
            if speeds[i] < val_stop:
                speed_plans.append("stop")
            elif speed_diff >= vel_diff_thresh:
                speed_plans.append("accelerate")
            elif speed_diff <= -vel_diff_thresh:
                speed_plans.append("decelerate")
            else:
                speed_plans.append("const")

        pad_length = len(speeds) - len(speed_plans)
        
        speed_plans += [speed_plans[-1]] * pad_length
       
    else:
        for speed in speeds:
            if speed < val_stop:
                speed_plans.append("stop")
            else:
                speed_plans.append("const")
    

    path_plans = []
    required_points = num_fut + 1
    if len(raw_xy) >= required_points:
        for i in range(len(raw_xy) - num_fut):
            xys = raw_xy[i:i + num_fut]
            if len(xys) < 2:
                path_plan = "straight"
            else:
                start_angle = cal_angel(xys[1][0] - xys[0][0], xys[1][1] - xys[0][1])
                end_angle = cal_angel(xys[-1][0] - xys[-2][0], xys[-1][1] - xys[-2][1])
                angle_diff = end_angle - start_angle
                dis = point_to_line_distance(xys) if len(xys) >= 2 else 0.0
                
                path_plan = "straight"
                if dis<lat_thresh:
                    path_plan = "straight"
                elif angle_diff <= -angle_thresh:
                    path_plan = "right turn"
                elif angle_diff >= angle_thresh:
                    path_plan = "left turn"
                elif dis > lat_thresh:
                    if angle_diff < 0:
                        path_plan = "right lane change"
                    else:
                        path_plan = "left lane change"
                else:
                    path_plan = "straight"
            path_plans.append(path_plan)

        pad_length = len(raw_xy) - len(path_plans)
        if path_plans:
            path_plans += [path_plans[-1]] * pad_length
        else:
            path_plans = ["straight"] * len(raw_xy)
    else:
       
        path_plans = ["straight"] * len(raw_xy)
    
    # navigation
    navi_commands = []
    if len(raw_xy) >= num_fut_navi + 1: 
        for i in range(len(raw_xy) - num_fut_navi):
            xys = raw_xy[i: i + num_fut_navi]
            start_angle = cal_angel(xys[1][0] - xys[0][0], xys[1][1] - xys[0][1])
            end_angle = cal_angel(xys[-1][0] - xys[-2][0], xys[-1][1] - xys[-2][1])
            angle_diff = end_angle - start_angle
            dis = point_to_line_distance(xys)
            dis_forward=point_to_line_projection_distance(xys)

            if dis_forward >= 20.0 and dis >= 10.0 and angle_diff>0:
                    navi_command = 'go straight and turn left'
            elif dis_forward >= 20.0 and dis >= 10.0 and angle_diff<0:
                navi_command = 'go straight and turn right'
            elif dis_forward < 20.0 and dis >= 10.0 and angle_diff>0:
                navi_command = 'turn left'
            elif dis_forward < 20.0 and dis >= 10.0 and angle_diff<0:
                navi_command = 'turn right'
            else:
                navi_command = 'go straight'
            navi_commands.append(navi_command)
        
        if navi_commands:  
            pad_length = len(raw_xy) - len(navi_commands)
            navi_commands += [navi_commands[-1]] * pad_length
        else:
            navi_commands = ["go straight"] * len(raw_xy)
    else:
        navi_commands = ["go straight"] * len(raw_xy)
    assert len(speeds) == len(speed_plans) == len(path_plans)==len(navi_commands)
    return speeds.tolist() if isinstance(speeds, np.ndarray) else speeds, speed_plans, path_plans,navi_commands    
          
            
pedal_status = {
    'const': 'KEEP',
    'accelerate': 'ACCELERATE',
    'decelerate': 'DECELERATE',
    'stop': 'STOP'
}

path_status = {
    'right turn': 'RIGHT_TURN',
    'right lane change': 'RIGHT_CHANGE',
    'left turn': 'LEFT_TURN',
    'left lane change': 'LEFT_CHANGE',
    'straight': 'STRAIGHT'
}     
# (['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist'])

def q3(images, infos):
    qas = []
    pedal_decision = {
        'KEEP': 'maintain the current speed',
        'ACCELERATE': 'accelerate',
        'DECELERATE': 'decelerate',
        'STOP': 'stop the car'
    }

    path_decision = {
        'RIGHT_TURN': 'turn right',
        'RIGHT_CHANGE': 'change to the right lane',
        'LEFT_TURN': 'turn left',
        'LEFT_CHANGE': 'change to the left lane',
        'STRAIGHT': 'go straight'
    }
    for i, info in enumerate(infos):
        speed, speed_plan, path_plan,navigation_command= info

        if speed_plan == 'stop':
            decision = pedal_decision[pedal_status[speed_plan]]
        else:
            decision = pedal_decision[pedal_status[speed_plan]] + ' and ' + path_decision[path_status[path_plan]]
        image_prompt = "<FRONT VIEW>:\n<image>\n"
                 
        question = image_prompt + "You are driving, " \
                f"your current speed is {int(speed)} m/s, " \
                f"and the navigation command is {navigation_command} " \
                "your driving decision for the next three seconds is to " \
                f"{decision}. " \
                "Based on the provided image of the driving environment, " \
                "explain the most likely reason for this decision in one or two concise sentence."
        qas.append({"images": images[i], "messages": [{"role":"user", "content": question}, {"role":"assistant", "content": ""}]})
    return qas

def q4(images):
    qas = []
    question = "Given the provided forward-facing image <image> from a car's perspective, identify if there is a traffic light that affects the car's behavior. Respond with 'Red', 'Green', 'Yellow', or 'None'."
    for imgs in images:
        qas.append({"images": [imgs[0]], "messages": [{"role":"user", "content": question}, {"role":"assistant", "content": ""}]})
    return qas

def q5(images):
    qas = []
    views = ["front center"]
    for imgs in images:
        for vi, view in enumerate(views):
            question = "Suppose you are driving, and I'm providing you with the image " \
            f"captured by the car's {view} <image>, generate a description of the driving scene " \
            "which includes the key factors for driving planning, including the positions " \
            "and movements of vehicles and pedestrians; prevailing weather conditions; " \
            "time of day, distinguishing between daylight and nighttime; road conditions, " \
            "indicating smooth surfaces or the presence of obstacles; and the status of traffic lights " \
            "which influence your decision making, specifying whether they are red or green. " \
            "The description should be concise, providing an accurate understanding " \
            "of the driving environment to facilitate informed decision-making."
            qas.append({"images": [imgs[vi]], "messages": [{"role":"user", "content": question}, {"role":"assistant", "content": ""}]})
    return qas

def q6(images, infos):
    qas = []
  
    for i, info in enumerate(infos):
        speed, speed_plan, path_plan,navigation_command = info
        image_prompt = "<FRONT VIEW>:\n<image>\n"
                 
        question = image_prompt + f"Your current speed is {int(speed)} m/s, the navigation command is {navigation_command}," \
                f" based on the understanding of the driving scene and the navigation information," \
                f"what is your plan for the next three seconds?" \
                "Please answer your SPEED plan and your PATH plan. SPEED includes KEEP, ACCELERATE and DECELERATE, and STOP, " \
                "PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. " \
                "Based on the provided image of the driving environment, " \
                "For example, a correct answer format is like 'KEEP, LEFT_CHANGE'."
        answer = f"{pedal_status[speed_plan]},{path_status[path_plan]}"
        qas.append({"images": images[i], "messages": [{"role":"user", "content": question}, {"role":"assistant", "content": answer}]})
    return qas

def gen_info(root):
    city_dirs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda x: x.name)
    for type_dir  in ["database","query"]:
        for city_dir in city_dirs:
            print(f"正在处理城市: {city_dir.name}")
            type_path = city_dir / type_dir
            if not type_path.exists():
                print(f"跳过: {city_dir.name}/{type_dir} 不存在")
                continue

            try:
                post_df = pd.read_csv(city_dir / type_dir / "postprocessed.csv")
 
            except FileNotFoundError:
                continue  
            seq_df = pd.read_csv(city_dir / type_dir / "seq_info.csv")
            raw_df = pd.read_csv(city_dir / type_dir / "raw.csv")
            merged_df = pd.merge(post_df, seq_df, on='key')
            merged_df = pd.merge(merged_df, raw_df, on='key')
        
            
            for seq_key, seq_group in merged_df.groupby('sequence_key'):
                
                seq_group_sorted = seq_group.sort_values('frame_number')
                sub_groups = []
                current_sub = []
                prev_frame = None
                
                for _, row in seq_group_sorted.iterrows():
                    current_frame = row['frame_number']
                    if prev_frame is None:
                        current_sub.append(row)
                    else:
                        if current_frame - prev_frame != 1:
                            sub_groups.append(pd.DataFrame(current_sub))
                            current_sub = [row]
                        else:
                            current_sub.append(row)
                    prev_frame = current_frame
                if current_sub:
                    sub_groups.append(pd.DataFrame(current_sub))
                
                for sub_group in sub_groups:
                    
                    start_frame = sub_group.iloc[0]['frame_number']
                    scene_id = f"{seq_key}-{start_frame}"


                    traj_enu = sub_group[['easting', 'northing']].values  

                    speeds, speed_plans, path_plans,navi_commands = get_plan(traj_enu)
                    results = [(float(speed), sp, pp,nc) for speed, sp, pp,nc in zip(speeds, speed_plans, path_plans,navi_commands)]
                    with open(f"{city_dir}/{type_dir}/{scene_id}_ego_results.json", "w") as f:
                        json.dump(results, f)


def gen_qa(root, qa_root):
  
    city_dirs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda x: x.name)
    for type_dir  in ["database","query"]:
    # for type_dir  in ["database"]:
        q3s = []
        q4s = []
        q5s = []
        q6s = []
        for city_dir in city_dirs:
            images=[]
            print(f"正在处理城市: {city_dir.name}")
            type_path = city_dir / type_dir
            if not type_path.exists():
                print(f"跳过: {city_dir.name}/{type_dir} 不存在")
                continue

            try:
                post_df = pd.read_csv(city_dir / type_dir / "postprocessed.csv")

            except FileNotFoundError:
                continue  
            seq_df = pd.read_csv(city_dir / type_dir / "seq_info.csv")
            raw_df = pd.read_csv(city_dir / type_dir / "raw.csv")
            merged_df = pd.merge(post_df, seq_df, on='key')
            merged_df = pd.merge(merged_df, raw_df, on='key')
            
            for seq_key, seq_group in merged_df.groupby('sequence_key', sort=True):
                
                seq_group_sorted = seq_group.sort_values('frame_number')
                sub_groups = []
                current_sub = []
                prev_frame = None
                
                for _, row in seq_group_sorted.iterrows():
                    current_frame = row['frame_number']
                    if prev_frame is None:
                        current_sub.append(row)
                    else:
                        if current_frame - prev_frame != 1:
                            sub_groups.append(pd.DataFrame(current_sub))
                            current_sub = [row]
                        else:
                            current_sub.append(row)
                    prev_frame = current_frame
                if current_sub:
                    sub_groups.append(pd.DataFrame(current_sub))
                
                for sub_group in sub_groups:
                    
                    start_frame = sub_group.iloc[0]['frame_number']
                    scene_id = f"{seq_key}-{start_frame}"
                 
                    with open(f"{city_dir}/{type_dir}/{scene_id}_ego_results.json", "r") as f:
                        ego = json.load(f)
    
                    views = ['camera_FRONT']
                    
                    file_list = {}
                
                    for view in views:
                        dir_path = f"{city_dir}/{type_dir}/simages/{scene_id}"
                        if os.path.exists(dir_path) and os.path.isdir(dir_path):
                            file_list[view] = sorted(
                                [f"{dir_path}/{img}" for img in os.listdir(dir_path)],
                                reverse=False
                            )
                        else:
                            continue  # Skip if directory doesn't exist
                    # print(len(file_list['camera_FRONT']))
                    # print(len(ego))

                    if 'camera_FRONT' not in file_list or len(file_list['camera_FRONT']) != len(ego):
                        print(f"跳过 {scene_id}：图片数量 {len(file_list.get('camera_FRONT', []))} 不等于 ego 数据长度 {len(ego)}")
                        continue
                    images = [[file_list[key][i] for key in views] for i in range(len(ego))]

                            
            if images==[]:
                continue
            q3s += q3(images, ego)
            q4s += q4(images)
            q5s += q5(images)
            q6s += q6(images, ego)
        print( len(q3s), len(q4s), len(q5s))
        os.makedirs(qa_root, exist_ok=True)
        with open(f"{qa_root}/q3_{type_dir}.json", "w") as f:
            json.dump(q3s, f)
        with open(f"{qa_root}/q4_{type_dir}.json", "w") as f:
            json.dump(q4s, f)
        with open(f"{qa_root}/q5_{type_dir}.json", "w") as f:
            json.dump(q5s, f)
        with open(f"{qa_root}/q6_{type_dir}.json", "w") as f:
            json.dump(q6s, f)


gen_info(data_root)
gen_qa(data_root, qa_root)   

