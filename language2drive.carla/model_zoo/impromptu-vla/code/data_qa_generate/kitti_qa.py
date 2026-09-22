import os
import json
from tqdm import tqdm
import numpy as np
import pdb
import math
import sys

from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys_path=f"{project_root}/data_qa_generate/data_traj_generate"
if sys_path not in sys.path:
    sys.path.append(sys_path)
from kitti_utils import load_oxts_packets_and_poses, rotz

data_root = project_root / "data_raw" / "kitti"/"data_tracking_oxts"
image_root=project_root / "data_raw" / "kitti"/"data_tracking_image_2"
qa_root = project_root / "data_qa_results" / "kitti"
os.makedirs(qa_root, exist_ok=True)

def cal_angel(x, y):
    angle = math.degrees(math.atan2(y,x))
    return angle if angle >= 0 else angle + 360

def point_to_line_distance(points):
    if len(points) < 3:
        raise ValueError("至少需要三个点")
    
    x1, y1 = points[0]
    x2, y2 = points[1]
    xn, yn = points[-1]

    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    if ((A ** 2 + B ** 2) ** 0.5) == 0 or np.isnan(((A ** 2 + B ** 2) ** 0.5)):
        distance = 0
    else:
        distance = abs(A * xn + B * yn + C) / ((A ** 2 + B ** 2) ** 0.5)
    return distance
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
     
def get_plan(raw_poses, num_fut = 4, num_fut_navi = 12,vel_navi_thresh=4.0,vel_diff_thresh=3.0, val_stop=2.0, lat_thresh=1.0, angle_thresh=5.0,angle_thresh_navi=8.0, data_fps = 10, target_fps = 2):
    interval = int(data_fps / target_fps)
    raw_xy = np.array([pose[:2] for pose in raw_poses])[::interval]
    # raw_xy = np.array([pose[:2,3] for pose in raw_poses])[::interval]
    # print(raw_xy)
    # quit()
    xy_diffs = np.diff(raw_xy, axis = 0)
    distances = np.sqrt(np.sum(xy_diffs**2, axis=1))
    speeds = distances * target_fps
    speeds = np.append(speeds, speeds[-1])
    
    speeds_diff = speeds[num_fut:] - speeds[:-num_fut]
    speed_plans = []
    for i, speed_diff in enumerate(speeds_diff):
        if speeds[i] < val_stop:
            speed_plans.append("stop")
        elif speed_diff >= vel_diff_thresh:
            speed_plans.append("accelerate")
        elif speed_diff <= -vel_diff_thresh:
            speed_plans.append("decelerate")
        else:
            speed_plans.append("const")
    speed_plans += [speed_plans[-1]] * num_fut
    

    path_plans = []
    for i in range(len(raw_xy) - num_fut):
        xys = raw_xy[i: i + num_fut]
        start_angle = cal_angel(xys[1][0] - xys[0][0], xys[1][1] - xys[0][1])
        end_angle = cal_angel(xys[-1][0] - xys[-2][0], xys[-1][1] - xys[-2][1])
        angle_diff = end_angle - start_angle
        # print(angle_diff)
        dis = point_to_line_distance(xys)
        path_plan = "straight"
        if dis<lat_thresh/2:
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
    path_plans += [path_plans[-1]] * num_fut
    
 # navigation
    navi_commands = []
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

    navi_commands += [navi_commands[-1]] * num_fut_navi

    
    assert len(speeds) == len(speed_plans) == len(path_plans)==len(navi_commands)
    return speeds, speed_plans, path_plans,navi_commands
          
            
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
        speed, speed_plan, path_plan,navigation_command = info

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

    for sp in ["training", "testing"]:
            for seq in tqdm(os.listdir(f"{root}/{sp}/oxts")):
                if seq.endswith('.txt'): 
                    file_path = f"{root}/{sp}/oxts/{seq}"
                    # print(file_path)
                    # quit()
                    scene_data = load_oxts_packets_and_poses([file_path])
                    enu_positions = [data.T_w_imu[:3, 3] for data in scene_data]
                  
                    speeds, speed_plans, path_plans,navi_commands = get_plan(enu_positions)
                    results = [(float(speed), sp, pp,nc) for speed, sp, pp,nc in zip(speeds, speed_plans, path_plans,navi_commands)]
                    seq_name = os.path.splitext(seq)[0]
                    with open(f"{root}/{sp}/oxts/{seq_name}_ego_results.json", "w") as f:
                        json.dump(results, f)
  
def gen_qa(data_root, qa_root,data_fps = 10, target_fps = 2):
    for sp in ["training", "testing"]:
      
        q3s = []
        q4s = []
        q5s = []
        q6s = []
        for seq in tqdm(sorted(os.listdir(f"{data_root}/{sp}/oxts"))):
            if seq.endswith('.txt'):  
                seq_name = os.path.splitext(seq)[0]
                with open(f'{data_root}/{sp}/oxts/{seq_name}_ego_results.json', "r") as f:
                    ego = json.load(f)
                seq=seq.replace('.txt', '')
                views = ['camera_FRONT']
                
                file_list = {}
                for view in views:
                    file_list[view] = [f"{image_root}/{sp}/image_02/{seq}/{img}" for img in sorted(os.listdir(f"{image_root}/{sp}/image_02/{seq}"))]
                
                interval = max(1, int(data_fps / target_fps))
                images = [[file_list[key][i] for key in views][::interval] 
                 for i in range(0, len(file_list['camera_FRONT']), interval)]
            
                q3s += q3(images, ego)
                q4s += q4(images)
                q5s += q5(images)
                q6s += q6(images, ego)
        print(sp, len(q3s), len(q4s), len(q5s),len(q6s))
        os.makedirs(qa_root, exist_ok=True)
        sp = "test" if sp == "testing" else "train"
        with open(f"{qa_root}/q3_{sp}.json", "w") as f:
            json.dump(q3s, f)
        with open(f"{qa_root}/q4_{sp}.json", "w") as f:
            json.dump(q4s, f)
        with open(f"{qa_root}/q5_{sp}.json", "w") as f:
            json.dump(q5s, f)
        with open(f"{qa_root}/q6_{sp}.json", "w") as f:
            json.dump(q6s, f)



gen_info(data_root)
gen_qa(data_root, qa_root)   

    