import os
import json
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from pathlib import Path
import sys
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.parent.parent
sys.path.append(f"{project_root}/data_qa_generate/")
from data_engine.datasets.navsim.dataset_navsim import VLMNavsim
qa_root = project_root / "data_qa_results" / "navsim"
os.makedirs(qa_root, exist_ok=True)
from data_engine.datasets.navsim.dataset_navsim import VLMNavsim
import pdb

class Config:
    # Trajectory planning parameters
    NUM_FUT = 4
    NUM_FUT_NAVI = 12
    VEL_NAVI_THRESH = 4.0
    VEL_DIFF_THRESH = 3.0
    VAL_STOP = 2.0
    LAT_THRESH = 2
    ANGLE_THRESH = 30.0
    ANGLE_THRESH_NAVI = 8.0  # abused
    DATA_FPS = 2
    TARGET_FPS = 2

    # Navigation command thresholds
    NAVI_DIS_FORWARD_THRESH = 20.0
    NAVI_DIS_THRESH = 10.0

    # Path and output configurations
    DATA_ROOT = qa_root
    QA_ROOT = qa_root
    BASE_PATH = f"{project_root}/data_qa_generate/"


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
# Common prompts
image_prompt = """<FRONT VIEW>:\n<image>\n
<FRONT LEFT VIEW>:\n<image>\n
<FRONT RIGHT VIEW>:\n<image>\n
<LEFT VIEW>:\n<image>\n
<RIGHT VIEW>:\n<image>\n
<BACK LEFT VIEW>:\n<image>\n
<BACK RIGHT VIEW>:\n<image>\n
<BACK VIEW>:\n<image>\n"""


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
        speed, speed_plan, path_plan, navigation_command = info

        if speed_plan == 'stop':
            decision = pedal_decision[pedal_status[speed_plan]]
        else:
            decision = pedal_decision[pedal_status[speed_plan]] + \
                ' and ' + path_decision[path_status[path_plan]]

        question = image_prompt + "You are driving, " \
            f"your current speed is {int(speed)} m/s, " \
            f"and the navigation command is {navigation_command} " \
            "your driving decision for the next three seconds is to " \
            f"{decision}. " \
            "Based on the provided image of the driving environment, " \
            "explain the most likely reason for this decision in one or two concise sentence."
        qas.append({"images": images[i], "messages": [
                   {"role": "user", "content": question}, {"role": "assistant", "content": ""}]})
    return qas


def q4(images, **kwargs):
    qas = []
    question = "Given the provided forward-facing image <image> from a car's perspective, identify if there is a traffic light that affects the car's behavior. Respond with 'Red', 'Green', 'Yellow', or 'None'."
    for imgs in images:
        qas.append({"images": [imgs[0]], "messages": [
                   {"role": "user", "content": question}, {"role": "assistant", "content": ""}]})
    return qas


def q5(images, **kwargs):
    qas = []
    views = ["ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left",
             "ring_side_right", "ring_rear_left", "ring_rear_right", "ring_back_center"]
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
            qas.append({"images": [imgs[vi]], "messages": [
                       {"role": "user", "content": question}, {"role": "assistant", "content": ""}]})
    return qas


def q6(images, infos):
    qas = []

    for i, info in enumerate(infos):
        speed, speed_plan, path_plan, navigation_command = info

        question = image_prompt + f"Your current speed is {int(speed)} m/s, the navigation command is {navigation_command}," \
            f" based on the understanding of the driving scene and the navigation information," \
            f"what is your plan for the next three seconds?" \
            "Please answer your SPEED plan and your PATH plan. SPEED includes KEEP, ACCELERATE and DECELERATE, and STOP, " \
            "PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. " \
            "Based on the provided image of the driving environment, " \
            "For example, a correct answer format is like 'KEEP, LEFT_CHANGE'."
        answer = f"{pedal_status[speed_plan]},{path_status[path_plan]}"
        qas.append({"images": images[i], "messages": [
                   {"role": "user", "content": question}, {"role": "assistant", "content": answer}]})
    return qas


def calculate_angle(x, y):
    """Calculate angle in degrees from x and y coordinates"""
    angle = math.degrees(math.atan2(x, -y))
    return angle if angle >= 0 else angle + 360


def point_to_line_distance(points):
    """Calculate minimum distance from last point to the line formed by first two points"""
    if len(points) < 3:
        raise ValueError("At least three points required")

    points = np.asarray(points)
    (x1, y1), (x2, y2), (xn, yn) = points[0], points[1], points[-1]

    if np.allclose([x1, y1], [x2, y2]):
        # Fallback to point-to-point distance
        return np.sqrt((xn - x1)**2 + (yn - y1)**2)

    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    numerator = np.abs(A * xn + B * yn + C)
    denominator = np.sqrt(A**2 + B**2)
    return float(numerator / denominator) if denominator > 1e-10 else 0.0


def point_to_line_projection_distance(points):
    """Calculate projection distance of last point onto the line formed by first two points"""
    if len(points) < 3:
        raise ValueError("At least three points required")

    x1, y1 = points[0]
    x2, y2 = points[1]
    xn, yn = points[-1]
    dx, dy = x2 - x1, y2 - y1
    vx, vy = xn - x1, yn - y1

    line_length = (dx ** 2 + dy ** 2) ** 0.5
    if line_length == 0 or np.isnan(line_length):
        return 0

    return (vx * dx + vy * dy) / line_length


def get_plan(raw_poses, num_fut=Config.NUM_FUT, num_fut_navi=Config.NUM_FUT_NAVI,
             vel_navi_thresh=Config.VEL_NAVI_THRESH, vel_diff_thresh=Config.VEL_DIFF_THRESH,
             val_stop=Config.VAL_STOP, lat_thresh=Config.LAT_THRESH,
             angle_thresh=Config.ANGLE_THRESH, angle_thresh_navi=Config.ANGLE_THRESH_NAVI,
             data_fps=Config.DATA_FPS, target_fps=Config.TARGET_FPS):
    """Generate speed and path plans based on trajectory points"""
    if not raw_poses or len(raw_poses) == 0:
        return [], [], [], []

    interval = max(int(data_fps / target_fps), 1)
    raw_xy = np.array(raw_poses)[::interval]

    # Speed calculations
    speeds = np.zeros(len(raw_xy)) if len(raw_xy) <= 1 else np.append(
        np.sqrt(np.sum(np.diff(raw_xy, axis=0)**2, axis=1)) * target_fps,
        [0]
    )

    # Speed planning
    speed_plans = []
    if len(speeds) > num_fut:
        speed_diffs = speeds[num_fut:] - speeds[:-num_fut]
        speed_plans = [
            'stop' if speeds[i] < val_stop else
            'accelerate' if diff >= vel_diff_thresh else
            'decelerate' if diff <= -vel_diff_thresh else 'const'
            for i, diff in enumerate(speed_diffs)
        ]
        speed_plans += [speed_plans[-1]] * (len(speeds) - len(speed_plans))
    else:
        speed_plans = ['stop' if s < val_stop else 'const' for s in speeds]

    # Path planning
    path_plans = []
    if len(raw_xy) >= Config.NUM_FUT + 1:
        for i in range(len(raw_xy) - Config.NUM_FUT):
            xys = raw_xy[i:i+Config.NUM_FUT]
            start_angle = calculate_angle(
                xys[1][0]-xys[0][0], xys[1][1]-xys[0][1])
            end_angle = calculate_angle(
                xys[-1][0]-xys[-2][0], xys[-1][1]-xys[-2][1])
            angle_diff = end_angle - start_angle
            dis = point_to_line_distance(xys) if len(xys) >= 2 else 0.0

            if dis < Config.LAT_THRESH:
                path_plan = "straight"
            else:
                path_plan = (
                    "right turn" if angle_diff <= -Config.ANGLE_THRESH else
                    "left turn" if angle_diff >= Config.ANGLE_THRESH else
                    "right lane change" if angle_diff < 0 else "left lane change"
                )
            path_plans.append(path_plan)
        path_plans += [path_plans[-1]] * (len(raw_xy) - len(path_plans))
    else:
        path_plans = ["straight"] * len(raw_xy)

    # Navigation commands
    navi_commands = []
    if len(raw_xy) >= Config.NUM_FUT_NAVI + 1:
        for i in range(len(raw_xy) - Config.NUM_FUT_NAVI):
            xys = raw_xy[i:i+Config.NUM_FUT_NAVI]
            start_angle = calculate_angle(
                xys[1][0]-xys[0][0], xys[1][1]-xys[0][1])
            end_angle = calculate_angle(
                xys[-1][0]-xys[-2][0], xys[-1][1]-xys[-2][1])
            angle_diff = end_angle - start_angle
            dis = point_to_line_distance(xys)
            dis_forward = point_to_line_projection_distance(xys)

            if dis < Config.NAVI_DIS_THRESH:
                navi_command = 'go straight'
            elif dis_forward >= Config.NAVI_DIS_FORWARD_THRESH and dis >= Config.NAVI_DIS_THRESH:
                navi_command = f"go straight and turn {'left' if angle_diff > 0 else 'right'}"
            else:
                navi_command = f"turn {'left' if angle_diff > 0 else 'right'}"
            navi_commands.append(navi_command)
        navi_commands += [navi_commands[-1]] * \
            (len(raw_xy) - len(navi_commands))
    else:
        navi_commands = ["go straight"] * len(raw_xy)

    return speeds.tolist(), speed_plans, path_plans, navi_commands


def plot_positions(positions, title, save_path):
    """Helper function to plot positions with start and end points marked"""
    x, y = zip(*positions) if positions else ([], [])

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=range(len(positions)), cmap='viridis', s=10)
    plt.plot(x, y, alpha=0.3)  # Connect points with a line

    # Mark start and end points
    if len(positions) > 0:
        plt.scatter([x[0]], [y[0]], c='green', s=100, label='Start')
        plt.scatter([x[-1]], [y[-1]], c='red', s=100, label='End')

    plt.title(title)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.colorbar(label='Frame index')

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def gen_info(root):
    os.makedirs(root, exist_ok=True)
    for mode in ["test", "train"]:
        dataset = VLMNavsim(mode=mode)
        enu_positions = []
        for sid in tqdm(range(len(dataset)), desc=f"Processing {mode} samples"):
            container = dataset.get_container_in(sid)
            # tolist [x, y]
            xy1 = container["frame_data"][3]["ego_status"].ego_pose[:2].tolist()
            enu_positions.append(xy1)  # add like  [[x1, y1], [x2, y2], ...]
        # title = f"{mode} Trajectory"
        # save_path = os.path.join(OUTPUT_DIR, f"{mode}_trajectory.png")
        # plot_positions(enu_positions, title, save_path)

        speeds, speed_plans, path_plans, navi_commands = get_plan(
            enu_positions)
        results = [(float(speed), sp, pp, nc) for speed, sp, pp,
                   nc in zip(speeds, speed_plans, path_plans, navi_commands)]

        with open(f"{root}/{mode}_ego_results.json", "w") as f:
            json.dump(results, f)


def gen_qa(data_root, qa_root,data_fps =2, target_fps = 2):
    for mode in ["test","train" ]:

        q3s = []
        q4s = []
        q5s = []
        q6s = []

        with open(f'{data_root}/{mode}_ego_results.json', "r") as f:
            ego = json.load(f)

        views = ['cam_f0', 'cam_10', 'cam_r0', 'cam_11', 'cam_r1', 'cam_12', 'cam_r2', 'cam_b0']
        dataset = VLMNavsim(mode=mode)
        images=[]
        for sid in tqdm(range(len(dataset)), desc=f"Processing {mode} samples"):

            images.append([
                os.path.join(Config.BASE_PATH, cam["image_path"])
                for cam in dataset.get_container_in(sid)["frame_data"][3]["cameras"].values()
                if "image_path" in cam
            ])

        q3s += q3(images, ego)
        q4s += q4(images)
        q5s += q5(images)
        q6s += q6(images, ego)
        print(mode, len(q3s), len(q4s), len(q5s),len(q6s))
        os.makedirs(qa_root, exist_ok=True)
        with open(f"{qa_root}/{mode}_q3.json", "w") as f:
            json.dump(q3s, f)
        with open(f"{qa_root}/{mode}_q4.json", "w") as f:
            json.dump(q4s, f)
        with open(f"{qa_root}/{mode}_q5.json", "w") as f:
            json.dump(q5s, f)
        with open(f"{qa_root}/{mode}_q6.json", "w") as f:
            json.dump(q6s, f)

gen_info()
gen_qa()