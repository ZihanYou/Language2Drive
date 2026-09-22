import os
import json
from tqdm import tqdm
import numpy as np
import pdb
import math
import sys
import matplotlib.pyplot as plt
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
data_root = f"{project_root}/data_raw/ONCE/3D_infos/untar/data/"
pic_root = f"{project_root}/data_raw/ONCE/3D_images/data"
qa_root = project_root / "data_qa_results" / "once"
os.makedirs(qa_root, exist_ok=True)

scene_sum_num = 581
def cal_angel(x, y):
    angle = math.degrees(math.atan2(x, -y))
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


def get_plan(raw_poses, num_fut=4, num_fut_navi=12, vel_navi_thresh=4.0, vel_diff_thresh=3.0, val_stop=2.0, lat_thresh=1.0, angle_thresh=10.0, angle_thresh_navi=8.0, data_fps=2, target_fps=2):
    interval = int(data_fps / target_fps)

    raw_xy = np.array(raw_poses)[::interval]

    xy_diffs = np.diff(raw_xy, axis=0)
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
        dis = point_to_line_distance(xys)
        path_plan = "straight"

        if dis < lat_thresh:
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
        dis_forward = point_to_line_projection_distance(xys)
        navi_command = "go straight"
        if dis_forward >= 20.0 and dis >= 10.0 and angle_diff > 0:
            navi_command = 'go straight and turn left'
        elif dis_forward >= 20.0 and dis >= 10.0 and angle_diff < 0:
            navi_command = 'go straight and turn right'
        elif dis_forward < 20.0 and dis >= 10.0 and angle_diff > 0:
            navi_command = 'turn left'
        elif dis_forward < 20.0 and dis >= 10.0 and angle_diff < 0:
            navi_command = 'turn right'
        else:
            navi_command = 'go straight'

        navi_commands.append(navi_command)
    navi_commands += [navi_commands[-1]] * num_fut_navi

    assert len(speeds) == len(speed_plans) == len(
        path_plans) == len(navi_commands)
    return speeds, speed_plans, path_plans, navi_commands


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
        speed, speed_plan, path_plan, navigation_command = info

        if speed_plan == 'stop':
            decision = pedal_decision[pedal_status[speed_plan]]
        else:
            decision = pedal_decision[pedal_status[speed_plan]] + \
                ' and ' + path_decision[path_status[path_plan]]


        current_step = images[i]
        view_names = list(current_step.keys())
        image_paths = []
        image_prompt = ""

        for view in view_names:
            formatted_view = view.replace(
                "ring_", "").replace("_", " ").title()
            image_prompt += f"<{formatted_view}>:\n<image>\n"
            image_paths.append(current_step[view])


        question = image_prompt + "You are driving, " \
            f"your current speed is {int(speed)} m/s, " \
            f"and the navigation command is {navigation_command} " \
            "your driving decision for the next three seconds is to " \
            f"{decision}. " \
            "Based on the provided image of the driving environment, " \
            "explain the most likely reason for this decision in one or two concise sentence."

        qas.append({
            "images": image_paths,  # 传递所有视角的路径
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": ""}
            ]
        })
    return qas


def q4(images):
    qas = []
    question = "Given the provided forward-facing image <image> from a car's perspective, identify if there is a traffic light that affects the car's behavior. Respond with 'Red', 'Green', 'Yellow', or 'None'."
    for step in images:
        front_center_path = step.get("ring_front_center", "")
        if not front_center_path:
            continue 
        qas.append({
            "images": [front_center_path],
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": ""}
            ]
        })
    return qas


def q5(images):
    qas = []
    required_views = [
        "ring_front_left",          # cam01
        "ring_front_center",        # cam03
        "ring_side_left_front",     # cam05
        "ring_side_left_center",    # cam06
        "ring_side_right_front",    # cam07
        "ring_side_right_center",   # cam08
        "ring_rear_center"          # cam09
    ]

    for step in images:
        for view_name in required_views:

            image_path = step.get(view_name, "")
            if not image_path:
                continue  

            view_desc = view_name.replace(
                "ring_", "").replace("_", " ").title()
            question = "Suppose you are driving, and I'm providing you with the image " \
                f"captured by the car's {view_desc} <image>, generate a description of the driving scene " \
                "which includes the key factors for driving planning, including the positions " \
                "and movements of vehicles and pedestrians; prevailing weather conditions; " \
                "time of day, distinguishing between daylight and nighttime; road conditions, " \
                "indicating smooth surfaces or the presence of obstacles; and the status of traffic lights " \
                "which influence your decision making, specifying whether they are red or green. " \
                "The description should be concise, providing an accurate understanding of the driving environment to facilitate informed decision-making."

            qas.append({
                "images": [image_path],
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": ""}
                ]
            })
    return qas


def q6(images, infos):
    qas = []

    for i, info in enumerate(infos):
        speed, speed_plan, path_plan, navigation_command = info

        current_step = images[i]
        view_names = list(current_step.keys())
        image_paths = []
        image_prompt = ""

        for view in view_names:
            # （ "ring_front_center" → "Front Center"）
            formatted_view = view.replace(
                "ring_", "").replace("_", " ").title()
            image_prompt += f"<{formatted_view}>:\n<image>\n"
            image_paths.append(current_step[view])

 
        question = image_prompt + f"Your current speed is {int(speed)} m/s, the navigation command is {navigation_command}," \
            f" based on the understanding of the driving scene and the navigation information," \
            f"what is your plan for the next three seconds?" \
            "Please answer your SPEED plan and your PATH plan. SPEED includes KEEP, ACCELERATE and DECELERATE, and STOP, " \
            "PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. " \
            "Based on the provided image of the driving environment, " \
            "For example, a correct answer format is like 'KEEP, LEFT_CHANGE'."
        # answer = f"My SPEED plan is {speed_plan},and my PATH plan is {path_plan}."
        answer = f"{pedal_status[speed_plan]},{path_status[path_plan]}"
        qas.append({
            "images": image_paths,  
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })
    return qas


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

    for seq in tqdm(sorted(os.listdir(root))):
        file_path = os.path.join(root, seq, f"{seq}.json")
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            scene_data = file_data.get('frames', [])

            # Original coordinate system: x right, y back
            ref_positions = list(
                map(lambda x: (x['pose'][4], x['pose'][5]), scene_data))

            speeds, speed_plans, path_plans, navigation_commands = get_plan(
                ref_positions)

            results = [(float(speed), sp, pp, nc) for speed, sp, pp, nc in zip(
                speeds, speed_plans, path_plans, navigation_commands)]
            with open(f"{root}/{seq}/{seq}_ego_results.json", "w") as f:
                json.dump(results, f)


def gen_qa(root, qa_root):

    q3s = []
    q4s = []
    q5s = []
    q6s = []

    views = [
        "ring_front_left",          # cam01
        "ring_front_right",         # cam02（not exist actually）
        "ring_front_center",        # cam03
        "ring_rear_left",           # cam04（not exist actually）
        "ring_side_left_front",     # cam05
        "ring_side_left_center",    # cam06
        "ring_side_right_front",    # cam07
        "ring_side_right_center",   # cam08
        "ring_rear_center"          # cam09
    ]

    for seq in tqdm(sorted(os.listdir(root))):
        with open(f"{root}/{seq}/{seq}_ego_results.json", "r") as f:
            ego = json.load(f)

        file_list = {}
        seq_data_dir = os.path.join(pic_root, seq)
        if not os.path.exists(seq_data_dir):
            print(f"路径 {seq_data_dir} 不存在，跳过当前循环。")
            continue
        # print(seq_data_dir)

        for cam_dir in os.listdir(seq_data_dir):
            # print(cam_dir)
            if not cam_dir.startswith("cam"):
                continue

            cam_number = cam_dir[3:]  
            # print(cam_number)
            try:
                view_index = int(cam_number)  # cam01 → 1 → views[0]
            except ValueError:
                continue

            if view_index < 1 or view_index > len(views):
                continue

            view_name = views[view_index - 1]
            if not view_name.strip():
                continue

            cam_path = os.path.join(seq_data_dir, cam_dir)
            for img in sorted(os.listdir(cam_path)):
                img_path = os.path.join(cam_path, img)
                if view_name not in file_list:
                    file_list[view_name] = []
                file_list[view_name].append(img_path)

        # pdb.set_trace()
        if 'ring_front_center' in file_list:
            valid_views = ['ring_front_center']
        else:
            continue

        min_image_count = min(len(file_list[view]) for view in valid_views)
        max_ego_length = min(len(ego), min_image_count)
        # print(len(ego))
        # print(min_image_count)

        images = []
        for i in range(max_ego_length):
            current_step = {}
            for view in valid_views:
                current_step[view] = file_list[view][i] 
            images.append(current_step)
        # print(images)
        q3s += q3(images, ego[:max_ego_length])
        q4s += q4(images)
        q5s += q5(images)
        q6s += q6(images, ego[:max_ego_length])
    print(len(q3s), len(q4s), len(q5s), len(q6s))
    os.makedirs(qa_root, exist_ok=True)
    with open(f"{qa_root}/q3.json", "w") as f:
        json.dump(q3s, f)
    with open(f"{qa_root}/q4.json", "w") as f:
        json.dump(q4s, f)
    with open(f"{qa_root}/q5.json", "w") as f:
        json.dump(q5s, f)
    with open(f"{qa_root}/q6.json", "w") as f:
        json.dump(q6s, f)


gen_info(data_root)
gen_qa(data_root, qa_root)

