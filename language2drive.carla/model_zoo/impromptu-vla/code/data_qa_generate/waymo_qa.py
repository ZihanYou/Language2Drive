import os
import json
from tqdm import tqdm
import numpy as np
import pdb
import math
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
data_root = project_root / "data_raw" / "waymo"
qa_root = project_root / "data_qa_results" / "waymo"
os.makedirs(qa_root, exist_ok=True)


def get_obj_rel_position(loc):
    x, y = loc[0], loc[1]
    angle = math.degrees(math.atan2(x, -y))
    angle1 = angle if angle >= 0 else angle + 360

    rel_p = ""
    if 75 <= angle1 < 105:
        rel_p = "front center"
    elif 15 <= angle1 < 75:
        rel_p = "front right"
    elif 270 <= angle1 < 325:
        rel_p = "rear right"
    elif (325 <= angle1 < 360) or (0 <= angle1 < 15):
        rel_p = "side right"
    elif 105 <= angle1 < 165:
        rel_p = "front left"
    elif 215 <= angle1 < 270:
        rel_p = "rear left"
    elif 165 <= angle1 < 215:
        rel_p = "side left"
    return rel_p, [x, y]
    
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
     
def get_plan(raw_poses, num_fut = 4, num_fut_navi = 12,vel_navi_thresh=4.0,vel_diff_thresh=3.0, val_stop=2.0, lat_thresh=1.0, angle_thresh=5.0,angle_thresh_navi=8.0, data_fps = 10, target_fps = 2):
    interval = int(data_fps / target_fps)
    raw_xy = np.array([pose[:2, 3] for pose in raw_poses])[::interval]
    
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
        dis = point_to_line_distance(xys)
      

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
    path_plans += [path_plans[-1]] * num_fut
    
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

def process_dynamic(dynamic_info_p, dynamic_info, data_fps = 10, target_fps = 2, num_fut = 4):
    interval = int(data_fps / target_fps)
    find = False
    n_full_frames = -1
    for dcls, cinfo in dynamic_info_p.items():
        for dobj in cinfo:
            n_full_frames = dobj['n_full_frames']
            find = True
            break
        if find:
            break

    results = [[] for _ in range((n_full_frames + interval - 1)// interval)]
    for dcls, cinfo in dynamic_info_p.items():
        for did, dobj in enumerate(cinfo):
            strat_idx = dobj["segments"][0]["start_frame"]
            n_frames = dobj["segments"][0]["n_frames"]
            if (n_frames + interval - 1) // interval <= num_fut :
                continue
            rel_poses = []
            xys = []
            for trans in dobj["segments"][0]["data"]["transform"][::interval]:
                rel_p, dis = get_obj_rel_position(trans)
                rel_poses.append(rel_p)
                xys.append(dis)

            speeds, speed_plans, path_plans,navi_commands = get_plan(np.array(dynamic_info[dcls][did]["segments"][0]["data"]["transform"]))

            for i in range(len(rel_poses)):
                frame_idx = strat_idx + i * interval
                try:
                    results[frame_idx // interval].append((dcls, rel_poses[i], xys[i], speeds[i], speed_plans[i], path_plans[i]))
                except Exception as e:
                    pdb.set_trace()
                    print(e)

    return results
def gen_info(root):
    for sp in [ "validation","training"]:
        for seq in tqdm(os.listdir(f"{root}/{sp}")):
        # for seq in ["segment-15832924468527961_1564_160_1584_160_with_camera_labels"]:
            raw_poses = np.load(f"{root}/{sp}/{seq}/raw_poses.npy")
            base_pose = np.load(f"{root}/{sp}/{seq}/base_pose.npy")
            re_poses = np.load(f"{root}/{sp}/{seq}/re_poses.npy")
            
            speeds, speed_plans, path_plans ,navi_commands= get_plan(raw_poses)
            
            results = [(float(speed), sp, pp,nc) for speed, sp, pp,nc in zip(speeds, speed_plans, path_plans,navi_commands)]
            with open(f"{root}/{sp}/{seq}/ego_results1.json", "w") as f:
                json.dump(results, f)
                
            with open(f"{root}/{sp}/{seq}/dynamic_objects_pose_processed.json", "r") as f:
                dynamic_info_p = json.load(f)

            with open(f"{root}/{sp}/{seq}/dynamic_objects.json", "r") as f:
                dynamic_info = json.load(f)
                
            results = process_dynamic(dynamic_info_p, dynamic_info)

            with open(f"{root}/{sp}/{seq}/dynamic_objects_results1.json", "w") as f:
                json.dump(results, f)
            
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
def q1(images, infos, vru_dis_thresh=20.0):
    question = f"<FRONT VIEW>:<image>\nDo you see any vulnerable road users within {int(vru_dis_thresh)} meters ahead of you, such as pedestrians or cyclists?"  
    qas = []  
    for i, info in enumerate(infos):
        vru_list = []
        for obj in info:
            cls, rel_p, loc, speed, speed_plan, path_plan = obj
            dis = math.sqrt(loc[0]**2 + loc[1]**2)
            x, y = loc
            if rel_p == "front center" and cls in ["Pedestrian", "Cyclist"] and dis <= vru_dis_thresh:
                if y <= -2:
                    lat_pos = f" and {int(abs(y))} meters to the right"
                elif y >= 2:
                    lat_pos = f" and {int(abs(y))} meters to the left"
                else:
                    lat_pos = ""
                vru_description = f"a {cls.lower()} located {int(abs(x))} meters ahead of me{lat_pos}"
                vru_list.append(vru_description)
        if vru_list:
            answer = "Yes, I see " + ", and ".join(vru_list) + "."
        else:
            answer = "No, I don't see any vulnerable road users ahead of me, " \
                    "such as pedestrians or cyclists."
        qas.append({"images": [images[i][0]], "messages": [{"role":"user", "content": question}, {"role":"assistant", "content": answer}]})
    return qas

def q2(images, infos, dis_thresh=40.0):
    qas = []
    for i, info in enumerate(infos):
        views = ["front center", "front left", "front right", "side left", "side right"]
        obj_cnt = {view: 0 for view in views}
        qs = {view: "You are driving, I will now provide you with the location " \
            f"and velocity information of dynamic objects in the {view} view image <image>. " \
            "Please predict their future driving behaviors, " \
            "which can be divided into SPEED decisions and PATH decisions. " \
            "SPEED includes KEEP, ACCELERATE, DECELERATE, and STOP, " \
            "while PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, and LEFT_TURN." \
            "I will now provide you with the position and velocity information of the dynamic objects: \n" for view in views}
        ans = {view: "" for view in views}
        for obj in info:
            cls, rel_p, loc, speed, speed_plan, path_plan = obj
            dis = math.sqrt(loc[0]**2 + loc[1]**2)
            if dis >= dis_thresh:
                continue
            if cls == "unknown" or cls == "Sign":
                continue
            if rel_p == "rear left" or rel_p == "rear right":
                continue
            x, y = loc
            if x >= 0:
                log_describe = f"{int(x)} meters ahead"
            else:
                log_describe = f"{abs(int(x))} meters behind"
            if y >= 0:
                lat_describe = f"{int(y)} meters to the left"
            else:
                lat_describe = f"{abs(int(y))} meters to the right"
            obj_cnt[rel_p] += 1
            obj_info = f'Object {obj_cnt[rel_p]}: {cls}, {log_describe}, {lat_describe}, speed of {int(speed)} m/s.'

            qs[rel_p] += obj_info + '\n'
            ans[rel_p] += f"Object {obj_cnt[rel_p]}: {pedal_status[speed_plan]}, {path_status[path_plan]}\n"
        for vi, view in enumerate(views):
            qas.append({"images": [images[i][vi]], "messages": [{"role":"user", "content": qs[view]}, {"role":"assistant", "content": ans[view]}]})
    return qas

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
        image_prompt = "<FRONT VIEW>:\n<image>\n" \
                   "<FRONT LEFT VIEW>:\n<image>\n" \
                   "<FRONT RIGHT VIEW>:\n<image>\n" \
                   "<SIDE LEFT VIEW>:\n<image>\n" \
                   "<SIDE RIGHT VIEW>:\n<image>\n"
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
    views = ["front center", "front left", "front right", "side left", "side right"]
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
        image_prompt = "<FRONT VIEW>:\n<image>\n" \
                   "<FRONT LEFT VIEW>:\n<image>\n" \
                   "<FRONT RIGHT VIEW>:\n<image>\n" \
                   "<SIDE LEFT VIEW>:\n<image>\n" \
                   "<SIDE RIGHT VIEW>:\n<image>\n"
                 
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

def gen_qa(data_root, qa_root):
    for sp in ["validation","training"]:
        q1s = []
        q2s = []
        q3s = []
        q4s = []
        q5s = []
        q6s = []
        for seq in tqdm(os.listdir(f"{data_root}/{sp}")):
            with open(f'{data_root}/{sp}/{seq}/ego_results1.json', "r") as f:
                ego = json.load(f)
            with open(f"{data_root}/{sp}/{seq}/dynamic_objects_results1.json", "r") as f:
                dynamic = json.load(f)
            views = ['camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT','camera_SIDE_LEFT', 'camera_SIDE_RIGHT']
            
            file_list = {}
            for view in views:
                file_list[view] = [f"{data_root}/{sp}/{seq}/images/{view}/{img}" for img in sorted(os.listdir(f"{data_root}/{sp}/{seq}/images/{view}"))[::5]]
            
            # pdb.set_trace()
            images = [[file_list[key][i] for key in views] for i in range(len(ego))]
            q1s += q1(images, dynamic)
            q2s += q2(images, dynamic)
            q3s += q3(images, ego)
            q4s += q4(images)
            q5s += q5(images)
            q6s += q6(images,ego)
        print(sp, len(q1s), len(q2s), len(q3s), len(q4s), len(q5s),len(q6s))
        if sp=="validation":
            sp="val"
        elif sp=="training":
            sp="train"
        with open(f"{qa_root}/q1_{sp}.json", "w") as f:
            json.dump(q1s, f)
        with open(f"{qa_root}/q2_{sp}.json", "w") as f:
            json.dump(q2s, f)
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
