# We follow Senna to load the nuScenes dataset: GitHub - hustvl/Senna: Bridging Large Vision-Language Models and End-to-End Autonomous Driving.
'''
Senna driving QA dataset based on nuScenes
using LlaVA format
using LLaVA-1.6-34b for generate surround img description
'''
import sys
import re
import os
import uuid
import json
import math
import copy
import requests
import argparse
from io import BytesIO
from os import path as osp

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix

import math
import re
import requests
from io import BytesIO

import torch
import numpy as np
from PIL import Image


def get_obj_acc_or_dec(trajectory, vel_diff_thresh=3.0):
    velocity = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=-1) / 0.5

    if np.max(velocity) < 2.0:
        return "stop"

    vel_diff = velocity[-1] - velocity[0]

    if vel_diff >= vel_diff_thresh:
        return "accelerate"
    elif vel_diff <= -vel_diff_thresh:
        return "decelerate"
    else:
        return "const"

# 判断变道或拐弯


def get_obj_turn_or_lane_change(trajectory, lat_thresh=4.0, angle_thresh=5.0):
    # 提取横向位置和纵向位置
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    # 计算车辆角度变化
    endpoint_angle = math.degrees(math.atan2(y[-1], x[-1]))
    angle_diff = endpoint_angle - 90.0

    # 判断是否进行变道或转弯
    if x[-1] > lat_thresh and angle_diff <= -angle_thresh:
        return "right turn"
    elif x[-1] > lat_thresh and abs(angle_diff) < angle_thresh:
        return "right lane change"
    elif x[-1] <= -lat_thresh and angle_diff >= angle_thresh:
        return "left turn"
    elif x[-1] <= -lat_thresh and abs(angle_diff) < angle_thresh:
        return "left lane change"
    else:
        return "straight"

# 判断车辆在自车的什么位置 （前面、左前、右前、左后、右后、后面）


def get_obj_rel_position(loc):
    # nuscenes camera fov: 70 (except rear cam: 110)
    cf_fov = 70.0
    cf_start = 90.0 - cf_fov / 2
    cam_offset = 55
    cb_fov = 110

    cf_range = [cf_start, cf_start+cf_fov]  # [55, 125]
    cfl_range = [cf_start+cam_offset, cf_start+cf_fov+cam_offset]  # [110, 180]
    cbl_range = [cf_start+2*cam_offset, cf_start +
                 cf_fov+2*cam_offset]  # [165, 235]
    cfr_range = [cf_start-cam_offset, cf_start+cf_fov-cam_offset]  # [0, 70]
    cbr_range = [cf_start-2*cam_offset,
                 cf_start+cf_fov-2*cam_offset]  # [-55, 15]
    cb_range = [(cb_fov-180)/2, (cb_fov-180)/2-cb_fov]  # [-35, -145]

    x, y = loc[0], loc[1]
    angle = math.degrees(math.atan2(y, x))
    angle1 = angle if angle >= 0 else angle + 360

    if angle1 >= cf_range[0] and angle1 < cf_range[1]:
        return "front"
    elif angle1 >= cfl_range[0] and angle1 < cfl_range[1]:
        return "front left"
    elif angle1 >= cbl_range[0] and angle1 < cbl_range[1]:
        return "back left"
    elif angle1 >= cfr_range[0] and angle1 < cfr_range[1]:
        return "front right"
    elif angle >= cbr_range[0] and angle < cbr_range[1]:
        return "back right"
    elif angle < cb_range[0] and angle >= cb_range[1]:
        return "back"  # overlap with side cams
    else:
        raise Exception("Not in any camera range!")


idx_class_mapping = ["car", "cyclist", "pedestrian"]

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


# def eval_llava_34b_wo_init(args, tokenizer, model, image_processor):
#     # Model
#     disable_torch_init()
#     image = load_image(args.img_file)
#     image_tensor = process_images([image], image_processor, model.config)
#     image_tensor = [_image.to(dtype=torch.float16, device=model.device) for _image in image_tensor]

#     conv_template = "chatml_direct" # Make sure you use correct chat template for different models
#     question = DEFAULT_IMAGE_TOKEN + "\n" + args.query
#     conv = copy.deepcopy(conv_templates[conv_template])
#     conv.append_message(conv.roles[0], question)
#     conv.append_message(conv.roles[1], None)
#     prompt_question = conv.get_prompt()

#     input_ids = tokenizer_image_token(
#         prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
#     image_sizes = [image.size]

#     cont = model.generate(
#         input_ids,
#         images=image_tensor,
#         image_sizes=image_sizes,
#         do_sample=False,
#         temperature=0,
#         max_new_tokens=512,
#     )
#     text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

#     return text_outputs


nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

NameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}

ego_width, ego_length = 1.85, 4.084


def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw


def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i


def create_nuscenes_infos(root_path,
                          out_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    print(version, root_path)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    # train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
    #     nusc, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps)
    _fill_trainval_infos(
        nusc, nusc_can_bus, train_scenes, val_scenes, out_path=out_path, test=test, max_sweeps=max_sweeps, version=version)

    # metadata = dict(version=version)

    # if test:
    #     # print('test sample: {}'.format(len(train_nusc_infos)))
    #     data = dict(infos=train_nusc_infos, metadata=metadata)
    #     info_path = osp.join(out_path,
    #                          '{}_infos_temporal_test.json'.format(info_prefix))
    #     with open(info_path, "w") as f:
    #         json.dump(data, f)
    # else:
    #     info_path = osp.join(out_path,
    #                          '{}_train.json'.format(info_prefix))
    #     with open(info_path, "w") as f:
    #         json.dump(train_nusc_infos, f)
    #     info_val_path = osp.join(out_path,
    #                              '{}_val.json'.format(info_prefix))
    #     with open(info_val_path, "w") as f:
    #         json.dump(val_nusc_infos, f)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not os.path.isfile(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0., 0.])
    return np.array(can_bus)


def _fill_trainval_infos(nusc,
                         nusc_can_bus,
                         train_scenes,
                         val_scenes,
                         out_path='',
                         test=False,
                         max_sweeps=10,
                         version='v1.0-trainval',
                         fut_ts=10,
                         his_ts=6):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """

    # load vlm model and generate
    # from llava_next.llava.model.builder import load_pretrained_model
    # pretrained = "/path/to/llava-v1.6-34b"
    # model_name = "llava-v1.6-34b"
    # device_map = "auto"
    # tokenizer, model, image_processor, max_length = load_pretrained_model(
    #     pretrained,
    #     None,
    #     model_name,
    #     device_map=device_map,
    #     # load_4bit=True,
    #     load_8bit=True,
    #     attn_implementation=None)
    # model.eval()
    # model.tie_weights()

    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    cat2idx = {}
    for idx, dic in enumerate(nusc.category):
        cat2idx[dic['name']] = idx

    for sample in tqdm(nusc.sample):
        map_location = nusc.get('log', nusc.get(
            'scene', sample['scene_token'])['log_token'])['location']
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        if sample['prev'] != '':
            sample_prev = nusc.get('sample', sample['prev'])
            sd_rec_prev = nusc.get(
                'sample_data', sample_prev['data']['LIDAR_TOP'])
            pose_record_prev = nusc.get(
                'ego_pose', sd_rec_prev['ego_pose_token'])
        else:
            pose_record_prev = None
        if sample['next'] != '':
            sample_next = nusc.get('sample', sample['next'])
            sd_rec_next = nusc.get(
                'sample_data', sample_next['data']['LIDAR_TOP'])
            pose_record_next = nusc.get(
                'ego_pose', sd_rec_next['ego_pose_token'])
        else:
            pose_record_next = None

        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        assert os.path.isfile(lidar_path)
        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)
        fut_valid_flag = True
        test_sample = copy.deepcopy(sample)
        for i in range(fut_ts):
            if test_sample['next'] != '':
                test_sample = nusc.get('sample', test_sample['next'])
            else:
                fut_valid_flag = False
        ##
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'can_bus': can_bus,
            'frame_idx': frame_idx,  # temporal related info
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],  # temporal related info
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'fut_valid_flag': fut_valid_flag,
            'map_location': map_location
        }

        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        annotations = [
            nusc.get('sample_annotation', token)
            for token in sample['anns']
        ]
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        velocity = np.array(
            [nusc.box_velocity(token)[:2] for token in sample['anns']])
        valid_flag = np.array(
            [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                for anno in annotations],
            dtype=bool).reshape(-1)
        # convert velo from global to lidar
        for i in range(len(boxes)):
            velo = np.array([*velocity[i], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                l2e_r_mat).T
            velocity[i] = velo[:2]

        names = [b.name for b in boxes]
        for i in range(len(names)):
            if names[i] in NameMapping:
                names[i] = NameMapping[names[i]]
        names = np.array(names)
        # we need to convert rot to SECOND format.
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        assert len(gt_boxes) == len(
            annotations), f'{len(gt_boxes)}, {len(annotations)}'

        # get future coords for each box
        # [num_box, fut_ts*2]
        num_box = len(boxes)
        gt_fut_trajs = np.zeros((num_box, fut_ts, 2))
        gt_fut_trajs_vcs = np.zeros((num_box, fut_ts, 2))
        gt_fut_yaw = np.zeros((num_box, fut_ts))
        gt_fut_yaw_vcs = np.zeros((num_box, fut_ts))
        gt_fut_masks = np.zeros((num_box, fut_ts))
        gt_boxes_yaw = -(gt_boxes[:, 6] + np.pi / 2)
        # agent lcf feat (x, y, yaw, vx, vy, width, length, height, type)
        agent_lcf_feat = np.zeros((num_box, 9))
        gt_fut_goal = np.zeros((num_box))
        for i, anno in enumerate(annotations):
            cur_box = boxes[i]
            box_lcf_vcs = Box(anno['translation'],
                              anno['size'],
                              Quaternion(anno['rotation']))
            box_lcf_vcs.translate(-np.array(cs_record['translation']))
            box_lcf_vcs.rotate(Quaternion(cs_record['rotation']).inverse)
            cur_anno = anno
            lcf_anno = copy.deepcopy(anno)
            agent_lcf_feat[i, 0:2] = cur_box.center[:2]
            agent_lcf_feat[i, 2] = gt_boxes_yaw[i]
            agent_lcf_feat[i, 3:5] = velocity[i]
            agent_lcf_feat[i, 5:8] = anno['size']  # width,length,height
            agent_lcf_feat[i, 8] = cat2idx[anno['category_name']
                                           ] if anno['category_name'] in cat2idx.keys() else -1
            for j in range(fut_ts):
                if cur_anno['next'] != '':
                    anno_next = nusc.get(
                        'sample_annotation', cur_anno['next'])
                    box_next = Box(
                        anno_next['translation'], anno_next['size'], Quaternion(
                            anno_next['rotation'])
                    )
                    # Move box to vehicle lcf coord system.
                    box_next_vcs = copy.deepcopy(box_next)
                    box_next_vcs.translate(
                        -np.array(lcf_anno['translation']))
                    box_next_vcs.rotate(Quaternion(
                        lcf_anno['rotation']).inverse)
                    # Move box to sensor coord system.
                    box_next_vcs.translate(
                        -np.array(cs_record['translation']))
                    box_next_vcs.rotate(Quaternion(
                        cs_record['rotation']).inverse)
                    # Move box to ego vehicle coord system.
                    box_next.translate(
                        -np.array(pose_record['translation']))
                    box_next.rotate(Quaternion(
                        pose_record['rotation']).inverse)
                    # Move box to sensor coord system.
                    box_next.translate(-np.array(cs_record['translation']))
                    box_next.rotate(Quaternion(
                        cs_record['rotation']).inverse)
                    gt_fut_trajs[i, j] = box_next.center[:2] - \
                        cur_box.center[:2]
                    gt_fut_masks[i, j] = 1
                    gt_fut_trajs_vcs[i, j] = box_next_vcs.center[:2]
                    # add yaw diff
                    _, _, box_yaw = quart_to_rpy([cur_box.orientation.x, cur_box.orientation.y,
                                                  cur_box.orientation.z, cur_box.orientation.w])
                    _, _, box_yaw_next = quart_to_rpy([box_next.orientation.x, box_next.orientation.y,
                                                       box_next.orientation.z, box_next.orientation.w])
                    gt_fut_yaw[i, j] = box_yaw_next - box_yaw
                    _, _, box_lcf_yaw_vcs = quart_to_rpy([box_lcf_vcs.orientation.x, box_lcf_vcs.orientation.y,
                                                          box_lcf_vcs.orientation.z, box_lcf_vcs.orientation.w])
                    _, _, box_yaw_vcs_next = quart_to_rpy([box_next_vcs.orientation.x, box_next_vcs.orientation.y,
                                                           box_next_vcs.orientation.z, box_next_vcs.orientation.w])
                    gt_fut_yaw_vcs[i, j] = box_yaw_vcs_next - \
                        box_lcf_yaw_vcs

                    cur_anno = anno_next
                    cur_box = box_next
                else:
                    gt_fut_trajs[i, j:] = 0
                    break
            # get agent goal
            gt_fut_coords = np.cumsum(gt_fut_trajs[i], axis=-2)
            coord_diff = gt_fut_coords[-1] - gt_fut_coords[0]
            if coord_diff.max() < 1.0:  # static
                gt_fut_goal[i] = 9
            else:
                box_mot_yaw = np.arctan2(
                    coord_diff[1], coord_diff[0]) + np.pi
                # 0-8: goal direction class
                gt_fut_goal[i] = box_mot_yaw // (np.pi / 4)

        # get ego history traj (offset format)
        ego_his_trajs = np.zeros((his_ts+1, 3))
        ego_his_trajs_diff = np.zeros((his_ts+1, 3))
        sample_cur = sample
        for i in range(his_ts, -1, -1):
            if sample_cur is not None:
                pose_mat = get_global_sensor_pose(
                    sample_cur, nusc, inverse=False)
                ego_his_trajs[i] = pose_mat[:3, 3]
                has_prev = sample_cur['prev'] != ''
                has_next = sample_cur['next'] != ''
                if has_next:
                    sample_next = nusc.get('sample', sample_cur['next'])
                    pose_mat_next = get_global_sensor_pose(
                        sample_next, nusc, inverse=False)
                    ego_his_trajs_diff[i] = pose_mat_next[:3,
                                                          3] - ego_his_trajs[i]
                sample_cur = nusc.get(
                    'sample', sample_cur['prev']) if has_prev else None
            else:
                ego_his_trajs[i] = ego_his_trajs[i+1] - \
                    ego_his_trajs_diff[i+1]
                ego_his_trajs_diff[i] = ego_his_trajs_diff[i+1]

        # global to ego at lcf
        ego_his_trajs = ego_his_trajs - \
            np.array(pose_record['translation'])
        rot_mat = Quaternion(
            pose_record['rotation']).inverse.rotation_matrix
        ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
        # ego to lidar at lcf
        ego_his_trajs = ego_his_trajs - np.array(cs_record['translation'])
        rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
        ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
        ego_his_trajs = ego_his_trajs[1:] - ego_his_trajs[:-1]

        # get ego futute traj (offset format)
        ego_fut_trajs = np.zeros((fut_ts+1, 3))
        ego_fut_masks = np.zeros((fut_ts+1))
        sample_cur = sample
        for i in range(fut_ts+1):
            pose_mat = get_global_sensor_pose(
                sample_cur, nusc, inverse=False)
            ego_fut_trajs[i] = pose_mat[:3, 3]
            ego_fut_masks[i] = 1
            if sample_cur['next'] == '':
                ego_fut_trajs[i+1:] = ego_fut_trajs[i]
                break
            else:
                sample_cur = nusc.get('sample', sample_cur['next'])
        # global to ego at lcf
        ego_fut_trajs = ego_fut_trajs - \
            np.array(pose_record['translation'])
        rot_mat = Quaternion(
            pose_record['rotation']).inverse.rotation_matrix
        ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
        # ego to lidar at lcf
        ego_fut_trajs = ego_fut_trajs - np.array(cs_record['translation'])
        rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
        ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
        # drive command according to final fut step offset from lcf
        if ego_fut_trajs[-1][0] >= 2:
            command = np.array([1, 0, 0])  # Turn Right
        elif ego_fut_trajs[-1][0] <= -2:
            command = np.array([0, 1, 0])  # Turn Left
        else:
            command = np.array([0, 0, 1])  # Go Straight
        # offset from lcf -> per-step offset
        ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]

        # Senna: get navi info from long-horzion (6s) future trajs
        long_fut_ts = 12
        ego_navi_trajs = np.zeros((long_fut_ts+1, 3))
        sample_cur = sample
        for i in range(13):
            pose_mat = get_global_sensor_pose(
                sample_cur, nusc, inverse=False)
            ego_navi_trajs[i] = pose_mat[:3, 3]
            if sample_cur['next'] == '':
                ego_navi_trajs[i+1:] = ego_navi_trajs[i]
                break
            else:
                sample_cur = nusc.get('sample', sample_cur['next'])
        # global to ego at lcf
        ego_navi_trajs = ego_navi_trajs - \
            np.array(pose_record['translation'])
        rot_mat = Quaternion(
            pose_record['rotation']).inverse.rotation_matrix
        ego_navi_trajs = np.dot(rot_mat, ego_navi_trajs.T).T
        # ego to lidar at lcf
        ego_navi_trajs = ego_navi_trajs - \
            np.array(cs_record['translation'])
        rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
        ego_navi_trajs = np.dot(rot_mat, ego_navi_trajs.T).T
        # drive command according to final fut step offset from lcf
        # discard current timestamp
        ego_navi_trajs = ego_navi_trajs[1:, :2]
        target_point = ego_navi_trajs[[-1], :2]
        if target_point[0, 0] >= 20.0 and target_point[0, 1] >= 10.0:
            ego_navi_cmd = 'go straight and turn left'
        elif target_point[0, 0] >= 20.0 and target_point[0, 1] <= -10.0:
            ego_navi_cmd = 'go straight and turn right'
        elif target_point[0, 0] < 20.0 and target_point[0, 1] >= 10.0:
            ego_navi_cmd = 'turn left'
        elif target_point[0, 0] < 20.0 and target_point[0, 1] <= -10.0:
            ego_navi_cmd = 'turn right'
        else:
            ego_navi_cmd = 'go straight'

        # ego lcf feat (vx, vy, ax, ay, w, length, width, vel, steer)
        ego_lcf_feat = np.zeros(9)
        _, _, ego_yaw = quart_to_rpy(pose_record['rotation'])
        ego_pos = np.array(pose_record['translation'])
        if pose_record_prev is not None:
            _, _, ego_yaw_prev = quart_to_rpy(pose_record_prev['rotation'])
            ego_pos_prev = np.array(pose_record_prev['translation'])
        if pose_record_next is not None:
            _, _, ego_yaw_next = quart_to_rpy(pose_record_next['rotation'])
            ego_pos_next = np.array(pose_record_next['translation'])
        assert (pose_record_prev is not None) or (
            pose_record_next is not None), 'prev token and next token all empty'
        if pose_record_prev is not None:
            ego_w = (ego_yaw - ego_yaw_prev) / 0.5
            ego_v = np.linalg.norm(ego_pos[:2] - ego_pos_prev[:2]) / 0.5
            ego_vx, ego_vy = ego_v * \
                math.cos(ego_yaw + np.pi/2), ego_v * \
                math.sin(ego_yaw + np.pi/2)
        else:
            ego_w = (ego_yaw_next - ego_yaw) / 0.5
            ego_v = np.linalg.norm(ego_pos_next[:2] - ego_pos[:2]) / 0.5
            ego_vx, ego_vy = ego_v * \
                math.cos(ego_yaw + np.pi/2), ego_v * \
                math.sin(ego_yaw + np.pi/2)

        ref_scene = nusc.get("scene", sample['scene_token'])
        try:
            pose_msgs = nusc_can_bus.get_messages(
                ref_scene['name'], 'pose')
            steer_msgs = nusc_can_bus.get_messages(
                ref_scene['name'], 'steeranglefeedback')
            pose_uts = [msg['utime'] for msg in pose_msgs]
            steer_uts = [msg['utime'] for msg in steer_msgs]
            ref_utime = sample['timestamp']
            pose_index = locate_message(pose_uts, ref_utime)
            pose_data = pose_msgs[pose_index]
            steer_index = locate_message(steer_uts, ref_utime)
            steer_data = steer_msgs[steer_index]
            # initial speed
            # [0] means longitudinal velocity  m/s
            v0 = pose_data["vel"][0]
            # curvature (positive: turn left)
            steering = steer_data["value"]
            # flip x axis if in left-hand traffic (singapore)
            flip_flag = True if map_location.startswith(
                'singapore') else False
            if flip_flag:
                steering *= -1
            Kappa = 2 * steering / 2.588
        except:
            delta_x = ego_his_trajs[-1, 0] + ego_fut_trajs[0, 0]
            delta_y = ego_his_trajs[-1, 1] + ego_fut_trajs[0, 1]
            v0 = np.sqrt(delta_x**2 + delta_y**2)
            Kappa = 0

        ego_lcf_feat[:2] = np.array([ego_vx, ego_vy])  # can_bus[13:15]
        ego_lcf_feat[2:4] = can_bus[7:9]
        ego_lcf_feat[4] = ego_w  # can_bus[12]
        ego_lcf_feat[5:7] = np.array([ego_length, ego_width])
        ego_lcf_feat[7] = v0
        ego_lcf_feat[8] = Kappa

        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['gt_velocity'] = velocity.reshape(-1, 2)
        info['num_lidar_pts'] = np.array(
            [a['num_lidar_pts'] for a in annotations])
        info['num_radar_pts'] = np.array(
            [a['num_radar_pts'] for a in annotations])
        info['valid_flag'] = valid_flag
        info['gt_agent_fut_trajs'] = gt_fut_trajs.reshape(
            -1, fut_ts*2).astype(np.float32)
        info['gt_agent_fut_masks'] = gt_fut_masks.reshape(
            -1, fut_ts).astype(np.float32)
        info['gt_agent_lcf_feat'] = agent_lcf_feat.astype(np.float32)
        info['gt_agent_fut_yaw'] = gt_fut_yaw.astype(np.float32)
        info['gt_agent_fut_goal'] = gt_fut_goal.astype(np.float32)
        info['gt_ego_his_trajs'] = ego_his_trajs[:, :2].astype(np.float32)
        info['gt_ego_fut_trajs'] = ego_fut_trajs[:, :2].astype(np.float32)
        info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32)
        info['gt_ego_fut_cmd'] = command.astype(np.float32)
        info['gt_ego_lcf_feat'] = ego_lcf_feat.astype(np.float32)

        info['gt_agent_fut_trajs_vcs'] = gt_fut_trajs_vcs.astype(
            np.float32)
        info['gt_agent_fut_yaw_vcs'] = gt_fut_yaw_vcs.astype(np.float32)

        ######## Senna data processing ########
        info['ego_navi_cmd'] = ego_navi_cmd
        camera_types = ['CAM_FRONT',
                        'CAM_FRONT_RIGHT',
                        'CAM_FRONT_LEFT',
                        'CAM_BACK',
                        'CAM_BACK_LEFT',
                        'CAM_BACK_RIGHT']
        info['images'] = [info['cams'][idx]['data_path']
                          for idx in camera_types]
        generate_drive_qa(info, out_path, version=version)
        # ego_fut_trajs = np.cumsum(info['gt_ego_fut_trajs'], axis=-2).reshape(-1)
        # ego_fut_trajs = str(list(ego_fut_trajs))
        # vlm_info = {}
        # for i in range(len(qa_data)):
        #     uid = uuid.uuid4()
        #     uid_str = str(uid)
        #     vlm_info['id'] =  uid_str

        #     vlm_info = {
        #         'token': info['token'],
        #         'image': info['images'][0],
        #         'images': info['images'],
        #         'prev': info['prev'],
        #         'next': info['next'],
        #         'scene_token': info['scene_token'],
        #         'fut_valid_flag': str(info['fut_valid_flag']),
        #         'ego_fut_trajs': ego_fut_trajs,
        #         'conversations': qa_data[i]
        #     }

        #     if vlm_info['scene_token'] in train_scenes:
        #         train_nusc_infos.append(vlm_info)
        #     else:
        #         val_nusc_infos.append(vlm_info)

    # return train_nusc_infos, val_nusc_infos


def get_global_sensor_pose(rec, nusc, inverse=False):
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])

    sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor",
                     lidar_sample_data["calibrated_sensor_token"])
    if inverse is False:
        global_from_ego = transform_matrix(
            sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(
            sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
        # translation equivalent writing
        # pose_translation = np.array(sd_cs["translation"])
        # rot_mat = Quaternion(sd_ep['rotation']).rotation_matrix
        # pose_translation = np.dot(rot_mat, pose_translation)
        # # pose_translation = pose[:3, 3]
        # pose_translation = pose_translation + np.array(sd_ep["translation"])
    else:
        sensor_from_ego = transform_matrix(
            sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
        ego_from_global = transform_matrix(
            sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    return pose


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)


###  Senna nuScenes QA tool functions ###

def get_vru_qa(info, img_paths, vru_dis_thresh=20.0):
    # question = f"Do you see any vulnerable road users within {int(vru_dis_thresh)} meters ahead of you, " \
    #            "such as cyclists, motorcyclists, or pedestrians?"
    question = f"<FRONT VIEW>:\n<image>\nDo you see any vulnerable road users within {int(vru_dis_thresh)} meters ahead of you, " \
               "such as cyclists, motorcyclists, or pedestrians?"
    vru_list = []
    num_objects = info['gt_boxes'].shape[0]
    vru_classes = ['bicycle', 'motorcycle', 'pedestrian']

    for i in range(num_objects):
        obj_loc = info['gt_boxes'][i, :2]
        obj_rel_loc = get_obj_rel_position(obj_loc)
        if obj_rel_loc != "front":
            continue  # only consider VRUs at front

        obj_cls = info['gt_names'][i]
        lat_dis, log_dis = info['gt_boxes'][i, 0], info['gt_boxes'][i, 1]

        if obj_cls in vru_classes and np.linalg.norm(obj_loc) < vru_dis_thresh:
            if lat_dis <= -2.0:
                lat_pos = f" and {int(abs(lat_dis))} meters to the left"
            elif lat_dis >= 2.0:
                lat_pos = f" and {int(abs(lat_dis))} meters to the right"
            else:
                lat_pos = ""
            vru_description = f"a {obj_cls} located {int(abs(log_dis))} meters ahead of me{lat_pos}"
            vru_list.append(vru_description)

    if vru_list:
        answer = "Yes, I see " + ", and ".join(vru_list) + "."
    else:
        answer = "No, I don't see any vulnerable road users ahead of me, " \
                 "such as bicycles, motorcycles, or pedestrians."

    return {
        "images": img_paths,
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    }


def get_mot_pred_qa(info,
                    img_type='front',
                    img_paths=[],
                    dis_thresh=40.0):

    # question = "You are driving, I will now provide you with the location " \
    #            f"and velocity information of dynamic objects in the {img_type} view image. " \
    #            "Please predict their future driving behaviors, " \
    #            "which can be divided into SPEED decisions and PATH decisions. " \
    #            "SPEED includes KEEP, ACCELERATE, DECELERATE, and STOP, " \
    #            "while PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, and LEFT_TURN." \
    #            "I will now provide you with the position and velocity information of the dynamic objects: \n"
    question = "<FRONT VIEW>:\n<image>\nYou are driving, I will now provide you with the location and velocity information of dynamic objects in the front view image. Please predict their future driving behaviors, which can be divided into SPEED decisions and PATH decisions. SPEED includes KEEP, ACCELERATE, DECELERATE, and STOP, while PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, and LEFT_TURN.I will now provide you with the position and velocity information of the dynamic objects: "
    num_objects = info['gt_boxes'].shape[0]
    obj_cnt = 0
    answer = ""
    for i in range(num_objects):
        obj_loc = info['gt_boxes'][i, :2]
        obj_rel_loc = get_obj_rel_position(obj_loc)
        if obj_rel_loc != img_type:
            continue  # only care corresbouding view objects
        if np.linalg.norm(obj_loc, axis=-1) >= dis_thresh:
            continue  # only care objects with distance thresholds
        if np.any(info['gt_agent_fut_masks'][i] == 0):
            continue  # 如果object没有未来轨迹信息，则不考虑此object
        lat_dis, log_dis = info['gt_boxes'][i, 0], info['gt_boxes'][i, 1]
        obj_fut_traj = info["gt_agent_fut_trajs_vcs"][i]
        obj_pedal_status = get_obj_acc_or_dec(obj_fut_traj)
        obj_wheel_status = get_obj_turn_or_lane_change(obj_fut_traj)
        obj_speed_plan = pedal_status[obj_pedal_status]
        obj_path_plan = path_status[obj_wheel_status]
        obj_cls = info['gt_names'][i]
        obj_speed = np.linalg.norm(
            obj_fut_traj[1] - obj_fut_traj[0], axis=-1) / 0.5
        obj_cnt = obj_cnt + 1
        if log_dis >= 0:
            log_describe = f"{int(log_dis)} meters ahead"
        else:
            log_describe = f"{abs(int(log_dis))} meters behind"
        if lat_dis >= 0:
            lat_describe = f"{int(lat_dis)} meters to the right"
        else:
            lat_describe = f"{abs(int(lat_dis))} meters to the left"

        obj_info = f'Object {obj_cnt}: {obj_cls}, {log_describe}, {lat_describe}, speed of {int(obj_speed)} m/s.'

        question = question + obj_info + '\n'
        answer = answer + \
            f"Object {obj_cnt}: {obj_speed_plan}, {obj_path_plan}\n"

    # question_end = "Please predict the future driving behaviors of these objects " \
    #                f"based on the {img_type} view image. " \
    #                "For example, a well-formatted answer should be like:\n" \
    #                "Object 1: KEEP, STRAIGHT\n" \
    #                "Object 2: DECELERATE, RIGHT_TURN\n" \
    #                "Object 3: ACCELERATE, LEFT_CHANGE\n"
    question_end = "Please predict the future driving behaviors of these objects based on the front view image. For example, a well-formatted answer should be like:\nObject 1: KEEP, STRAIGHT\nObject 2: DECELERATE, RIGHT_TURN\nObject 3: ACCELERATE, LEFT_CHANGE\n For example, a correct answer format is like '<DYNAMIC OBJECTS>Object 1: KEEP, STRAIGHT\nObject 2: STOP, STRAIGHT\n</DYNAMIC OBJECTS>."

    question = question + question_end
    answer = f"<DYNAMIC OBJECTS>{answer}</DYNAMIC OBJECTS>"
    qa = format_qa(question, answer)

    if obj_cnt == 0:
        return None

    return {
        "images": img_paths,
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    }


"Given the provided forward-facing image <image> from a car's perspective, identify if there is a traffic light that affects the car's behavior. Respond with 'Red', 'Green', 'Yellow', or 'None'."


def get_traffic_light_qa(img_paths):
    '''
    nuscenes does not contain traffic light labels, so we use pseudo labels generated by vlm
    '''

    # question = "Given the provided forward-facing image from a car's perspective, " \
    #            "identify if there is a traffic light that affects the car's behavior. " \
    #            "Respond with 'Red', 'Green', 'Yellow', or 'None'."
    question = "<FRONT VIEW>:\n<image>\nGiven the provided forward-facing image <image> from a car's perspective, identify if there is a traffic light that affects the car's behavior. Respond with 'Red', 'Green', 'Yellow', or 'None'."
    # args = type('Args', (), {
    #     "query": question,
    #     "img_file": cf_img_path,
    # })()

    # answer = eval_llava_34b_wo_init(args, tokenizer, model, image_processor)
    answer = ""
    # qa = format_qa(question, answer)

    return {
        "images": img_paths,
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    }


img_dict = {
    'front camera': "<FRONT VIEW>:\n<image>\n",
    'front right camera': "<FRONT RIGHT VIEW>:\n<image>\n",
    'front left camera': "<FRONT LEFT VIEW>:\n<image>\n",
    'back camera': "<BACK VIEW>:\n<image>\n",
    'back left camera': "<BACK LEFT VIEW>:\n<image>\n",
    'back right camera': "<BACK RIGHT VIEW>:\n<image>\n"
}


def get_img_description_qa(img_paths,
                           img_type):

    # question = "Suppose you are driving, and I'm providing you with the image " \
    #            f"captured by the car's {img_type}, generate a description of the driving scene " \
    #            "which includes the key factors for driving planning, including the positions " \
    #            "and movements of vehicles and pedestrians; prevailing weather conditions; " \
    #            "time of day, distinguishing between daylight and nighttime; road conditions, " \
    #            "indicating smooth surfaces or the presence of obstacles; and the status of traffic lights " \
    #            "which influence your decision making, specifying whether they are red or green. " \
    #            "The description should be concise, providing an accurate understanding " \
    #            "of the driving environment to facilitate informed decision-making."
    question = f"{img_dict[img_type]}Suppose you are driving, and I'm providing you with the image captured by the car's {img_type}, generate a description of the driving scene which includes the key factors for driving planning, including the positions and movements of vehicles and pedestrians; prevailing weather conditions; time of day, distinguishing between daylight and nighttime; road conditions, indicating smooth surfaces or the presence of obstacles; and the status of traffic lights which influence your decision making, specifying whether they are red or green. The description should be concise, providing an accurate understanding of the driving environment to facilitate informed decision-making."
    # args = type('Args', (), {
    #     "query": question,
    #     "img_file": img_path,
    # })()

    # answer = eval_llava_34b_wo_init(args, tokenizer, model, image_processor)
    answer = ""
    # qa = format_qa(question, answer)

    return {
        "images": img_paths,
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    }


def get_plan_qa(info, img_paths):

    if np.any(info['gt_ego_fut_masks'] == 0):
        return None

    ego_cur_vel = info['gt_ego_lcf_feat'][7]
    ego_navi_cmd = info['ego_navi_cmd']

    # question = f"Your current speed is {int(ego_cur_vel)} m/s, " \
    #            f"the navigation command is '{ego_navi_cmd}', " \
    #             "based on the understanding of the driving scene and the navigation information, " \
    #             "what is your plan for the next three seconds? " \
    #             "Please answer your SPEED plan and your PATH plan. " \
    #             "SPEED includes KEEP, ACCELERATE and DECELERATE, and STOP, " \
    #             "PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. " \
    #             "For example, a correct answer format is like 'KEEP, LEFT_CHANGE'."
    question = f"<FRONT VIEW>:\n<image>\n<FRONT RIGHT VIEW>:\n<image>\n<FRONT LEFT VIEW>:\n<image>\n<BACK VIEW>:\n<image>\n<BACK LEFT VIEW>:\n<image>\n<BACK RIGHT VIEW>:\n<image>\nYour current speed is {int(ego_cur_vel)} m/s, the navigation command is '{ego_navi_cmd}', based on the understanding of the driving scene and the navigation information, what is your plan for the next three seconds? Please answer your SPEED plan and your PATH plan. SPEED includes KEEP, ACCELERATE and DECELERATE, and STOP, PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. For example, a correct answer format is like '<SPEED PATH PLAN>KEEP, LEFT_CHANGE'</SPEED PATH PLAN>."

    ego_fut_traj = np.cumsum(info['gt_ego_fut_trajs'], axis=-2)
    ego_pedal_status = get_obj_acc_or_dec(ego_fut_traj)
    ego_speed_plan = pedal_status[ego_pedal_status]

    ego_path_plan = get_obj_turn_or_lane_change(ego_fut_traj)
    ego_path_plan = path_status[ego_path_plan]

    answer = ego_speed_plan + ', ' + ego_path_plan + '\n'
    answer = f"<SPEED PATH PLAN>{answer}</SPEED PATH PLAN>"
    # qa = format_qa(question, answer)

    return {
        "images": img_paths,
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    }


def get_plan_explaination_qa(info, img_paths):

    cf_img_name = info['images'][0]

    ego_cur_vel = info['gt_ego_lcf_feat'][7]
    ego_navi_cmd = info['ego_navi_cmd']

    ego_fut_traj = np.cumsum(info['gt_ego_fut_trajs'], axis=-2)
    ego_pedal_status = get_obj_acc_or_dec(ego_fut_traj)
    ego_speed_plan = pedal_status[ego_pedal_status]
    ego_path_plan = get_obj_turn_or_lane_change(ego_fut_traj)
    ego_path_plan = path_status[ego_path_plan]

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

    if ego_speed_plan == 'STOP':
        decision = pedal_decision[ego_speed_plan]
    else:
        decision = pedal_decision[ego_speed_plan] + \
            ' and ' + path_decision[ego_path_plan]

    # question = "You are driving, " \
    #            f"your current speed is {int(ego_cur_vel)} m/s, " \
    #            f"and the navigation command is '{ego_navi_cmd}', " \
    #            "your driving decision for the next three seconds is to " \
    #            f"{decision}. " \
    #            "Based on the provided image of the driving environment, " \
    #            "explain the most likely reason for this decision in one or two concise sentence." \
    question = f"<FRONT VIEW>:\n<image>\n<FRONT RIGHT VIEW>:\n<image>\n<FRONT LEFT VIEW>:\n<image>\n<BACK VIEW>:\n<image>\n<BACK LEFT VIEW>:\n<image>\n<BACK RIGHT VIEW>:\n<image>\nYou are driving, your current speed is {int(ego_cur_vel)} m/s, and the navigation command is '{ego_navi_cmd}', your driving decision for the next three seconds is to {decision}. Based on the provided image of the driving environment, explain the most likely reason for this decision in one or two concise sentence."
    args = type('Args', (), {
        "query": question,
        "img_file": cf_img_name,
    })()

    # answer = eval_llava_34b_wo_init(args, tokenizer, model, image_processor)
    answer = ""
    # qa = format_qa(question, answer)

    return {
        "images": img_paths,
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    }


def get_waypoint_prediction_qa(info, img_paths):
    """
    Generate waypoint prediction QA based on ego vehicle's historical trajectory
    """
    if np.any(info['gt_ego_fut_masks'] == 0):
        return None

    xs, ys, vxs, vys = [], [], [], []
    # Get historical trajectory data (last 3 seconds at 0.5s intervals)
    ego_his_trajs = info['gt_ego_his_trajs']  # shape: (his_ts, 2)
    # [vx, vy, ax, ay, w, length, width, vel, steer]
    ego_lcf_feat = info['gt_ego_lcf_feat']
    for i in range(len(ego_his_trajs)):
        y, x = ego_his_trajs[i]
        y = -y
        vx = x / 0.5
        vy = y / 0.5
        vxs.append(vx)
        vys.append(vy)
    # Format historical trajectory data
    his_trajs_str = ""
    # 1. 沿着目标轴翻转数组

    reversed_data = np.flip(ego_his_trajs, axis=-2)
    reversed_cumsum = np.cumsum(reversed_data, axis=-2)
    ego_his_trajs = np.flip(reversed_cumsum, axis=-2)
    for i in range(len(ego_his_trajs)):
        y, x = ego_his_trajs[i]
        x, y = -x, y
        xs.append(x)
        ys.append(y)
    xs.append(0.0)
    ys.append(0.0)
    ego_fut_trajs = info['gt_ego_fut_trajs']
    vxs.append(ego_fut_trajs[0, 1] / 0.5)
    vys.append(-ego_fut_trajs[0, 0] / 0.5)
    vxs.append(ego_fut_trajs[1, 1] / 0.5)
    vys.append(-ego_fut_trajs[1, 0] / 0.5)

    for i in range(len(ego_his_trajs) + 1):
        t_offset = -0.5 * len(ego_his_trajs) + i * 0.5
        x = xs[i]
        y = ys[i]
        vx = vxs[i]
        vy = vys[i]
        ax = (vxs[i+1] - vxs[i]) / 0.5
        ay = (vys[i+1] - vys[i]) / 0.5

        his_trajs_str += f"(t{t_offset:+.1f}s) [{x:.2f}, {y:.2f}], Acceleration: X {ax:.2f}, Y {ay:.2f} m/s^2, Velocity: X {vx:.2f}, Y {vy:.2f} m/s"
        if i < len(ego_his_trajs):
            his_trajs_str += ", "

    # Get future trajectory for ground truth answer
    ego_fut_trajs = np.cumsum(ego_fut_trajs, axis=-2)  # shape: (fut_ts, 2)
    # Format future trajectory as answer
    fut_trajs_str = ""
    for i in range(len(ego_fut_trajs)):
        y, x = ego_fut_trajs[i]
        y = -y
        fut_trajs_str += f"[{x:.2f}, {y:.2f}]"
        if i < len(ego_fut_trajs) - 1:
            fut_trajs_str += ", "

    question = f"You are an autonomous driving agent. You have access to multi-view camera images of a vehicle: (1) front view (which you should focus on with the most attention) <image>, (2) front right view <image>, and (3) front left view <image>. Your task is to do your best to predict future waypoints for the vehicle over the next 10 timesteps, given the vehicle's intent inferred from the images. Provided are the previous ego vehicle status recorded over the last 3.0 seconds (at 0.5-second intervals). This includes the x and y coordinates of the ego vehicle. Positive x means forward direction while positive y means leftwards. The data is presented in the format [x, y]: {his_trajs_str}\n"

    answer = f"<PLANNING>Predicted future movement details for the next {len(ego_fut_trajs)//2} seconds (sampled at 0.5-second intervals), including BEV location in x and y directions (in meters). Positive x means forward direction while positive y means leftwards. The output is formatted as [x, y]: {fut_trajs_str}</PLANNING>"

    # qa = format_qa(question, answer)
    return {
        "images": img_paths,
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    }


# def format_qa(question, answer):

#     image_prompt = "<FRONT VIEW>:\n<image>\n" \
#                    "<FRONT LEFT VIEW>:\n<image>\n" \
#                    "<FRONT RIGHT VIEW>:\n<image>\n" \
#                    "<BACK LEFT VIEW>:\n<image>\n" \
#                    "<BACK RIGHT VIEW>:\n<image>\n" \
#                    "<BACK VIEW>:\n<image>\n"

#     question_dict, answer_dict = {}, {}
#     question_dict["from"] = "human"
#     question_dict["value"] = image_prompt + question
#     answer_dict["from"] = "gpt"
#     answer_dict["value"] = answer

#     return [question_dict, answer_dict]

def format_qa(question, answer):

    question_dict, answer_dict = {}, {}
    question_dict["role"] = "user"
    question_dict["content"] = question
    answer_dict["role"] = "assistant"
    answer_dict["content"] = answer

    return [question_dict, answer_dict]


def generate_drive_qa(info, output_dir, version='v1.0-trainval'):
    os.makedirs(output_dir, exist_ok=True)
    # q1
    img_paths = info['images']
    vru_qa = get_vru_qa(info, img_paths=[img_paths[0]])
    if vru_qa is not None:
        with open(os.path.join(output_dir, f"{version}_q1.json"), "a") as f:
            f.write(json.dumps(vru_qa) + "\n")

    img_types = ["front", "front right", "front left",
                 "back", "back left", "back right"]
    dis_thresh_imgs = [40.0, 20.0, 20.0, 20.0, 20.0, 20.0]
    for type, thresh in zip(img_types, dis_thresh_imgs):
        # q2
        mot_pred_qa = get_mot_pred_qa(
            info, img_paths=[img_paths[0]], img_type=type, dis_thresh=thresh)
        if mot_pred_qa is not None:
            with open(os.path.join(output_dir, f"{version}_q2.json"), "a") as f:
                f.write(json.dumps(mot_pred_qa) + "\n")

    # q4
    traffic_light_qa = get_traffic_light_qa(img_paths=[img_paths[0]])
    if traffic_light_qa is not None:
        with open(os.path.join(output_dir, f"{version}_q4.json"), "a") as f:
            f.write(json.dumps(traffic_light_qa) + "\n")

    cams = [
        'front camera',
        'front right camera',
        'front left camera',
        'back camera',
        'back left camera',
        'back right camera']

    for i, cam in enumerate(cams):
        # q5
        description_qa = get_img_description_qa(
            img_paths=[img_paths[i]], img_type=cam)
        if description_qa is not None:
            with open(os.path.join(output_dir, f"{version}_q5.json"), "a") as f:
                f.write(json.dumps(description_qa) + "\n")
    # q6
    plan_qa = get_plan_qa(info, img_paths=img_paths)
    if plan_qa is not None:
        with open(os.path.join(output_dir, f"{version}_q6.json"), "a") as f:
            f.write(json.dumps(plan_qa) + "\n")

    # q7 - waypoint prediction
    waypoint_qa = get_waypoint_prediction_qa(info, img_paths=img_paths[:3])
    if waypoint_qa is not None:
        with open(os.path.join(output_dir, f"{version}_q7.json"), "a") as f:
            f.write(json.dumps(waypoint_qa) + "\n")

    # q3
    plan_explaination_qa = get_plan_explaination_qa(info, img_paths=img_paths)
    if plan_explaination_qa is not None:
        with open(os.path.join(output_dir, f"{version}_q3.json"), "a") as f:
            f.write(json.dumps(plan_explaination_qa) + "\n")


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('--dataset', type=str,
                    default='nuscenes', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data_qa_generate/data_engine/data_storage/external_datasets/nuscenes',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./data_qa_generate/data_engine/data_storage/external_datasets/nuscenes',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./output',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='nuscenes')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
