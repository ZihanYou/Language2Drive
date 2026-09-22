"""
This script generates the dreamer labels for the SimLingo paper.
Known problems:
 - lane changes on highways when there is a barriere between oncoming and ego lane
 - sometimes too early towards static object for crashes -> cuts turns and if there is a barriere it doesnt work
"""

import glob
import gzip
import json
import os
import random
import time
from pathlib import Path

import carla
import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import simlingo_training.utils.transfuser_utils as t_u
from dataset_generation.dreamer_data.dreamer_utils import *
from dataset_generation.dreamer_data.kinematic_bicycle_model import KinematicBicycleModel
from dataset_generation.language_labels.utils import *
from dataset_generation.dreamer_data.dreamer_instructions import get_info
from team_code.config import GlobalConfig
from team_code.lateral_controller import LateralPIDController
from team_code.longitudinal_controller import LongitudinalLinearRegressionController

random.seed(42)
np.random.seed(42)

class CarlaAlternativeCreator():

    # Sampling parameters
    random_subset_count = -1 # -1 for all samples
    sample_uniform_interval = 1
    filter_routes_by_result = True
    
    # Visualization and saving options
    save_viz = False                # Should examples be saved
    viz_for_video = False
    save_samples = True             # Save labels
    overwrite = False               # Overwrite existing files
    save_instructions = True
 
    # Dataset and path settings
    base_folder = 'database'
    dataset_name = 'simlingo'
    data_directory = f'{base_folder}/{dataset_name}/data'
    viz_save_path = 'viz/dreamer' if not viz_for_video else 'viz/dreamer_video'
    Path(viz_save_path).mkdir(parents=True, exist_ok=True)
    save_folder_name = 'dreamer' if not viz_for_video else 'dreamer_video'

    # Region of interest (ROI) for image projection
    original_image_size = [1024, 512]
    target_image_size = [1024, 384]
    original_fov = 110
    MIN_X = original_image_size[0] // 2 - target_image_size[0] // 2
    MAX_X = original_image_size[0] // 2 + target_image_size[0] // 2
    MIN_Y = 0
    MAX_Y = target_image_size[1]
    CAMERA_MATRIX = build_projection_matrix(original_image_size[0],
                                                    original_image_size[1],
                                                    original_fov)
    # config
    FUTURE_LEN = 10
    carla_frame_rate = 20 # 20 frames per second get simulated
    dataset_frame_rate = 4 # 4 frames per second get saved

    default_forecast_length = 2.5
    num_future_frames_carla_fps = int(carla_frame_rate * default_forecast_length)
        
    bicycle_model = KinematicBicycleModel(carla_frame_rate)
    config = GlobalConfig()
    _turn_controller = LateralPIDController(config)
    _longitudinal_controller = LongitudinalLinearRegressionController(config)

    def __init__(self):
        # all the paths to the boxes in the data
        self.data_boxes_paths = glob.glob(os.path.join(self.data_directory, 'simlingo/*/*/*/*/boxes/*.json.gz'))
        print(f"Number of data boxes: {len(self.data_boxes_paths)}")

        # Randomly sample a subset of data (if random_subset_count > 0)
        if self.random_subset_count > 0:
            random.shuffle(self.data_boxes_paths)
            self.data_boxes_paths = self.data_boxes_paths[:self.random_subset_count]

        self.data_boxes_paths = list(sorted(self.data_boxes_paths))
        if self.sample_uniform_interval > 1:
            self.data_boxes_paths = self.data_boxes_paths[::self.sample_uniform_interval]

    def process_data(self, path_id):
        path_boxes = self.data_boxes_paths[path_id]

        path_save = path_boxes.replace('/data/', f"/{self.save_folder_name}/").replace('boxes',  self.save_folder_name)
        Path(path_save).parent.mkdir(exist_ok=True, parents=True)
        if not self.overwrite and os.access(path_save, os.F_OK):
            return

        save_list = []
        path_measurement = path_boxes.replace('boxes', 'measurements')
        path_rgb_image = path_boxes.replace('boxes', 'rgb').replace('.json.gz', '.jpg')

        # Skip frames if RGB image does not exist
        if not os.access(path_rgb_image, os.F_OK) or not os.access(path_boxes, os.F_OK) or not os.access(path_measurement, os.F_OK):
            return
        
        # Read results file
        if self.filter_routes_by_result:
            results_file = path_boxes.split('boxes')[0] + 'results.json.gz'
            try:
                with gzip.open(results_file, 'rb') as f:
                    route_results = json.loads(f.read().decode('utf-8'))
            except Exception as e:
                print(f"Error reading {results_file}: {e}")
                return

            other_infractions_than_min_speed_infractions_exist = False
            for key in route_results['infractions'].keys():
                if len(route_results['infractions'][key]) > 0 and key != 'min_speed_infractions':
                    other_infractions_than_min_speed_infractions_exist = True
                    break

            # Skip data where the expert did not achieve perfect driving score
            if other_infractions_than_min_speed_infractions_exist and \
                        (route_results['scores']['score_composed'] < 98.0 or
                        route_results['scores']['score_route'] < 98.0):
                return
            elif not other_infractions_than_min_speed_infractions_exist and \
                                (route_results['scores']['score_route'] < 98.0):
                return
            
        # Read data and measurements files
        with gzip.open(path_boxes, 'rb') as f:
            file_content = f.read()
            current_boxes = json.loads(file_content.decode('utf-8'))

        with gzip.open(path_measurement, 'rb') as f:
            file_content = f.read()
            current_measurements = json.loads(file_content.decode('utf-8'))

        time_stamp = path_boxes.split('/')[-1].split('_')[-1].replace('.json.gz', '')
        time_stamp_int = int(time_stamp)
        next_10_time_stamps = [str(time_stamp_int + i).zfill(4) for i in range(1, self.FUTURE_LEN)] # 4 digits string

        future_boxes_paths = [os.path.join(path_boxes.replace(time_stamp, i)) for i in next_10_time_stamps]
        future_measurement_paths = [i.replace('boxes', 'measurements') for i in future_boxes_paths]

        future_boxes = []
        future_measurements = []
        changed_route = False
        for path_box, path_measure in zip(future_boxes_paths, future_measurement_paths):
            if os.path.exists(path_box) and os.path.exists(path_measure):
                with gzip.open(path_box, 'rb') as f:
                    file_content = f.read()
                    future_boxes.append(json.loads(file_content.decode('utf-8')))
                with gzip.open(path_measure, 'rb') as f:
                    file_content = f.read()
                    future_measurements.append(json.loads(file_content.decode('utf-8')))
                    if future_measurements[-1]['changed_route'] or np.any(np.asarray(future_measurements[-1]['route']) != np.asarray(future_measurements[-1]['route_original'])):
                        changed_route = True
            else:
                return

        # load data
        target_point = current_measurements['target_point']
        target_point = [round(target_point[0], 2), round(target_point[1], 2)]
        target_point_next = current_measurements['target_point_next']
        target_point_next = [round(target_point_next[0], 2), round(target_point_next[1], 2)]

        target_speed = current_measurements['target_speed']
        current_speed = current_measurements['speed']

        command_map = {
            1: 'turn left at the next intersection',
            2: 'turn right at the next intersection',
            3: 'drive straight at the next intersection',
            4: 'follow the road',
            5: 'lane change to the left',
            6: 'lane change to the right',
        }
        command = current_measurements['command']
        command = command_map[command]
        command_next = current_measurements['next_command']
        command_next = command_map[command_next]

        ego_position = current_measurements['pos_global']
        ego_yaw = current_measurements['theta']
        
        steer = current_measurements['steer']
        throttle = current_measurements['throttle']
        brake = current_measurements['brake']
        
        route = current_measurements['route']
        route_global = np.asarray([t_u.conversion_2d(rout, ego_position, -ego_yaw) for rout in route])
        route_local = np.asarray([t_u.inverse_conversion_2d(rout, ego_position, ego_yaw) for rout in route_global])

        route_adjusted = np.array(current_measurements['route'])
        route_adjusted = equal_spacing_route(route_adjusted)

        route_original = current_measurements['route_original']
        route_original = equal_spacing_route(route_original)


        nearby_actors = [box for box in current_boxes if box['class'] == 'car'] # or box['class'] == 'walker']
        nearby_actors_by_id = {box['id']: box for box in nearby_actors}
        ids = [box['id'] for box in nearby_actors]

        nearby_walkers = [box for box in current_boxes if box['class'] == 'walker']
        ids_walkers = [box['id'] for box in nearby_walkers]

        walker_close = False
        for box in current_boxes:
            if box['class'] == 'walker' and box['distance'] < 10:
                walker_close = True
                break

        ids_all = ids + ids_walkers


        # load data for future frames, if not available repeat the last available frame
        # this is not ideal but works for good enough
        future_nearby_actors = []
        future_nearby_walkers = []
        future_nearby_actors_by_id = {id: [] for id in ids}
        future_nearby_actors_used_time_stamps_by_id = {id: [] for id in ids}
        future_nearby_walkers_by_id = {id: [] for id in ids_walkers}
        future_nearby_walkers_used_time_stamps_by_id = {id: [] for id in ids_walkers}

        for i, future in enumerate(future_boxes):
            tmp_actors = []
            tmp_walkers = []
            for id in ids_all:
                tmp_box = [box for box in future if 'id' in box and box['id'] == id]
                if tmp_box:
                    if tmp_box[0]['class'] == 'car':
                        tmp_actors.append(tmp_box[0])
                    elif tmp_box[0]['class'] == 'walker':
                        tmp_walkers.append(tmp_box[0])
                    else:
                        raise ValueError('Unknown class')
                else:
                    if id in future_nearby_actors_by_id:
                        # if actor is not present at a future frame, we assume reuse the last frame
                        if len(future_nearby_actors_by_id[id]) > 0:
                            tmp_actors.append(future_nearby_actors_by_id[id][-1])
                            future_nearby_actors_used_time_stamps_by_id[id].append(future_nearby_actors_used_time_stamps_by_id[id][-1])
                        else:
                            tmp_actors.append(nearby_actors_by_id[id])
                            future_nearby_actors_used_time_stamps_by_id[id].append(0)
                    # following is commented out because we dont want to keep the old position of pedestrians in case they despawn!
                    # elif id in future_nearby_walkers_by_id:
                    #     if len(future_nearby_walkers_by_id[id]) > 0:
                    #         tmp_walkers.append(future_nearby_walkers_by_id[id][-1])
                    #     else:
                    #         tmp_walkers.append(nearby_walkers_by_id[id])
                if id in future_nearby_actors_by_id:
                    future_nearby_actors_by_id[id].append(tmp_actors[-1])
                    if tmp_box:
                        future_nearby_actors_used_time_stamps_by_id[id].append(i)
                elif id in future_nearby_walkers_by_id and len(tmp_walkers) > 0:
                    future_nearby_walkers_by_id[id].append(tmp_walkers[-1])
                    future_nearby_walkers_used_time_stamps_by_id[id].append(i)

            if len(tmp_actors) > 0:
                future_nearby_actors.append(tmp_actors)
            if len(tmp_walkers) > 0:
                future_nearby_walkers.append(tmp_walkers)

        ego_actor = [box for box in current_boxes if box['class'] == 'ego_car'][0]
        ego_info = [box for box in current_boxes if box['class'] == 'ego_info'][0]

        next_egos = [box for all_future_actors in future_boxes for box in all_future_actors if box['class'] == 'ego_car']
        # add actions to the dict
        ego_actor["steer"] = steer
        ego_actor["throttle"] = throttle
        ego_actor["brake"] = brake
        ego_actor["id"] = 0

        for i, next_ego in enumerate(next_egos):
            next_ego["id"] = 0
            next_ego["steer"] = future_measurements[i]['steer']
            next_ego["throttle"] = future_measurements[i]['throttle']
            next_ego["brake"] = future_measurements[i]['brake']

        next_ego_by_id = {0: next_egos}
        all_ego_positions = [ego_position] + [future_measurements[i]['pos_global'] for i in range(len(next_egos))]
        all_ego_yaws = [ego_yaw] + [future_measurements[i]['theta'] for i in range(len(next_egos))]

        # we can either use the ground truth positions or the forecasted positions
        # ground truth is more acurate in turns but if the vehicle is out of sight in the future frames we need to use the last available one
        bbs_walkers = self.get_bbs(nearby_walkers, future_nearby_walkers_by_id, future_nearby_walkers_used_time_stamps_by_id, all_ego_positions, all_ego_yaws)
        bbs_other_actors = self.get_bbs(nearby_actors, future_nearby_actors_by_id, future_nearby_actors_used_time_stamps_by_id, all_ego_positions, all_ego_yaws)
        # # add walkers
        bbs_other_actors = {**bbs_other_actors, **bbs_walkers}
        ## we can also use forecasting instead of using the ground truth positions:
        # forecasts_other_actors = self.forecast_vehicles(nearby_actors, future_nearby_actors_by_id, use_gt_positions=False, ego_position=ego_position, ego_yaw=ego_yaw)
        ## for debugging of coordinate transformation and compare forecast and gt bounding boxes:
        # self.viz_forecastings_new(path_rgb_image, forecasts_other_actors, bbs_other_actors)


        
        # get ego forecast for the current route
        bbs_ego = self.get_bbs([ego_actor], next_ego_by_id, {0: list(range(len(next_egos)))}, all_ego_positions, all_ego_yaws)
        forecast_ego_org, gt_speeds = self.forecast_vehicles(ego_actor, next_ego_by_id, ego_position=ego_position, ego_yaw=ego_yaw, route=route_global, use_wps_speed_controller=True, return_gt_speeds=True)
        # self.viz_forecastings_new(path_rgb_image, forecast_ego_org, bbs_ego)

        forecast_ego_org_wps = [[f.location.x, f.location.y] for f in forecast_ego_org[0]]
        forecasts_ego_adjusted = []
        forecasts_ego_adjusted_route = []
        forecasts_ego_adjusted_allowed = []
        forecasts_ego_adjusted_mode = []
        forecasts_ego_adjusted_info = []
        save_keys = ['class', 'name', 'color_rgb', 'type_id', 'distance', 'id', 'position']
        
        # # ############################################################
        # # # Slower/ Faster / Target speed / Stop
        # # ############################################################
        # Target speed
        # at least 40% of cases should be in a reachable speed range
        # to avoid that the optimal speed is always maximal acceleration/ deceleration
        if random.random() < 0.6:
            random_target_speed = round(random.uniform(0, 35), 1)
        else:
            random_target_speed = round(random.uniform(ego_actor['speed']*0.6, ego_actor['speed']*1.4), 1)
        tmp_forecasts, final_speed = self.forecast_vehicles(ego_actor, next_ego_by_id, ego_position=ego_position, ego_yaw=ego_yaw, route=route_global, target_speed=random_target_speed, return_final_speed=True)
        forecasts_ego_adjusted.append(tmp_forecasts)
        forecasts_ego_adjusted_route.append('org')
        forecasts_ego_adjusted_allowed.append(True)
        forecasts_ego_adjusted_mode.append(f'target speed {random_target_speed}')
        forecasts_ego_adjusted_info.append({'allowed': True, 'mode': f'target_speed', 'target_speed': random_target_speed, 'final_speed': final_speed, 'current_speed': current_speed})

        # Stop 
        if random_target_speed > 0.01: # if random target speed is 0, we do not need to add additional stop mode
            stop_target_speed = 0
            tmp_forecasts, final_speed = self.forecast_vehicles(ego_actor, next_ego_by_id, ego_position=ego_position, ego_yaw=ego_yaw, route=route_global, target_speed=stop_target_speed, return_final_speed=True)
            forecasts_ego_adjusted.append(tmp_forecasts)
            forecasts_ego_adjusted_route.append('org')
            forecasts_ego_adjusted_allowed.append(True)
            forecasts_ego_adjusted_mode.append(f'stop')
            forecasts_ego_adjusted_info.append({'allowed': True, 'mode': f'stop', 'target_speed': stop_target_speed, 'final_speed': final_speed, 'current_speed': current_speed})

        # Faster, factor of original trajectory -> if stopped, this mode still stops
        faster_factor = random.uniform(1.1, 1.5)
        desired_speeds = faster_factor * gt_speeds
        tmp_forecasts, final_speed = self.forecast_vehicles(ego_actor, next_ego_by_id, ego_position=ego_position, ego_yaw=ego_yaw, route=route_global, speeds_to_follow=desired_speeds, return_final_speed=True)
        forecasts_ego_adjusted.append(tmp_forecasts)
        forecasts_ego_adjusted_route.append('org')
        forecasts_ego_adjusted_allowed.append(True)
        forecasts_ego_adjusted_mode.append('faster_factor')
        forecasts_ego_adjusted_info.append({'allowed': True, 'mode': 'faster_factor', 'final_speed': final_speed, 'factor': faster_factor, 'current_speed': current_speed})

        # Faster compared to current speed -> if stopped, this mode does not stop -> so you could say go faster on a red light (could be slower than the expert, e.g., if expert heavily accelerates)
        # get int between 0 and 3
        mode = random.randint(0, 2)
        if mode == 0:
            desired_throttle = random.uniform(0.5, 0.7)
        elif mode == 1:
            desired_throttle = random.uniform(0.7, 0.85)
        else:
            desired_throttle = random.uniform(0.85, 1.0)
        tmp_forecasts, final_speed = self.forecast_vehicles(ego_actor, next_ego_by_id, ego_position=ego_position, ego_yaw=ego_yaw, route=route_global, desired_throttle=desired_throttle, return_final_speed=True)
        forecasts_ego_adjusted.append(tmp_forecasts)
        forecasts_ego_adjusted_route.append('org')
        forecasts_ego_adjusted_allowed.append(True)
        forecasts_ego_adjusted_mode.append('faster')
        forecasts_ego_adjusted_info.append({'allowed': True, 'mode': 'faster', 'final_speed': final_speed, 'desired_throttle': desired_throttle, 'rate': mode, 'current_speed': current_speed})

        # 'Slower', factor of original trajectory -> if stopped, this mode still stops, this does not mean actually driving slower than current speed, only slower than the expert would drive
        slower_factor = random.uniform(0.3, 0.9)
        desired_speeds = slower_factor * gt_speeds
        tmp_forecasts, final_speed = self.forecast_vehicles(ego_actor, next_ego_by_id, ego_position=ego_position, ego_yaw=ego_yaw, route=route_global, speeds_to_follow=desired_speeds, return_final_speed=True)
        forecasts_ego_adjusted.append(tmp_forecasts)
        forecasts_ego_adjusted_route.append('org')
        forecasts_ego_adjusted_allowed.append(True)
        forecasts_ego_adjusted_mode.append('slower_factor')
        forecasts_ego_adjusted_info.append({'allowed': True, 'mode': 'slower_factor', 'final_speed': final_speed, 'factor': slower_factor, 'current_speed': current_speed})

        # 'Slower', compared to current speed (could be faster than the expert, e.g., if expert heavily brakes)
        # get int between 0 and 3
        mode = random.randint(2, 2)
        if mode == 0:
            brake_probability = random.uniform(0.1, 0.2)
        elif mode == 1:
            brake_probability = random.uniform(0.2, 0.3)
        else:
            if ego_actor['speed'] < 12:
                brake_probability = random.uniform(0.3, 0.35)
            else:
                brake_probability = random.uniform(0.3, 0.6)
        tmp_forecasts, final_speed = self.forecast_vehicles(ego_actor, next_ego_by_id, ego_position=ego_position, ego_yaw=ego_yaw, route=route_global, brake_probability=brake_probability, return_final_speed=True)
        forecasts_ego_adjusted.append(tmp_forecasts)
        forecasts_ego_adjusted_route.append('org')
        forecasts_ego_adjusted_allowed.append(True)
        forecasts_ego_adjusted_mode.append('slower')
        forecasts_ego_adjusted_info.append({'allowed': True, 'mode': 'slower', 'final_speed': final_speed, 'brake_probability': brake_probability, 'rate': mode, 'current_speed': current_speed})
        
        


        # ############################################################
        # # CRASHES
        # ############################################################
        crash_actors_id = []
        static_actors = []
        actor_considered_positions = None # to avoid that we consider the same actor multiple times
        for box in current_boxes:
            if 'position' not in box or ('position' in box and box['position'][0] < 3):
                continue
            visible_in_image = is_vehicle_visible_in_image(box, self.MIN_X, self.MAX_X, self.MIN_Y, self.MAX_Y, self.CAMERA_MATRIX)
            if not visible_in_image:
                continue

            pos = box['position']
            min_dist_to_route = np.min(np.linalg.norm(route_local - pos[:2], axis=1))
            argmin_dist_to_route = np.argmin(np.linalg.norm(route_local - pos[:2], axis=1))
            if min_dist_to_route < 7: # and argmin_dist_to_route < len(route_local) - 5:
                # we use a higher distance threshold since oncoming vehciles might be further away and come closer
                # and we check later if it is reachable in the given time
                if box['class'] == 'car' or box['class'] == 'walker' and box['distance'] < 40: 
                    if actor_considered_positions is None or np.min(np.linalg.norm(actor_considered_positions[:,:2] - pos[:2], axis=1)) > 1:
                        crash_actors_id.append(box['id'])
                        actor_considered_positions = np.array([pos]) if actor_considered_positions is None else np.append(actor_considered_positions, np.array([pos]), axis=0)

                elif box['class'] in ['static', 'landmark'] or '_vqa' in box['class'] and box['distance'] < 20: 
                    if 'type_id' in box and 'dirtdebris' in box['type_id']:
                        continue
                    if actor_considered_positions is None or np.min(np.linalg.norm(actor_considered_positions[:,:2] - pos[:2], axis=1)) > 1:
                        static_actors.append(box)
                        actor_considered_positions = np.array([pos]) if actor_considered_positions is None else np.append(actor_considered_positions, np.array([pos]), axis=0)
        
        if len(static_actors) > 0:
            static_actors_ids = [f'st_{i}' for i in range(len(static_actors))]
            static_actor_by_id = {st_id: static_actors[i] for i, st_id in enumerate(static_actors_ids)}
            crash_actors_id = crash_actors_id + static_actors_ids

        crash_positions = []
        crash_target_speeds = []
        crash_type_str = []
        crash_objects = []
        for actor_id in crash_actors_id:
            boxes, distances, time_stamps, positions = [], [], [], []
            if 'st_' in str(actor_id):
                actor = static_actor_by_id[actor_id]
                # duplicate actor to have list of actors for each time stamp
                actor = [actor for _ in range(self.FUTURE_LEN)]
                # only consider actors that are in front of the ego vehicle and therefore visible in the front frame
                if actor[0]['position'][0] > 3:
                    for i in range(2, self.FUTURE_LEN):
                        positions.append(np.array(actor[i]['position']))
                        boxes.append(actor[i])
                        time_stamps.append(i)
                        distances.append(np.linalg.norm(actor[i]['position'][:2]))
                else:
                    continue
            else:
                actor = future_nearby_actors_by_id[actor_id] if actor_id in future_nearby_actors_by_id else future_nearby_walkers_by_id[actor_id]
                if len(actor) == 0:
                    continue
                actor_future_bbs = bbs_other_actors[actor[0]['id']]

                for i, box in enumerate(actor):
                    if i < 2: # skip the first 2 frames as it is not enough reaction time for the ego vehicle
                        continue
                    try:
                        position = actor_future_bbs[i].location
                    except:
                        return
                    position = np.array([position.x, position.y, position.z])
                    distance = np.linalg.norm(position[:2])
                    if position[0] > 3:
                        positions.append(position)
                        boxes.append(box)
                        time_stamps.append(i+1) # +1 becuase we only load future frames so current frame is not included
                        distances.append(distance)

            if len(time_stamps) > 0:
                    tried_idx = []
                    possible_idx = list(range(len(boxes)))
                    while len(tried_idx) < len(boxes):
                        # get random index and pop it from the list
                        random_idx = random.choice(possible_idx)
                        tried_idx.append(random_idx)
                        possible_idx.remove(random_idx)
                        
                        box = boxes[random_idx]
                        time_stamp = time_stamps[random_idx]
                        actor_time_id = time_stamp -1 # -1 because we save time_stamp as actor_id +1
                        distance = distances[random_idx]
                        pos = positions[random_idx]

                        # a bit hacky way to take the size of the vehicles into accout 
                        # -> this does not take into account the angle of the crash
                        if 'extent' not in actor[actor_time_id]:
                            distance = distance
                        elif abs(actor[actor_time_id]['position'][1]) < 1: # -1 because we save time_stamp as actor_id +1
                            distance = max(0, distance - actor[actor_time_id]['extent'][0] - current_boxes[0]['extent'][0])
                        else:
                            distance = max(0, distance - actor[actor_time_id]['extent'][1] - current_boxes[0]['extent'][1])
                        
                        # get the min and max position the ego vehicle can reach in the given time stamp
                        pos_min, pos_max = self.get_min_max_pos_given_speed_and_deltaT([0.,0.], current_speed, time_stamp)

                        if distance < pos_min[0] or distance > pos_max[0]:
                            continue

                        # get target speed based on current speed and distance and time stamp
                        target_speed = (distance / (time_stamp * (self.dataset_frame_rate / self.carla_frame_rate))) + 0.5
                        
                        crash_positions.append(pos)
                        crash_target_speeds.append(target_speed)
                        if 'type_id' in box:
                            if box['type_id'] == 'static.prop.mesh':
                                crash_type_str.append('mesh:' + box['mesh_path'])
                            else:
                                crash_type_str.append(box['type_id'])
                        elif 'name' in box:
                            crash_type_str.append(box['name'])
                        else:
                            crash_type_str.append(box['class'])
                        crash_objects.append({key: boxes[random_idx][key] for key in save_keys if key in boxes[random_idx]})
                        break
                            
        # change route to cross the crash positions
        crash_routes_global = []
        for pos in crash_positions:
            # find closest point in original route to pos
            closest_point_after = np.argmin(np.linalg.norm(route_local - pos[:2], axis=1)) + 7
            closest_point_before = np.argmin(np.linalg.norm(route_local - pos[:2], axis=1)) - 7
            if closest_point_before < 1:
                # check if ego is on route or not:
                org_route_before = route_local[:1]
                if abs(route_local[0][1]) > 2:
                    org_route_before[0][1] = 0.0
            else:
                org_route_before = route_local[:closest_point_before]
            
            if closest_point_after >= len(route_local):
                closest_point_after = -1
                org_route_after = route_local[closest_point_after:]
            else:
                org_route_after = route_local[closest_point_after:]
            
            # interpolate between org_route and crash_pos and get a point every 1m
            start_end = np.array([org_route_before[-1], np.array(pos[:2])])
            dist = np.linalg.norm(start_end[1] - start_end[0])
            num_points = max(int(dist), 1)
            org_to_crash_route = np.array([start_end[0] + i * (start_end[1] - start_end[0]) / num_points for i in range(num_points)])
            org_to_crash_route = np.concatenate([org_to_crash_route, np.array([pos[:2]])])

            if org_route_after is not None:
                start_end = np.array([np.array(pos[:2]), org_route_after[0]])
                dist = np.linalg.norm(start_end[1] - start_end[0])
                num_points = max(int(dist), 1)
                crash_to_org_route = np.array([start_end[0] + i * (start_end[1] - start_end[0]) / num_points for i in range(num_points)])

                # concat the org route with the crash route and back to the org route
                crash_route = np.concatenate([org_route_before, org_to_crash_route, crash_to_org_route, org_route_after])
            else:
                crash_route = np.concatenate([org_route_before, org_to_crash_route])

            # make sure that point are one meter apart and remove duplicate points
            crash_route = equal_spacing_route(crash_route)

            crash_route_global = np.array([t_u.conversion_2d(rout, ego_position, -ego_yaw) for rout in crash_route])
            crash_routes_global.append(crash_route_global)
        
        # get forecast for the crash routes
        for i, crash_route in enumerate(crash_routes_global):
            crash_bounding_boxes = self.forecast_vehicles(ego_actor, next_ego_by_id, ego_position=ego_position, ego_yaw=ego_yaw, route=crash_route, target_speed=crash_target_speeds[i])
            forecasts_ego_adjusted.append(crash_bounding_boxes)
            crash_route_local = np.asarray([t_u.inverse_conversion_2d(rout, ego_position, ego_yaw) for rout in crash_route])
            forecasts_ego_adjusted_route.append(np.round(crash_route_local, 2).tolist())
            forecasts_ego_adjusted_allowed.append(True)
            if 'mesh' in crash_type_str[i]:
                forecasts_ego_adjusted_mode.append(f'crash_{crash_type_str[i].split("/")[-1]}')
            else:
                forecasts_ego_adjusted_mode.append(f'crash_{crash_type_str[i]}')
            info = {
                'allowed': True,
                'mode': 'crash',
                'target_speed': round(crash_target_speeds[i], 3),
                'crash_position': np.round(crash_positions[i], 3).tolist(),
                'type': crash_type_str[i],
            }
            info.update(crash_objects[i])
            forecasts_ego_adjusted_info.append(info)
            

        # ############################################################
        # # LANE CHANGES
        # ############################################################
        
        # if ego is in parking lane or current command is lane change skip
        if not (changed_route or ego_info['lane_type_str'] == 'Parking' or 'lane change' in command or ego_info['is_in_junction'] or (ego_info['distance_to_junction'] is not None and ego_info['distance_to_junction'] < 10)):
            lane_change_options_same_direction = [lane_num - ego_info['ego_lane_number'] for lane_num in range(ego_info['num_lanes_same_direction'])]
            lane_change_options_directions_str = [f'{abs(lane_num)} {"left" if lane_num < 0 else "right"}' for lane_num in lane_change_options_same_direction]
            lane_change_options_types_str = ['driving' if lane_num != 0 else ego_info['lane_type_str'] for lane_num in lane_change_options_same_direction]
            if ego_info['parking_right']:
                lane_change_options_same_direction.append(max(lane_change_options_same_direction) + 1)
                lane_change_options_directions_str.append(f'{abs(lane_change_options_same_direction[-1])} right')
                lane_change_options_types_str.append('parking')
            if ego_info['sidewalk_right']:
                lane_change_options_same_direction.append(max(lane_change_options_same_direction) + 1)
                lane_change_options_directions_str.append(f'{abs(lane_change_options_same_direction[-1])} right')
                lane_change_options_types_str.append('sidewalk')
            
            lane_change_options_opposite_direction = []
            if ego_info['num_lanes_opposite_direction'] > 0:
                lane_change_options_opposite_direction = [-1*(lane_num + 1 + ego_info['ego_lane_number']) for lane_num in range(ego_info['num_lanes_opposite_direction'])]
                lane_change_options_directions_str.extend([f'{abs(lane_num)} {"left" if lane_num < 0 else "right"}' for lane_num in lane_change_options_opposite_direction])
                lane_change_options_types_str.extend(['driving opposite' if lane_num != 0 else ego_info['lane_type_str'] for lane_num in lane_change_options_opposite_direction])
                if ego_info['parking_left']:
                    lane_change_options_opposite_direction.append(min(lane_change_options_opposite_direction) - 1)
                    lane_change_options_directions_str.append(f'{abs(lane_change_options_opposite_direction[-1])} left')
                    lane_change_options_types_str.append('parking')
                if ego_info['sidewalk_left']:
                    lane_change_options_opposite_direction.append(min(lane_change_options_opposite_direction) - 1)
                    lane_change_options_directions_str.append(f'{abs(lane_change_options_opposite_direction[-1])} left')
                    lane_change_options_types_str.append('sidewalk')
                
            lane_change_options_directions = lane_change_options_same_direction + lane_change_options_opposite_direction
            left_lanes_info = ego_info['left_lanes']
            # remove type shoulder and add width to the width of the next entry
            remove_ids = []
            for i in range(len(left_lanes_info)):
                if left_lanes_info[i]['type:'] == 'Shoulder':
                    if i + 1 < len(left_lanes_info):
                        left_lanes_info[i + 1]['width'] += left_lanes_info[i]['width']
                    remove_ids.append(i)
            for index in sorted(remove_ids, reverse=True):
                del left_lanes_info[index]
            
            right_lanes_info = ego_info['right_lanes']
            # remove type shoulder and add width to the width of the next entry
            remove_ids = []
            for i in range(len(right_lanes_info)):
                if right_lanes_info[i]['type:'] == 'Shoulder':
                    if i + 1 < len(right_lanes_info):
                        right_lanes_info[i + 1]['width'] += right_lanes_info[i]['width']
                    remove_ids.append(i)
            for index in sorted(remove_ids, reverse=True):
                del right_lanes_info[index]


            for lane_change_options_direction, lane_change_options_direction_str, lane_change_options_type_str in zip(lane_change_options_directions, lane_change_options_directions_str, lane_change_options_types_str):
                if lane_change_options_direction == 0:
                    continue
                route_np = np.array(route)
                lane_width = 3.6
                max_dist_with_cur_speed = (current_speed * 2) - 1 #speed in m/s we predict wps for roughly 2 seconds
                start_lane_change = random.randint(0, max(int(max_dist_with_cur_speed/2), 5))
                transition = random.randint(int(current_speed/2), max(int(max_dist_with_cur_speed-start_lane_change), 5))
                transition = min(transition, len(route_np) - (start_lane_change+1) - 1)

                try:
                    if lane_change_options_direction < 0:
                        # sum up ego_info['left_lanes'][:abs(lane_change_options_direction)-1]['width']
                        lane_width = -1 * sum([left_lanes_info[i]['width'] for i in range(abs(lane_change_options_direction))])
                    else:
                        lane_width = sum([right_lanes_info[i]['width'] for i in range(abs(lane_change_options_direction))])
                except IndexError:
                    print('IndexError, skip lane change')
                    # breakpoint()
                    continue
                lane_change_in_transition_amount_meters = (start_lane_change, transition, lane_width)
                
                route_lc = route_np.copy()
                try:
                    route_lc_shifted = self.calculate_shifted_trajectory(route_lc, lane_change_in_transition_amount_meters)
                except IndexError:
                    print('IndexError in calculate_shifted_trajectory, skip lane change')
                    # breakpoint()
                    continue
                
                route_global_lc = np.asarray([t_u.conversion_2d(rout, ego_position, -ego_yaw) for rout in route_lc_shifted])
                route_local_lc = np.asarray([t_u.inverse_conversion_2d(rout, ego_position, ego_yaw) for rout in route_global_lc])

                forecasts_ego_adjusted.append(self.forecast_vehicles(ego_actor, next_ego_by_id, ego_position=ego_position, ego_yaw=ego_yaw, route=route_global_lc))
                forecasts_ego_adjusted_route.append(np.round(route_local_lc, 2).tolist())
                if lane_change_options_direction < 0:
                    if ego_info['lane_change'] in [0, 1]:
                        forecasts_ego_adjusted_allowed.append(False)
                    elif ego_info['lane_change'] in [2, 3]:
                        forecasts_ego_adjusted_allowed.append(True)
                else:
                    if ego_info['lane_change'] in [0, 2]:
                        forecasts_ego_adjusted_allowed.append(False)
                    elif ego_info['lane_change'] in [1, 3]:
                        forecasts_ego_adjusted_allowed.append(True)
                if lane_change_options_direction in lane_change_options_opposite_direction \
                    or lane_change_options_type_str == 'sidewalk':
                    forecasts_ego_adjusted_allowed[-1] = False
                forecasts_ego_adjusted_mode.append(f'lane change {lane_change_options_direction_str} {lane_change_options_type_str}')
                forecasts_ego_adjusted_info.append({'allowed': forecasts_ego_adjusted_allowed[-1], 'mode': 'lane_change', 'lane_change_direction': lane_change_options_direction_str, 'lane_change_type': lane_change_options_type_str, "lane_change_in_transition_amount_meters": lane_change_in_transition_amount_meters})
        
        # ############################################################
        # # Collision Checks
        # ############################################################
        # # check if new ego trajectories intersect with other actors
        intersection_bb = []
        intersection_timesteps = []
        if bbs_other_actors is not None:
            for j, ego_bounding_boxes_route_new_single in enumerate(forecasts_ego_adjusted):
                if len(intersection_bb) < j+1:
                    intersection_bb.append([])
                    intersection_timesteps.append([])
                for i, ego_bounding_box in enumerate(ego_bounding_boxes_route_new_single[0]):
                    intersects_with_ego = False
                    for vehicle_id, bounding_boxes in bbs_other_actors.items():
                        if i >= len(bounding_boxes):
                            continue
                        ego_bounding_box.location.z = 0
                        bounding_boxes[i].location.z = 0
                        intersects_with_ego = self.check_obb_intersection(ego_bounding_box, bounding_boxes[i])
                        if intersects_with_ego:
                            intersection_bb[j].append(bounding_boxes[i])
                            intersection_timesteps[j].append(i)
                            intersects_with_ego = False

        forecasts_ego_adjusted_wps = []
        for forecast_adjusted in forecasts_ego_adjusted:
            forecasts_ego_adjusted_wps.append([[f.location.x, f.location.y] for f in forecast_adjusted[0]])
        intersection_bb_bool = [len(intersection_bb[i])>0 for i in range(len(intersection_bb))]
        
        # add intersection to forecasts_ego_adjusted_info
        for i, (inter_bb_bool, inter_bb, inter_ts) in enumerate(zip(intersection_bb_bool, intersection_bb, intersection_timesteps)):
            forecasts_ego_adjusted_info[i]['dynamic_crash'] = inter_bb_bool
            forecasts_ego_adjusted_info[i]['dynamic_crash_timesteps'] = inter_ts
        
        if self.save_viz:
            self.viz_forecastings(path_rgb_image, bbs_other_actors, forecast_ego_org, forecasts_ego_adjusted, intersection_bb, forecasts_ego_adjusted_allowed, forecasts_ego_adjusted_mode)
            if self.viz_for_video:
                for i, (forecasts_ego_adjusted_i, intersection_bb_i, forecasts_ego_adjusted_allowed_i, forecasts_ego_adjusted_mode_i) in enumerate(zip(forecasts_ego_adjusted, intersection_bb, forecasts_ego_adjusted_allowed, forecasts_ego_adjusted_mode)):
                    self.viz_forecastings(path_rgb_image, bbs_other_actors, forecast_ego_org, [forecasts_ego_adjusted_i], [intersection_bb_i], [forecasts_ego_adjusted_allowed_i], [forecasts_ego_adjusted_mode_i], viz_for_video=True)
                    if i == 0:
                        self.viz_forecastings(path_rgb_image, bbs_other_actors, forecast_ego_org, [forecasts_ego_adjusted_i], [intersection_bb_i], [forecasts_ego_adjusted_allowed_i], [forecasts_ego_adjusted_mode_i], viz_for_video=True, viz_only_expert=True)
                    

        if self.save_samples:
            save_list.append({
                'path_rgb_image': path_rgb_image,
                'forecast_ego_org_wps': forecast_ego_org_wps,
                'forecasts_ego_adjusted': forecasts_ego_adjusted_wps,
                'forecasts_ego_adjusted_route': forecasts_ego_adjusted_route,
                'forecasts_ego_adjusted_info': forecasts_ego_adjusted_info,
                'forecasts_ego_adjusted_allowed': forecasts_ego_adjusted_allowed,
                'forecasts_ego_adjusted_mode': forecasts_ego_adjusted_mode,
                'intersection_bb_bool': intersection_bb_bool,
            })

            if self.save_instructions:
                dreamer_dict = get_info(save_list, route_adjusted, route_original, current_measurements, walker_close, ego_info)
            else:
                dreamer_dict = save_list
                
            Path(path_save).parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path_save, 'wt', encoding='utf-8') as f:
                json.dump(dreamer_dict, f, indent=4)   

    def viz_forecastings_new(self, image_path, forecasts_other_actors, bbs):
        """
        Visualize the forecasted bounding boxes for the ego vehicle and nearby actors on the image.
        """

        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels_per_meter = 10.0


        W = image.shape[1]
        H = image.shape[0]
        camera_intrinsics = np.asarray(build_projection_matrix(W,H,110))

        # Initialize the drawing context
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)

        white_img = Image.new('RGB', (H, H), (255, 255, 255))
        draw_white = ImageDraw.Draw(white_img)

        # Draw the predicted bounding boxes for the nearby actors
        idx = 0
        if forecasts_other_actors is not None:
            for actor_id, bounding_boxes in forecasts_other_actors.items():
                for bounding_box in bounding_boxes:
                    bounding_box.location.z = 0
                    color = (0, 0, 255-50*(idx))
                    if bounding_box.location.x > 0:
                        image_coords = project_center_corners(bounding_box, camera_intrinsics)
                    else:
                        image_coords = None
                    self.draw_bounding_box_center_new(draw, draw_white, bounding_box, image_coords, color=color, pixels_per_meter=pixels_per_meter)
                idx += 1
        
        if bbs is not None:
            for actor_id, bounding_boxes in bbs.items():
                for bounding_box in bounding_boxes:
                    bounding_box.location.z = 0
                    color = (0, 255-50*(idx), 0)
                    if bounding_box.location.x > 0:
                        image_coords = project_center_corners(bounding_box, camera_intrinsics)
                    else:
                        image_coords = None
                    self.draw_bounding_box_center_new(draw, draw_white, bounding_box, image_coords, color=color, pixels_per_meter=pixels_per_meter)
                idx += 1


        # concat img with white img
        full_img = Image.new('RGB', (W, H*2), (255, 255, 255))
        full_img.paste(img, (0, 0))
        full_img.paste(white_img, (0, H))

        # Save the image
        save_path_rgb = image_path.replace('rgb_0', 'rgb').replace('database', self.viz_save_path)
        time_str_yyyymmdd_hhmmss = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        save_path_rgb = save_path_rgb.replace('.jpg', f'_viz_{time_str_yyyymmdd_hhmmss}_{random.randint(1000, 9999)}.png')
        Path(save_path_rgb).parent.mkdir(parents=True, exist_ok=True)
        full_img.save(save_path_rgb)
        # full_img.save('forecasting.png')
        # white_img_concat.save('forecasting_white.png')  
        

    def viz_forecastings(self, image_path, forecasts_other_actors, ego_bounding_boxes, ego_bounding_boxes_news, intersection_bb, forecasts_ego_adjusted_allowed, forecasts_ego_adjusted_mode, viz_for_video=False, viz_only_expert=False):
        """
        Visualize the forecasted bounding boxes for the ego vehicle and nearby actors on the image.

        Args:
            image_path (str): The path to the image.
            forecasts_other_actors (dict): A dictionary containing the predicted bounding boxes for each actor.
            ego_bounding_boxes (list): A list of bounding boxes representing the future states of the ego vehicle.
        """
        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels_per_meter = 10.0

        W = image.shape[1]
        H = image.shape[0]
        camera_intrinsics = np.asarray(build_projection_matrix(W,H,110))

        # Initialize the drawing context
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)

        white_img = []
        draw_white = []
        draw_graph = []
        for viz_id in range(len(ego_bounding_boxes_news)+1):
            white_img.append(Image.new('RGB', (H, H), (255, 255, 255)))
            draw_white.append(ImageDraw.Draw(white_img[-1]))
            if self.viz_for_video:
                plt.style.use('dark_background')
                # matplotlib same size as white img
                fig, ax = plt.subplots(figsize=(H/100, H/100), dpi=800)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            else:
                fig, ax = plt.subplots(figsize=(H/100, H/100), dpi=100)

            # Draw the predicted bounding boxes for the nearby actors
            if not viz_for_video and not viz_only_expert:
                idx = 0
                if forecasts_other_actors is not None:
                    for actor_id, bounding_boxes in forecasts_other_actors.items():
                        for bounding_box in bounding_boxes:
                            color = (0, 0, 50*(idx+1))
                            self.draw_bounding_box_center(draw, draw_white[viz_id], bounding_box, camera_intrinsics, color=color, pixels_per_meter=pixels_per_meter)
                        idx += 1

            if viz_id >= len(ego_bounding_boxes_news):
                # Draw the predicted bounding boxes for the ego vehicle
                if not viz_for_video or viz_only_expert:

                    for bounding_box in ego_bounding_boxes[0]:
                        self.draw_bounding_box_center(draw, draw_white[viz_id], bounding_box, camera_intrinsics, color=(0, 255, 0), pixels_per_meter=pixels_per_meter)

                    # get speeds from bounding boxes
                    ego_locs = np.array([[bb.location.x, bb.location.y] for bb in ego_bounding_boxes[0]])
                    ego_speeds = np.linalg.norm(ego_locs[1:] - ego_locs[:-1], axis=1) * self.dataset_frame_rate
                    # ego_speeds = np.append(ego_speeds, ego_speeds[-1])
                    ax.plot(ego_speeds, color='g', label='ego speed')


                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Speed (m/s)')
                    ax.set_title('Ego Vehicle Speed Over Time')
                    ax.legend()

                    if self.viz_for_video:
                        save_path_rgb = image_path.replace('rgb_0', 'rgb').replace('database', self.viz_save_path)

                        time_str_yyyymmdd_hhmmss = time.strftime('%Y%m%d_%H%M%S', time.localtime())
                        save_path_rgb2 = save_path_rgb.replace('.jpg', f'_viz_{time_str_yyyymmdd_hhmmss}_{random.randint(1000, 9999)}_speed.png')
                        Path(save_path_rgb2).parent.mkdir(parents=True, exist_ok=True)
                        fig.savefig(save_path_rgb2)

                    draw_graph.append(fig)


            if viz_id < len(ego_bounding_boxes_news):
                if self.viz_for_video:
                    color = (0, 255, 0)
                else:
                    color = (0, 100*(viz_id+1), 0)
                    if not forecasts_ego_adjusted_allowed[viz_id]:
                        color = (255, 0, 0)

                if not viz_only_expert:
                    for bounding_box in ego_bounding_boxes_news[viz_id][0]:
                        self.draw_bounding_box_center(draw, draw_white[viz_id], bounding_box, camera_intrinsics, color=color, pixels_per_meter=pixels_per_meter)

                if not viz_for_video and not viz_only_expert:
                    if len(intersection_bb)>0 and intersection_bb[viz_id]:
                        for bounding_box in intersection_bb[viz_id]:
                            self.draw_bounding_box_center(draw, draw_white[viz_id], bounding_box, camera_intrinsics, color=(255, 0, 0), collide=True, pixels_per_meter=pixels_per_meter)

                # write the mode on top of the draw white
                font = ImageFont.truetype("arial.ttf", 20)
                draw_white[viz_id].text((10, 10), forecasts_ego_adjusted_mode[viz_id], font=font, fill=(0, 0, 0))
                # get speeds from bounding boxes
                ego_locs = np.array([[bb.location.x, bb.location.y] for bb in ego_bounding_boxes_news[viz_id][0]])
                ego_speeds = np.linalg.norm(ego_locs[1:] - ego_locs[:-1], axis=1) * self.dataset_frame_rate
                # ego_speeds = np.append(ego_speeds, ego_speeds[-1])
                ax.plot(ego_speeds, color='g', label='ego speed')


                ax.set_xlabel('Time Step')
                ax.set_ylabel('Speed (m/s)')
                ax.set_title('Ego Vehicle Speed Over Time')
                ax.legend()
                if self.viz_for_video:
                    save_path_rgb = image_path.replace('rgb_0', 'rgb').replace('database', self.viz_save_path)

                    time_str_yyyymmdd_hhmmss = time.strftime('%Y%m%d_%H%M%S', time.localtime())
                    save_path_rgb2 = save_path_rgb.replace('.jpg', f'_viz_{time_str_yyyymmdd_hhmmss}_{random.randint(1000, 9999)}_speed2.png')
                    Path(save_path_rgb2).parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(save_path_rgb2)

                draw_graph.append(fig)
            # add black corner around each white img
            # draw_white[viz_id].rectangle([0, 0, H, H], fill=(0, 0, 0))
        
        # concat all white imgs in a grif with max W width and mutliple of H height
        # round up to the nearest multiple of H
        num_rows = (len(ego_bounding_boxes_news)+1)  
        num_rows = int(np.ceil(num_rows))

        white_img_concat = Image.new('RGB', (W, H*num_rows), (255, 255, 255))
        for i, img_sing in enumerate(white_img):
            grid_row = i #// 2
            grid_col = 0 #i % 2 * 2  # Multiply by 2 to make space for graphs
            white_img_concat.paste(img_sing, (grid_col*H//2, grid_row*H))
            
            # Add matplotlib figure next to the white image
            if i < len(draw_graph):
                # Convert matplotlib figure to PIL Image
                draw_graph[i].canvas.draw()
                fig_data = np.frombuffer(draw_graph[i].canvas.tostring_rgb(), dtype=np.uint8)
                fig_data = fig_data.reshape(draw_graph[i].canvas.get_width_height()[::-1] + (3,))
                fig_img = Image.fromarray(fig_data)
                
                # Resize if needed to match height
                if fig_img.size[1] != H:
                    new_width = int(H * fig_img.size[0] / fig_img.size[1])
                    fig_img = fig_img.resize((new_width, H))
                    
                white_img_concat.paste(fig_img, ((grid_col+1)*W//2, grid_row*H))
                
                # Close matplotlib figure to free memory
                plt.close(draw_graph[i])

        # concat img with white img  
        full_img = Image.new('RGB', (W, H*(num_rows+1)), (255, 255, 255))
        full_img.paste(img, (0, 0))
        full_img.paste(white_img_concat, (0, H))

        # Save the image
        save_path_rgb = image_path.replace('rgb_0', 'rgb').replace('database', self.viz_save_path)
        time_str_yyyymmdd_hhmmss = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        save_path_rgb = save_path_rgb.replace('.jpg', f'_viz_{time_str_yyyymmdd_hhmmss}_{random.randint(1000, 9999)}.png')
        Path(save_path_rgb).parent.mkdir(parents=True, exist_ok=True)
        full_img.save(save_path_rgb)
        full_img.save('forecasting.png')
        # white_img_concat.save('forecasting_white.png')

    def draw_bounding_box_center_new(self, draw, draw_white, bounding_box, image_coords, color=(255, 0, 0), collide=False, pixels_per_meter=5.0):
        """
        Draw a bounding box on the image.

        Args:
            draw (PIL.ImageDraw): The drawing context.
            bounding_box (carla.BoundingBox): The bounding box to draw.
            color (tuple): The color of the bounding box.
        """
        # Get the bounding box corners
        center = bounding_box.location
        center = [[center.x, center.y, center.z]]

        pos = center
        extent = bounding_box.extent
        extent = [extent.x, extent.y, extent.z]
        yaw = -bounding_box.rotation.yaw # + np.pi / 2
        yaw = np.deg2rad(yaw)
            
        # get bbox corners coordinates
        corners = np.array([[-extent[0], -extent[1], 0.75],
                            [-extent[0], extent[1], 0.75],
                            [extent[0], extent[1], 0.75],
                            [extent[0], -extent[1], 0.75]])
        # rotate bbox
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                    [np.sin(yaw), np.cos(yaw), 0],
                                    [0, 0, 1]])
        corners = np.dot(corners, rotation_matrix)
        # translate bbox
        corners = corners + np.array(pos)

        corner_bev_img = [[corner[1] * pixels_per_meter + 256, -corner[0] * pixels_per_meter + 256] for corner in corners]
        for i in range(4):
            draw_white.line([tuple(corner_bev_img[i]), tuple(corner_bev_img[(i+1)%4])], fill=color, width=1)

        if image_coords is not None:
            # Draw the point
            size = 2
            avg_image_coords = np.mean(image_coords, axis=0)
            draw.ellipse([avg_image_coords[0] - size, avg_image_coords[1] - size, avg_image_coords[0] + size, avg_image_coords[1] + size], fill=color)
        
        return draw

    def draw_bounding_box_center(self, draw, draw_white, bounding_box, camera_intrinsics, color=(255, 0, 0), collide=False, pixels_per_meter=5.0):
        """
        Draw a bounding box on the image.

        Args:
            draw (PIL.ImageDraw): The drawing context.
            bounding_box (carla.BoundingBox): The bounding box to draw.
            color (tuple): The color of the bounding box.
        """

        # Get the bounding box corners
        center = bounding_box.location
        center = [[center.x, center.y, center.z]]


        pos = center
        extent = bounding_box.extent
        extent = [extent.x, extent.y, extent.z]
        yaw = -bounding_box.rotation.yaw # + np.pi / 2
        yaw = np.deg2rad(yaw)
            
        # get bbox corners coordinates
        corners = np.array([[-extent[0], -extent[1], 0.75],
                            [-extent[0], extent[1], 0.75],
                            [extent[0], extent[1], 0.75],
                            [extent[0], -extent[1], 0.75]])
        # rotate bbox
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                    [np.sin(yaw), np.cos(yaw), 0],
                                    [0, 0, 1]])
        corners = np.dot(corners, rotation_matrix)
        # translate bbox
        corners = corners + np.array(pos)

        corner_bev_img = [[corner[1] * pixels_per_meter + 256, -corner[0] * pixels_per_meter + 256] for corner in corners]
        for i in range(4):
            draw_white.line([tuple(corner_bev_img[i]), tuple(corner_bev_img[(i+1)%4])], fill=color, width=2)


        if collide:
            add = 2
        else:
            add = 0
        
        size = 2 + add
        center_bev_img = [center[0][1] * pixels_per_meter + 256, -center[0][0] * pixels_per_meter+256]
        draw_white.ellipse([center_bev_img[0] - size, center_bev_img[1] - size, center_bev_img[0] + size, center_bev_img[1] + size], fill=color)
        
        if center[0][0] < 0:
            return draw
        
        center_projected = project_points(center, camera_intrinsics)[0]

        # Draw the point
        size = 5 + add
        draw.ellipse([center_projected[0] - size, center_projected[1] - size, center_projected[0] + size, center_projected[1] + size], fill=color)
        
        return draw
    
    def get_bbs(self, actors, future_nearby_actors_by_id, future_nearby_actors_used_time_stamps_by_id, all_ego_positions, all_ego_yaws):
        if not actors:
            return  {}
        
        all_bbs_by_id = {}
        for actor in actors:
            # if actor['class'] == 'ego_car':
            #     ego_actor = actor
            #     pass
            # else:
                future_actor = future_nearby_actors_by_id[actor['id']]
                future_nearby_actors_used_time_stamps = [0] + [t+1 for t in future_nearby_actors_used_time_stamps_by_id[actor['id']]]
                actor_all_times = [actor] + future_actor
                all_positions = np.array([actor['position'] for actor in actor_all_times])
                all_yaws = np.array([actor['yaw'] for actor in actor_all_times])
                # convert all positions (current and future) (actor_all_times[i]['position']) first to global and then back to local of the current ego frame
                all_positions_global = np.array([t_u.conversion_2d(pos[:2], all_ego_positions[future_nearby_actors_used_time_stamps[t]], -all_ego_yaws[future_nearby_actors_used_time_stamps[t]]) for t, pos in enumerate(all_positions)])
                all_positions_local = np.array([t_u.inverse_conversion_2d(pos, all_ego_positions[0], all_ego_yaws[0]) for pos in all_positions_global])

                all_yaws_global = np.array([yaw + all_ego_yaws[t] for t, yaw in enumerate(all_yaws)])
                all_yaws_local = np.array([t_u.normalize_angle(yaw - all_ego_yaws[0]) for yaw in all_yaws_global])

                # get the bounding box for each actor for each timestep
                all_bbs = []
                extent_add_safety = 0
                if actor['class'] == 'walker':
                    extent_add_safety = 0.5
                
                for t in range(len(actor_all_times)):
                    location = carla.Location(
                        x=all_positions_local[t][0],
                        y=all_positions_local[t][1],
                        z=all_positions[t][2]
                    )
                    rotation = carla.Rotation(
                        pitch=0,
                        yaw=np.rad2deg(all_yaws_local[t]),
                        roll=0
                    )
                    extent = carla.Vector3D(x=actor_all_times[t]['extent'][0]+extent_add_safety, y=actor_all_times[t]['extent'][1]+extent_add_safety, z=actor_all_times[t]['extent'][2])
                    bounding_box = carla.BoundingBox(location, extent)
                    bounding_box.rotation = rotation
                    all_bbs.append(bounding_box)
                all_bbs_by_id[actor['id']] = all_bbs

        return all_bbs_by_id


    def forecast_vehicles(self, ego_actor, next_ego_actor_by_id, ego_position, ego_yaw, route=None, speeds_to_follow=None, desired_throttle=None, brake_probability=None, target_speed=None, return_final_speed=False, return_gt_speeds=False, use_wps_speed_controller=False):
        
        predicted_bounding_boxes = {}
        self._turn_controller.save_state()

        desired_speeds = []

        # between speeds_to_follow and target_speed and desired_throttle only one can be set
        if speeds_to_follow is not None:
            assert desired_throttle is None
            assert target_speed is None
        if desired_throttle is not None:
            assert speeds_to_follow is None
            assert target_speed is None
        if target_speed is not None:
            assert speeds_to_follow is None
            assert desired_throttle is None


        next_ego_actor = next_ego_actor_by_id[ego_actor["id"]]
        current_actions = np.array([[ego_actor["steer"], ego_actor["throttle"], ego_actor["brake"]]])
        actions_future = np.array([[actor["steer"], actor["throttle"], actor["brake"]] for actor in next_ego_actor])
        all_actions = np.concatenate([current_actions, actions_future])
        all_locations_global = np.concatenate([np.array([np.array(ego_actor["matrix"])[:3, 3]]), np.array([np.array(actor["matrix"])[:3, 3] for actor in next_ego_actor])])
        all_locations_local = np.array([t_u.inverse_conversion_2d(pos[:2], ego_position, ego_yaw) for pos in all_locations_global])
        all_yaws = np.concatenate([np.array([ego_actor["yaw"]]), np.array([actor["yaw"] for actor in next_ego_actor])])
        all_speeds = np.concatenate([np.array([ego_actor["speed"]]), np.array([actor["speed"] for actor in next_ego_actor])])

        # linearly interpolate to have 20 fps instead of 5 fps
        ratio = self.dataset_frame_rate / self.carla_frame_rate # we need ad datapoint every 0.2 seconds
        interp_steers = np.interp(np.arange(0, len(all_actions), ratio), np.arange(0, len(all_actions)), all_actions[:, 0])
        interp_throttles = np.interp(np.arange(0, len(all_actions), ratio), np.arange(0, len(all_actions)), all_actions[:, 1])
        interp_brakes = np.interp(np.arange(0, len(all_actions), ratio), np.arange(0, len(all_actions)), all_actions[:, 2])
        interp_location_global_x = np.interp(np.arange(0, len(all_locations_global), ratio), np.arange(0, len(all_locations_global)), all_locations_global[:, 0])
        interp_location_global_y = np.interp(np.arange(0, len(all_locations_global), ratio), np.arange(0, len(all_locations_global)), all_locations_global[:, 1])
        interp_location_global_z = np.interp(np.arange(0, len(all_locations_global), ratio), np.arange(0, len(all_locations_global)), all_locations_global[:, 2])
        interp_location_global = np.array([interp_location_global_x, interp_location_global_y, interp_location_global_z]).T
        interp_yaws = np.interp(np.arange(0, len(all_yaws), ratio), np.arange(0, len(all_yaws)), all_yaws)
        interp_speeds = np.interp(np.arange(0, len(all_speeds), ratio), np.arange(0, len(all_speeds)), all_speeds)

        # Get the previous control inputs (steering, throttle, brake) for the nearby actors
        previous_actions = np.array([ego_actor["steer"], ego_actor["throttle"], ego_actor["brake"]])

        # Get the current velocities, locations, and headings of the nearby actors
        velocities = np.array([ego_actor["speed"]])
        locations = np.array(np.asarray(ego_actor["matrix"])[:3, 3])
        # locations = np.array([actor["position"] for actor in ego_actor])
        headings = np.array([ego_actor["yaw"]+ego_yaw])

        # Initialize arrays to store future locations, headings, and velocities
        future_locations = np.zeros((self.num_future_frames_carla_fps, 3), dtype='float')
        future_headings = np.zeros((self.num_future_frames_carla_fps), dtype='float')
        future_velocities = np.zeros((self.num_future_frames_carla_fps), dtype='float')

        # Forecast the future locations, headings, and velocities for the nearby actors
        for i in range(self.num_future_frames_carla_fps):

            locations, headings, velocities = self.bicycle_model.forecast_ego_vehicle(locations, headings, velocities, previous_actions)
            previous_actions = np.array([interp_steers[i], interp_throttles[i], interp_brakes[i]])

            future_locations[i] = locations.copy()
            future_velocities[i] = velocities.copy()
            future_headings[i] = headings.copy()

            if route is not None:
                location_global = locations
                speed = velocities

                # the +1 is important, as otherwise the point is too close and the controller will not work -> leads to oscillations
                closest_route_point = np.argmin(np.linalg.norm(location_global[:2] - route[:, :2], axis=1)) + 1
                if closest_route_point >= len(route):
                    route = None # use original controll instead
                else:
                    closest_route_point = min(closest_route_point, len(route)-1)
                    route_ahead = route[closest_route_point:]

                    # get the steering angle from the PID and overwrite the original steering angle
                    steering = self._turn_controller.step(route_ahead, speed, locations[:2], headings, inference_mode=True)
                    previous_actions[0] = steering
            
            if use_wps_speed_controller:
                wps_global = interp_location_global[i:]
                # wps = np.array([t_u.inverse_conversion_2d(wp[:2], ego_position, ego_yaw) for wp in wps_global])[::(self.carla_frame_rate//self.dataset_frame_rate)]
                wps = np.array([t_u.inverse_conversion_2d(wp[:2], locations[:2], headings) for wp in wps_global])[::(self.carla_frame_rate//self.dataset_frame_rate)]
                one_second = self.dataset_frame_rate
                half_second = one_second // 2
                one_half_second = one_second + half_second

                if len(wps) >= one_second:
                    desired_speed = np.linalg.norm(wps[half_second - 2] - wps[one_second - 2] - wps[0]) * 2.0
                    desired_speeds.append(desired_speed)

                    throttle, control_brake = self._longitudinal_controller.get_throttle_and_brake(False, desired_speed, velocities[0])

                    previous_actions[1] = throttle
                    previous_actions[2] = control_brake

            if speeds_to_follow is not None:
                desired_speed = speeds_to_follow[i]
                throttle, control_brake = self._longitudinal_controller.get_throttle_and_brake(False, desired_speed, velocities[0])
                previous_actions[1] = throttle
                previous_actions[2] = control_brake
            
            if desired_throttle is not None:
                previous_actions[1] = desired_throttle
                previous_actions[2] = 0

            if brake_probability is not None:
                if random.random() < brake_probability:
                    previous_actions[1] = 0
                    previous_actions[2] = 1
                else:
                    previous_actions[1] = 0
                    previous_actions[2] = 0

            if target_speed is not None:
                throttle, control_brake = self._longitudinal_controller.get_throttle_and_brake(False, target_speed, velocities[0])
                previous_actions[1] = throttle
                previous_actions[2] = control_brake
                
        # Convert future headings to degrees
        future_headings = np.rad2deg(future_headings)

        # Convert global coordinates to egocentric coordinates
        ego_position = np.array(ego_position)
        ego_orientation = np.array(ego_yaw)
        for time_step in range(future_locations.shape[0]):
            target_point_2d = future_locations[time_step, :2]

            ego_target_point = t_u.inverse_conversion_2d(target_point_2d, ego_position, ego_orientation).tolist()
            future_locations[time_step, :2] = ego_target_point
            future_headings[time_step] -= np.rad2deg(ego_yaw)


            # Calculate the predicted bounding boxes for each future frame
            predicted_actor_boxes = []
            for i in range(self.num_future_frames_carla_fps):         
                location = carla.Location(
                    x=future_locations[i, 0].item(),
                    y=future_locations[i, 1].item(),
                    z=future_locations[i, 2].item()
                )
                rotation = carla.Rotation(
                    pitch=0,
                    yaw=future_headings[i],
                    roll=0
                )
                extent = ego_actor["extent"]
                extent = carla.Vector3D(x=extent[0], y=extent[1], z=extent[2])

                # Create the bounding box for the future frame
                bounding_box = carla.BoundingBox(location, extent)
                bounding_box.rotation = rotation

                # Append the bounding box to the list of predicted bounding boxes for this actor
                predicted_actor_boxes.append(bounding_box)

            # Store the predicted bounding boxes for this actor in the dictionary
            # sample the bounding boxes every 4 frames
            predicted_actor_boxes = predicted_actor_boxes[::(self.carla_frame_rate//self.dataset_frame_rate)]

            predicted_bounding_boxes[ego_actor["id"]] = predicted_actor_boxes
        self._turn_controller.load_state()

        return_tuple = (predicted_bounding_boxes,)
        
        if return_final_speed:
            final_speed = future_velocities[-1]
            final_speed = round(final_speed, 1)
            return_tuple += (final_speed,)
            # return predicted_bounding_boxes, final_speed

        if return_gt_speeds:
            return_tuple += (future_velocities,)
        if len(return_tuple) == 1:
            return_tuple = predicted_bounding_boxes
        return return_tuple
      
    
    def calculate_shifted_trajectory(self, original_trajectory, lane_change_in_transition_amount_meters):
        """
        Calculate a new trajectory that is shifted to the left by X meters.
        original_trajectory: The original trajectory to shift.
        Y_0: The starting index of the trajectory to shift.
        X: The distance to shift the trajectory to the left.
        """

        def normalize(v):
            norm = np.linalg.norm(v)
            if norm == 0:
                return v
            return v / norm
        
        Y_0, dY, X = lane_change_in_transition_amount_meters
        new_trajectory = list(original_trajectory[:Y_0]) if Y_0 > 0 else []
        for i in range(Y_0 + dY, len(original_trajectory) - 1):
            p1 = original_trajectory[i]
            p2 = original_trajectory[i + 1]
            
            # Direction vector from p1 to p2
            direction = p2 - p1
            
            # Perpendicular vector to the left (rotating 90 degrees counterclockwise)
            left_normal = np.array([-direction[1], direction[0]])
            left_normal = normalize(left_normal)
            
            # New point offset by X meters to the left
            new_point = p1 + X * left_normal
            
            # if len(new_trajectory) == 0 or np.linalg.norm(new_point - new_trajectory[-1]) >= 1.0:
            new_trajectory.append(new_point)
        
        # Ensure the last point is included
        last_point = original_trajectory[-1] + X * np.array([-direction[1], direction[0]])
        if np.linalg.norm(last_point - new_trajectory[-1]) >= 1.0:
            new_trajectory.append(last_point)

        spacing = 1.0
        new_trajectory = equal_spacing_route(new_trajectory)
        
        return np.array(new_trajectory)
    
    def _dot_product(self, vector1, vector2):
        """
        Calculate the dot product of two vectors.

        Args:
            vector1 (carla.Vector3D): The first vector.
            vector2 (carla.Vector3D): The second vector.

        Returns:
            float: The dot product of the two vectors.
        """
        return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z

    def cross_product(self, vector1, vector2):
        """
        Calculate the cross product of two vectors.

        Args:
            vector1 (carla.Vector3D): The first vector.
            vector2 (carla.Vector3D): The second vector.

        Returns:
            carla.Vector3D: The cross product of the two vectors.
        """
        x = vector1.y * vector2.z - vector1.z * vector2.y
        y = vector1.z * vector2.x - vector1.x * vector2.z
        z = vector1.x * vector2.y - vector1.y * vector2.x

        return carla.Vector3D(x=x, y=y, z=z)

    def get_separating_plane(self, relative_position, plane_normal, obb1, obb2):
        """
        Check if there is a separating plane between two oriented bounding boxes (OBBs).

        Args:
            relative_position (carla.Vector3D): The relative position between the two OBBs.
            plane_normal (carla.Vector3D): The normal vector of the plane.
            obb1 (carla.BoundingBox): The first oriented bounding box.
            obb2 (carla.BoundingBox): The second oriented bounding box.

        Returns:
            bool: True if there is a separating plane, False otherwise.
        """
        # Calculate the projection of the relative position onto the plane normal
        projection_distance = abs(self._dot_product(relative_position, plane_normal))

        # Calculate the sum of the projections of the OBB extents onto the plane normal
        obb1_projection = (
            abs(self._dot_product(obb1.rotation.get_forward_vector() * obb1.extent.x, plane_normal)) +
            abs(self._dot_product(obb1.rotation.get_right_vector() * obb1.extent.y, plane_normal)) +
            abs(self._dot_product(obb1.rotation.get_up_vector() * obb1.extent.z, plane_normal))
        )

        obb2_projection = (
            abs(self._dot_product(obb2.rotation.get_forward_vector() * obb2.extent.x, plane_normal)) +
            abs(self._dot_product(obb2.rotation.get_right_vector() * obb2.extent.y, plane_normal)) +
            abs(self._dot_product(obb2.rotation.get_up_vector() * obb2.extent.z, plane_normal))
        )

        # Check if the projection distance is greater than the sum of the OBB projections
        return projection_distance > obb1_projection + obb2_projection

    def check_obb_intersection(self, obb1, obb2):
        """
        Check if two 3D oriented bounding boxes (OBBs) intersect.

        Args:
            obb1 (carla.BoundingBox): The first oriented bounding box.
            obb2 (carla.BoundingBox): The second oriented bounding box.

        Returns:
            bool: True if the two OBBs intersect, False otherwise.
        """
        relative_position = obb2.location - obb1.location

        # Check for separating planes along the axes of both OBBs
        if (self.get_separating_plane(relative_position, obb1.rotation.get_forward_vector(), obb1, obb2) or
            self.get_separating_plane(relative_position, obb1.rotation.get_right_vector(), obb1, obb2) or
            self.get_separating_plane(relative_position, obb1.rotation.get_up_vector(), obb1, obb2) or
            self.get_separating_plane(relative_position, obb2.rotation.get_forward_vector(), obb1, obb2) or
            self.get_separating_plane(relative_position, obb2.rotation.get_right_vector(), obb1, obb2) or
            self.get_separating_plane(relative_position, obb2.rotation.get_up_vector(), obb1, obb2)):
            
            return False

        # Check for separating planes along the cross products of the axes of both OBBs
        if (self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_forward_vector()), obb1,obb2) or 
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_right_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_up_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_forward_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_right_vector()), obb1, obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_up_vector()), obb1, obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_forward_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_right_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_up_vector(),obb2.rotation.get_up_vector()), obb1, obb2)):
            
            return False

        # If no separating plane is found, the OBBs intersect
        return True
    
    def get_min_max_pos_given_speed_and_deltaT(self, pos, speed, time_steps):
        """
        Get the minimum and maximum position given the speed and deltaT.

        Args:
            pos (list): The current position.
            speed (float): The current speed.
            deltaT (float): The time step.

        Returns:
            tuple: The minimum and maximum position.
        """
        action_min = np.array([0, 0, 1]) # steer, throttle, brake
        action_max = np.array([0, 1, 0])

        pos_min = np.array(pos + [0.])
        pos_max = np.array(pos + [0.])

        heading = np.array([0.0])
        
        speed_min = np.array([speed])
        speed_max = np.array([speed])
        
        for i in range(time_steps*(self.carla_frame_rate//self.dataset_frame_rate)):
            pos_min, _, speed_min = self.bicycle_model.forecast_ego_vehicle(pos_min, heading, speed_min, action_min)
            pos_max, _, speed_max = self.bicycle_model.forecast_ego_vehicle(pos_max, heading, speed_max, action_max)

        return pos_min, pos_max



if __name__ == '__main__':
    import tqdm
    creator = CarlaAlternativeCreator()
    
    multi_processing = True
    print(f"Processing {len(creator.data_boxes_paths)} data boxes")
    
    if multi_processing:
        from tqdm.contrib.concurrent import process_map
        len_data_boxes = len(creator.data_boxes_paths)
        r = process_map(creator.process_data, range(0,len_data_boxes), max_workers=64, chunksize=10000)
    else:
        for i in tqdm.tqdm(range(len(creator.data_boxes_paths))):
            creator.process_data(i)