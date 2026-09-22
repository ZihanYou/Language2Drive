"""
This script generates buckets for the CARLA dataset.
partially taken from https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py
(MIT licence)
"""

# Standard library imports
import glob
import gzip
import itertools
import math
import os
import pickle as pkl
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, TypedDict

# Third party imports
import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import ujson
from einops import rearrange, repeat
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Local imports
from simlingo_base_training.utils.custom_types import DrivingExample, DrivingInput, DrivingLabel


class CARLA_Data(Dataset):
    """
    Custom dataset that dynamically loads a CARLA dataset from disk.
    """

    def __init__(self,
            batch_size,
            num_workers,
            data_path,
            hist_len,
            pred_len,
            wp_dilation,
            skip_first_n_frames,
            num_route_points,
            smooth_route,
            dense_route_planner_min_distance,
            dense_route_planner_max_distance,
            split,
            bucket_name=None,
            bucket_proportion=None,
            bucket_path=None,
        ):
        # self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.wp_dilation = wp_dilation
        self.skip_first_n_frames = skip_first_n_frames
        self.num_route_points = num_route_points
        self.smooth_route = smooth_route
        self.dense_route_planner_min_distance = dense_route_planner_min_distance
        self.dense_route_planner_max_distance = dense_route_planner_max_distance

        filter_infractions_by_coords = False
        filter_infractions_per_route = True

        self.images = []
        self.boxes = []
        self.measurements = []
        self.sample_start = []

        self.temporal_measurements = []

        total_routes = 0
        perfect_routes = 0
        crashed_routes = 0

        loss_taken = []
        loss_not_taken = []

        route_dirs = glob.glob(self.data_path + '/**/data/**/Town*', recursive=True)
        print(f'Found {len(route_dirs)} routes in {self.data_path}')

        random.seed(42)
        random.shuffle(route_dirs)
        # route_dirs = route_dirs[:10]

        for sub_root in tqdm(route_dirs, file=sys.stdout):

            route_dir = sub_root
            if not os.path.exists(route_dir + '/rgb'):
                continue

            num_seq = len(os.listdir(route_dir + '/rgb'))

            for seq in range(skip_first_n_frames, num_seq - self.pred_len - self.hist_len):
                image = []
                box = []
                measurement = []
                measurement_file = route_dir + '/measurements' + f'/{(seq + self.hist_len):04}.json.gz'
                
                # Loads the current (and past) frames (if seq_len > 1)
                skip = False
                for idx in range(self.hist_len):
                    image.append(route_dir + '/rgb' + (f'/{(seq + idx):04}.jpg'))
                    box.append(route_dir + '/boxes' + (f'/{(seq + idx):04}.json.gz'))

                if skip:
                    continue

                measurement.append(route_dir + '/measurements')

                self.images.append(image)
                self.boxes.append(box)
                self.measurements.append(measurement)
                self.sample_start.append(seq)

        # There is a complex "memory leak"/performance issue when using Python
        # objects like lists in a Dataloader that is loaded with
        # multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects
        # because they only have 1 refcount.
        self.images = np.array(self.images).astype(np.string_)
        self.boxes = np.array(self.boxes).astype(np.string_)
        self.measurements = np.array(self.measurements).astype(np.string_)

        self.sample_start = np.array(self.sample_start)
        print(f'[{split} samples]: Loading {len(self.images)} images from {self.data_path}')
        print('Total amount of routes:', total_routes)
        print('Crashed routes:', crashed_routes)
        print('Perfect routes:', perfect_routes)

    def __len__(self):
        """Returns the length of the dataset. """
        return self.images.shape[0]

    def __getitem__(self, index):
        """Returns the item at index idx. """
        # cv2.setNumThreads(0)

        data = {}
        images = self.images[index]
        boxes = self.boxes[index]
        measurements = self.measurements[index]
        sample_start = self.sample_start[index]

        # load measurements
        loaded_images = []
        loaded_boxes = []
        loaded_measurements = []

        # Since we load measurements for future time steps, we load and store them separately
        for i in range(self.hist_len):
            measurement_file = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + i):04}.json.gz')

            with gzip.open(measurement_file, 'r') as f1:
                measurements_i = ujson.load(f1)
            loaded_measurements.append(measurements_i)

            boxes_file = measurement_file.replace('measurements', 'boxes')
            with gzip.open(boxes_file, 'r') as f1:
                boxes_i = ujson.load(f1)
            loaded_boxes.append(boxes_i)


        end = self.pred_len + self.hist_len
        start = self.hist_len
        measurement_file_current = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + start-1):04}.json.gz')

        current_boxes = str(boxes[start-1], encoding='utf-8')
        with gzip.open(current_boxes, 'r') as f1:
            current_boxes = ujson.load(f1)

        for i in range(start, end, self.wp_dilation):
            measurement_file = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + i):04}.json.gz')

            with gzip.open(measurement_file, 'r') as f1:
                measurements_i = ujson.load(f1)
            loaded_measurements.append(measurements_i)

            boxes_file = measurement_file.replace('measurements', 'boxes')
            with gzip.open(boxes_file, 'r') as f1:
                boxes_i = ujson.load(f1)
            loaded_boxes.append(boxes_i)


        current_measurement = loaded_measurements[self.hist_len - 1]

        waypoints = self.get_waypoints(loaded_measurements[self.hist_len - 1:],
                                                                        y_augmentation=0,
                                                                        yaw_augmentation=0)

        data['waypoints'] = np.array(waypoints[:-1])

        future_speeds = [measurement["speed"] for measurement in loaded_measurements[self.hist_len - 1:]]
        data['future_speeds'] = np.array(future_speeds[:-1]) #-1 because current time step is also included in future time steps

        # get the acceleration from the future speeds
        label_acceleration_ms2 = np.diff(future_speeds) / 0.2
        data['label_acceleration_ms2'] = label_acceleration_ms2
        data['brake'] = current_measurement['brake']
        data['control_brake'] = current_measurement['control_brake']
        data['steer'] = current_measurement['steer']
        data['throttle'] = current_measurement['throttle']
        data['junction'] = current_measurement['junction']
        data['speed'] = [current_measurement['speed']]
        data['map_speed_limit'] = current_measurement['speed_limit'] #m/s privileged!!!!!
        data['target_speed'] = current_measurement['target_speed']
        data['hazards'] = {
            'vehicle': current_measurement['vehicle_hazard'],
            'light': current_measurement['light_hazard'],
            'walker': current_measurement['walker_hazard'],
            'stop_sign': current_measurement['stop_sign_hazard'],
        }

        ego_info = {}
        for box in current_boxes:
            if box['class'] == 'ego_info':
                ego_info = box
            if box['class'] == 'walker':
                if current_measurement['walker_affecting_id'] is not None:
                    if box['id'] == current_measurement['walker_affecting_id']:
                        data['walker_affecting'] = box
            if box['class'] == 'car':
                if current_measurement['vehicle_affecting_id'] is not None:
                    if box['id'] == current_measurement['vehicle_affecting_id']:
                        data['vehicle_affecting'] = box

            if 'traffic_light' in box['class']:
                if box['affects_ego']:
                    data['traffic_light'] = box
                    if box['distance'] < 15:
                        data['traffic_light_close'] = box

            if 'stop_sign' in box['class']:
                if box['affects_ego']:
                    data['stop_sign'] = box
                    if box['distance'] < 15:
                        data['stop_sign_close'] = box

        data['ego_info'] = ego_info
        data['measurement_path'] = measurement_file_current
        if data['ego_info']['distance_to_junction'] is None:
            data['ego_info']['distance_to_junction'] = 10000


        ##########################
        ###### data buckets ######
        ##########################
        data['buckets'] = []
        remove_sample = False

        is_affecting_in_box = True
        leading_object = current_measurement['speed_reduced_by_obj_type']
        if leading_object is not None and leading_object.split('.')[0] == 'vehicle' and current_measurement['speed_reduced_by_obj_distance'] < 10:
            if leading_object == 'vehicle.bh.crossbike' or leading_object == 'vehicle.diamondback.century' or leading_object == 'vehicle.gazelle.omafiets':
                pass
            else:
                for box in loaded_boxes[self.hist_len - 1:self.hist_len - 1 + 4]:
                    is_affecting_in_bb = False

                    for bb in box:
                        if 'id' in bb and bb['id'] == current_measurement['speed_reduced_by_obj_id']:
                            is_affecting_in_bb = True
                            break

                    if not is_affecting_in_bb:
                        is_affecting_in_box = False
                        break
                    
                if not is_affecting_in_box:
                    data['buckets'].append('vehicle_dissapears')
                    remove_sample = True
                    # we don't want these samples in the other buckets

        if not remove_sample:
            ########## speed limit #########
            speed_limit = data['map_speed_limit']
            data['buckets'].append(f'speed_limit_{speed_limit}')

            ######### target speed #########
            target_speed = data['target_speed']
            target_speed_bins = [0.5, 5, 10, 15, 20, 25, 1000000]
            for i, speed in enumerate(target_speed_bins):
                if target_speed < speed:
                    data['buckets'].append(f'target_speed_{speed}')
                    break

            ######### lateral control #########
            # from waypoints mean of y
            lateral_control = np.abs(np.mean(data['waypoints'][:, 1]))
            lateral_control_bins = [0.1, 1, 2, 5, 1000000]
            for i, control in enumerate(lateral_control_bins):
                if lateral_control < control:
                    data['buckets'].append(f'lateral_control_{control}')
                    break

            ######### start from stop #########
            # get mean distance between waypoints
            distance = np.mean(np.linalg.norm(data['waypoints'][1:] - data['waypoints'][:-1], axis=1))

            if data['speed'][0] < 0.5 and distance > 0.1:
                data['buckets'].append('start_from_stop_old')

            target_speed = data['target_speed']
            current_speed = current_measurement['speed']
            if current_speed < 0.5 and target_speed > 0.8:
                data['buckets'].append('start_from_stop')

            ######### acceleration ##########
            accel = np.mean(data['label_acceleration_ms2'][:4])
            accel_bins = [-40, -20, -5, -1, 1, 5, 20, 40, 1000000]
            for i, acc in enumerate(accel_bins):
                if accel < acc:
                    data['buckets'].append(f'acceleration_{acc}')
                    break
            

            leading_object = current_measurement['speed_reduced_by_obj_type']
            if leading_object is not None and current_measurement['speed_reduced_by_obj_distance'] < 30:
                distance = 30
                if leading_object.split('.')[0] == 'vehicle':
                    distance = 20
                    leading_object = 'vehicle'
                if leading_object.split('.')[0] == 'walker':
                    distance = 20
                    leading_object = 'walker'
                data['buckets'].append(f"leading_object_{leading_object}")

            ########## junction ##########
            if data['ego_info']['distance_to_junction'] < 10:
                data['buckets'].append('junction')

            ########## red light ##########
            if data['ego_info']['traffic_light_state'] == 'Red' and data['ego_info']['distance_to_junction'] < 20:
                data['buckets'].append('red_light')

            ######### green light #########
            if data['ego_info']['traffic_light_state'] == 'Green' and data['ego_info']['distance_to_junction'] < 20:
                data['buckets'].append('green_light')

            ######### changed route #########
            if current_measurement['changed_route']:
                data['buckets'].append('changed_route')

            ######### hazards #########
            hazards = data['hazards']
            for hazard in hazards.keys():
                if hazards[hazard]:
                    if hazard == 'vehicle' and 'vehicle_affecting' not in data:
                        continue
                    if hazard == 'walker' and 'walker_affecting' not in data:
                        data['buckets'].append('walker_hazard')
                        continue
                    data['buckets'].append(hazard)
                    # coming from the front so yaw should be around 180 degree
                    if hazard == 'vehicle' and data['vehicle_affecting']['yaw'] > np.pi - 0.6 and data['vehicle_affecting']['yaw'] < np.pi + 0.6:
                        if data['ego_info']['traffic_light_state'] == 'Red' and data['ego_info']['distance_to_junction'] < 20:
                            pass
                        else:
                            data['buckets'].append('vehicle_front')
                    # coming from the side so yaw should not match ego yaw
                    elif hazard == 'vehicle' and abs(data['vehicle_affecting']['yaw']) > 0.5:
                        if data['ego_info']['traffic_light_state'] == 'Red' and data['ego_info']['distance_to_junction'] < 20:
                            pass
                        else:
                            data['buckets'].append('vehicle_side')

            # vehicle hazard vehicel coming from the side
            ######### brake #########
            brake = data['brake'] or data['control_brake']
            if brake:
                data['buckets'].append('brake')

            ######### stop sign #########
            if 'stop_sign_close' in data:
                data['buckets'].append('stop_sign_close')


            if 'parking_lane' in measurement_file_current and abs(lateral_control) > 0.2:
                data['buckets'].append('parkinglane')

        ######### recovery data #########
        # if current_measurement['noise_level'] == 1:
        #     if current_measurement['noise_type'] is None and current_measurement['steps_till_next_begin_of_noise'] > 70 and current_measurement['steps_till_next_begin_of_noise'] != 100:
        #         # we don't want to use the data while adding noise but only afterwards
        #         data['buckets'].append('recovery_data_small')
        
        # elif current_measurement['noise_level'] == 2:
        #     if current_measurement['noise_type'] is not None:
        #         data['buckets'].append('recovery_data_large')

        return data
        

    def get_waypoints(self, measurements, y_augmentation=0.0, yaw_augmentation=0.0):
        """transform waypoints to be origin at ego_matrix"""
        origin = measurements[0]
        origin_matrix = np.array(origin['ego_matrix'])[:3]
        origin_translation = origin_matrix[:, 3:4]
        origin_rotation = origin_matrix[:, :3]

        waypoints = []
        for index in range(len(measurements)):
            waypoint = np.array(measurements[index]['ego_matrix'])[:3, 3:4]
            waypoint_ego_frame = origin_rotation.T @ (waypoint - origin_translation)
            # Drop the height dimension because we predict waypoints in BEV
            waypoints.append(waypoint_ego_frame[:2, 0])

        # Data augmentation
        waypoints_aug = []
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                                                                                                            np.cos(aug_yaw_rad)]])

        translation = np.array([[0.0], [y_augmentation]])
        for waypoint in waypoints:
            pos = np.expand_dims(waypoint, axis=1)
            waypoint_aug = rotation_matrix.T @ (pos - translation)
            waypoints_aug.append(np.squeeze(waypoint_aug))

        return waypoints_aug


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        bucket_path,
    ):

        super().__init__()
        self.batch_size = 64
        self.num_workers = 64
        self.data_path = data_path
        self.hist_len = 3
        self.pred_len = 11
        self.wp_dilation = 1
        self.skip_first_n_frames = 10
        self.num_route_points = 40
        self.smooth_route = False
        self.dense_route_planner_min_distance = 3.5
        self.dense_route_planner_max_distance = 50
        self.shuffle = False
        self.bucket_path = bucket_path

    def setup(self, stage=None):
       
        self.train_dataset = CARLA_Data(
            self.batch_size,
            self.num_workers,
            self.data_path,
            self.hist_len,
            self.pred_len,
            self.wp_dilation,
            self.skip_first_n_frames,
            self.num_route_points,
            self.smooth_route,
            self.dense_route_planner_min_distance,
            self.dense_route_planner_max_distance,
            split="train",
            bucket_name="all",
            bucket_path=self.bucket_path,
        )

        

    def predict_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.dl_collate_fn,
        )

    def dl_collate_fn(self, batch):
        paths = [data['measurement_path'] for data in batch if data['measurement_path'] is not None]
        buckets = [data['buckets'] for data in batch if data['buckets'] is not None]
        speeds = [data['target_speed'] for data in batch if data['target_speed'] is not None]

        return paths, buckets, speeds


def main():

    data_path = 'database/simlingo'
    save_path = f'database/bucketsv2_simlingo'

    Path(save_path).mkdir(parents=True, exist_ok=True)

    init_from_folder = None #'database/buckets_simlingo' #'database/buckets_expertv3_3' #'database/buckets_expertv3_2_woallweather'
    all_routes = None

    if init_from_folder is not None:
        with open(f'{init_from_folder}/buckets_paths.pkl', 'rb') as f:
            buckets_paths = pkl.load(f)
            all_paths = list(itertools.chain(*list(buckets_paths.values())))
            all_routes = [p.split('/measurements')[0] for p in all_paths]
            all_routes = list(set(all_routes))

        with open(f'{init_from_folder}/buckets_stats.json', 'r') as f:
            buckets_stats = ujson.load(f)

    else:
        buckets_paths = {}
        buckets_stats = {'total': 0}

    dm = DataModule(data_path=data_path, bucket_path=all_routes)
    dm.setup()
    dl = dm.predict_dataloader()

    all_speeds = []
    for idx, batch in enumerate(tqdm(dl)):

        all_speeds.extend(batch[2])

        paths, buckets,speeds = batch
        for path, bucket in zip(paths, buckets):
            for b in bucket:
                buckets_paths.setdefault(b, []).append(path)
                # buckets_stats[b] = buckets_stats.get(b, 0) + 1

        buckets_stats['total'] += len(paths)

        if idx % 1000 == 0:
            print(f'Processed {idx} batches')
            buckets_paths_to_save = dict(sorted(buckets_paths.items()))

            # save buckets as pkl
            with open(f'{save_path}/buckets_paths.pkl', 'wb') as f:
                pkl.dump(buckets_paths_to_save, f)

    # remove duplicates in buckets_paths[b] and count them to also adjust the stats
    for b in buckets_paths:
        buckets_paths[b] = list(set(buckets_paths[b]))
        buckets_stats[b] = len(buckets_paths[b])

    


    print(f'Processed {idx} batches')
    buckets_paths_to_save = dict(sorted(buckets_paths.items()))

    # save buckets as pkl
    with open(f'{save_path}/buckets_paths.pkl', 'wb') as f:
        pkl.dump(buckets_paths_to_save, f)


    # sort buckets_paths by key name
    buckets_paths = dict(sorted(buckets_paths.items()))
    buckets_stats = dict(sorted(buckets_stats.items()))

    # plot histogram of speeds, log scale y axis
    plt.figure(figsize=(20, 10))
    sns.histplot(all_speeds, kde=True, log_scale=(False, True))
    plt.title('Speeds distribution')
    plt.savefig(f'{save_path}/speeds_hist.png')

    # get histogramm as dict
    speeds_hist = {}
    for s in all_speeds:
        # assin to buckets
        target_speeds_bins = [0.2, 2.0, 5.0, 7.77, 11.11, 15.0, 18.0, 22.0, 26.5, 31.0]
        s_bin_ix = 0
        for i, speed in enumerate(target_speeds_bins):
            if s < speed:
                s_bin_ix = i
                break
        if s_bin_ix not in speeds_hist:
            speeds_hist[s_bin_ix] = 0
        speeds_hist[s_bin_ix] += 1

    # get loss weights for speed bins
    total = sum(speeds_hist.values())
    loss_weights = {}
    for k, v in speeds_hist.items():
        loss_weights[k] = total / (len(speeds_hist) * v)

    # sort by key
    loss_weights = dict(sorted(loss_weights.items()))
    # get list
    loss_weights = list(loss_weights.values())
    
    # sort speeds_hist by occurence, highest first
    speeds_hist = dict(sorted(speeds_hist.items(), key=lambda item: item[1], reverse=True))

    # save buckets as pkl
    with open(f'{save_path}/buckets_paths.pkl', 'wb') as f:
        pkl.dump(buckets_paths, f)

    # save buckets stats as json
    with open(f'{save_path}/buckets_stats.json', 'w') as f:
        ujson.dump(buckets_stats, f, indent=4)

    # plot stats as bar plot sorted by item number

    buckets_stats.pop('total')
    buckets_stats = dict(sorted(buckets_stats.items(), key=lambda item: item[1], reverse=True))
    plt.figure(figsize=(20, 10))
    sns.barplot(x=list(buckets_stats.keys()), y=list(buckets_stats.values()))
    plt.xticks(rotation=90)
    plt.title('Buckets distribution')
    plt.savefig(f'{save_path}/buckets_stats.png')


if __name__ == "__main__":
    main()