import os
import glob
import gzip
import json
import ujson
import tqdm
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re
import textwrap

from dataset_generation.language_labels.utils import *

"""
Main class for processing and converting the pdm_lite dataset to the Commentary for SimLingo.
Limitations:
- highway acceleration lane and exit lane are treated as intersections.

"""

class COMsGenerator():
    def __init__(self, args):
        # Image and camera parameters
        self.TARGET_IMAGE_SIZE = args.target_image_size
        self.ORIGINAL_IMAGE_SIZE = args.original_image_size
        self.ORIGINAL_FOV = args.original_fov

        # Region of interest (ROI) for image projection
        self.MIN_X = args.min_x
        self.MAX_X = args.max_x
        self.MIN_Y = args.min_y
        self.MAX_Y = args.max_y

        # Commentary config
        self.SKIP_FIRST_FRAMES = 10
        self.HISTORY_LEN = 5 # including the current time step
        self.FUTURE_LEN = 10

        # Sampling parameters
        self.random_subset_count = args.random_subset_count
        self.sample_frame_mode = args.sample_frame_mode
        self.sample_uniform_interval = args.sample_uniform_interval

        # Visualization and saving options
        self.save_examples = args.save_examples         
        self.visualize_projection = args.visualize_projection 
        self.filter_routes_by_result = args.filter_routes_by_result

        self.data_directory = args.data_directory 
        self.path_keyframes = args.path_keyframes 

        self.output_directory = args.output_directory 
        self.output_examples_directory = args.output_examples_directory
        self.skip_existing = args.skip_existing

        # Build camera projection matrix
        self.CAMERA_MATRIX = build_projection_matrix(self.ORIGINAL_IMAGE_SIZE[0],
                                                     self.ORIGINAL_IMAGE_SIZE[1],
                                                     self.ORIGINAL_FOV)

        # creaete the directories where we save the graph and some graph examples
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)
        if self.save_examples:
            Path(self.output_examples_directory).mkdir(parents=True, exist_ok=True)

        # all the paths to the boxes in the data
        # self.data_boxes_paths = glob.glob(os.path.join(self.data_directory, '**/boxes/*.json.gz'), recursive=True)
        # all the paths to the boxes in the data
        self.data_boxes_paths_all = glob.glob(os.path.join(self.data_directory, 'data/simlingo/*/*/*/*/boxes/*.json.gz'))
        print(f"Number of boxes paths: {len(self.data_boxes_paths_all)}")
        # if self.skip_existing:
        #     self.data_boxes_paths = []
        #     # all the paths to the results in the data
        #     data_results_paths = glob.glob(os.path.join(self.data_directory, 'commentary/simlingo/*/*/*/*/commentary'))
        #     data_results_parent_paths = glob.glob(os.path.join(self.data_directory, 'commentary/simlingo/*'))
        #     print(f"Number of results parent paths: {len(data_results_paths)}")
        #     # remove the paths that already have a vqa file
        #     for box_path in tqdm.tqdm(self.data_boxes_paths_all):
                
        #         parent_result_dir = '/'.join(Path(box_path).parts[:-5]).replace('/data/', '/commentary/')
        #         if parent_result_dir not in data_results_parent_paths:
        #             self.data_boxes_paths.append(box_path)
        #         else:
        #             # print(parent_result_dir)
        #             result_dir = str(Path(box_path).parent).replace('boxes', 'commentary').replace('/data/', '/commentary/')
        #             if result_dir not in data_results_paths:
        #                 self.data_boxes_paths.append(box_path)
        #             if result_dir in data_results_paths:
        #                 result_file = box_path.replace('boxes', 'commentary').replace('/data/', '/commentary/')
        #                 # check if the result file exists
        #                 if not os.access(result_file, os.F_OK):
        #                     self.data_boxes_paths.append(box_path) 

        #     # self.data_boxes_paths = [x for x in self.data_boxes_paths if str(Path(x).parent).replace('boxes', 'vqa').replace('/data/', '/drivelm/') not in self.data_results_paths]
        #     print(f"Number of boxes paths after removing existing commentary files: {len(self.data_boxes_paths)}")
        # else:
        self.data_boxes_paths = self.data_boxes_paths_all

        # Randomly sample a subset of data (if random_subset_count > 0)
        if self.random_subset_count > 0:
            random.seed(42)
            random.shuffle(self.data_boxes_paths)
            self.data_boxes_paths = self.data_boxes_paths[:self.random_subset_count]
        # self.data_boxes_paths = self.data_boxes_paths[14400:]

        self.data_boxes_paths = list(sorted(self.data_boxes_paths))

        # Load templates
        template_file = f"data/augmented_templates/commentary.json"
        with open(template_file, 'r') as f:
            self.templates = ujson.load(f)

        self.list_next_junction_id_minus_one = []

        self.all_labels = []
        self.stats = {
            'total_frames': 0,
            'frames_per_scenario': {},
            'num_visible_objects': 0,
            'num_not_visible_objects': 0
        }
        self.all_templates = []

        # Load keyframes list if sampling keyframes
        if self.sample_frame_mode == 'keyframes':
            keyframes_list_path = self.path_keyframes
            with open(keyframes_list_path, 'r', encoding="utf-8") as f:
                self.keyframes_list = f.readlines()
            self.keyframes_list = [x.strip() for x in self.keyframes_list]
            self.keyframes_list = [x.replace('rgb', 'boxes').replace('.jpg', '.json.gz') for x in self.keyframes_list]

    def create_commentary(self, path_id):
        """
        Create all commentary in llava format, convert them to NuScenes afterwards and finally save them
        """


        # Process each frame
        # for path in tqdm.tqdm(self.data_boxes_paths):
        path = self.data_boxes_paths[path_id]

        if self.skip_existing:
            save_dir = (self.output_directory + "/" + path.split("/data/")[1]).replace('boxes', 'commentary')
            if os.access(save_dir, os.F_OK):
                return

        path_measurements = path.replace('boxes', 'measurements')
        if 'lb1_split' in path_measurements:
            # database/simlingo/data/simlingo/lb1_split/routes_training/ControlLoss/Town01_Rep0_Town01_Scenario1_0_route0_01_11_14_15_07/measurements/0000.json.gz
            scenario_name = path_measurements.split('lb1_split/')[-1].split('/')[1]
            if scenario_name == 'noScenarios':
                scenario_name = None
            route_file_number = re.search(rf'Rep*(\d+)_Town[\d\w]*_[\d\w]*_*(\d+)_route_*(\d+)', path_measurements).group(2)
            route_number = re.search(rf'Rep*(\d+)_Town[\d\w]*_[\d\w]*_*(\d+)_route_*(\d+)', path_measurements).group(3)
        elif 'parking_lane' in path_measurements:
            scenario_name = None
            route_file_number = re.search(rf'Rep*(\d+)_*(\d+)_route_*(\d+)', path_measurements).group(2)
            route_number = re.search(rf'Rep*(\d+)_*(\d+)_route_*(\d+)', path_measurements).group(3)
        else:
            scenario_name = get_scenario_name(path_measurements)
            route_file_number = re.search(rf'Rep*(\d+)_*(\d+)_route_*(\d+)', path_measurements).group(2)
            route_number = re.search(rf'Rep*(\d+)_*(\d+)_route_*(\d+)', path_measurements).group(3)

        route_folder = path_measurements.split('simlingo/')[-1].split('/Town')[0].split('/')
        route_folder = '_'.join(route_folder)

        # Skip frames if RGB image does not exist
        if not os.path.isfile(path.replace('boxes', 'rgb').replace('.json.gz', '.jpg')):
            return

        # Skip frames based on keyframes list
        if self.sample_frame_mode == 'keyframes':
            if path not in self.keyframes_list:
                return

        frame_number = int(path.split('/')[-1].split('.')[0])

        # Skip first 10 frames (those get skipped during training anyways)
        if frame_number < self.SKIP_FIRST_FRAMES:
            return

        # Skip frames if sampling uniformly and frame number does not match
        if self.sample_frame_mode == 'uniform' and frame_number % self.sample_uniform_interval != 0:
            return

        # Check if files exist
        if not os.path.exists(path):
            return
        if not os.path.exists(path_measurements):
            return

        # Read results file
        if self.filter_routes_by_result:
            results_file = path.split('boxes')[0] + 'results.json.gz'

            try:
                with gzip.open(results_file, 'rb') as f:
                    route_results = json.loads(f.read().decode('utf-8'))
            except:
                print(f"Error reading {results_file}")
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
        with gzip.open(path, 'rb') as f:
            file_content = f.read()
            current_boxes = json.loads(file_content.decode('utf-8'))

        with gzip.open(path_measurements, 'rb') as f:
            file_content = f.read()
            current_measurements = json.loads(file_content.decode('utf-8'))

        # load self.HISTORY_LEN frames of measurements
        last_measurement_paths = [path_measurements]
        last_measurements = [current_measurements]
        for i in range(1, self.HISTORY_LEN):
            #replace frame_number with frame_number - i as 4 digit with leading zeros
            frame_number_minus_i = str(frame_number - i).zfill(4)
            frame_number_org_str = str(frame_number).zfill(4)
            last_measurement_paths.append(path_measurements.replace(f'{frame_number_org_str}.json', f'{frame_number_minus_i}.json'))

            with gzip.open(last_measurement_paths[-1], 'rb') as f:
                file_content = f.read()
                last_measurements.append(json.loads(file_content.decode('utf-8')))

        last_measurements_oldest_first = last_measurements[::-1]

        future_measurement_paths = [path_measurements]
        future_measurements = [current_measurements]
        skip = False
        for i in range(0, self.FUTURE_LEN):
            frame_number_plus_i = str(frame_number + i).zfill(4)
            frame_number_org_str = str(frame_number).zfill(4)
            future_measurement_paths.append(path_measurements.replace(f'{frame_number_org_str}.json', f'{frame_number_plus_i}.json'))

            if not os.path.exists(future_measurement_paths[-1]):
                skip = True
                break
            with gzip.open(future_measurement_paths[-1], 'rb') as f:
                file_content = f.read()
                future_measurements.append(json.loads(file_content.decode('utf-8')))
        if skip:
            return

        # Get perception questions
        image_path = path.replace('boxes', 'rgb').replace('.json.gz', '.jpg')
        relative_image_path = image_path
        self.current_path = image_path

        try:
            commentary, cause_object_visible_in_image, cause_object, cause_object_str = self.generate_commentary(current_boxes, last_measurements_oldest_first, future_measurements, scenario_name)
        except Exception as e:
            print(f"Error in {path_measurements}: {e}")
            # save all failed paths in list
            with open(f"{path.split('/data/')[0]}/commentary/failed_paths.txt", "a") as f:
                f.write(f"{path_measurements}\n")
            return


        # Save examples if specified
        if self.save_examples:
            # Load and draw on the image
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)

            # black image to be concatenated to the bottom of the image
            img_width = image.size[0]
            text_box = Image.new("RGB", (img_width, 200), "black")
            text_draw = ImageDraw.Draw(text_box)

            path_orgimg = f'{self.output_examples_directory}/com_viz/{scenario_name}/{route_folder}/{route_file_number}_{route_number}'
            Path(path_orgimg).mkdir(parents=True, exist_ok=True)

            assert image.width == self.ORIGINAL_IMAGE_SIZE[0], f'{image.width} != {self.ORIGINAL_IMAGE_SIZE[0]}'
            assert image.height == self.ORIGINAL_IMAGE_SIZE[1], f'{image.height} != {self.ORIGINAL_IMAGE_SIZE[1]}'
            
            # Draw a point for each object (e.g, car, traffic light, ...) on the image
            if self.visualize_projection:
                for single_object in current_boxes:
                    if 'position' in single_object:
                        if single_object['class'] == 'ego_car':
                            continue
                        if 'num_points' in single_object and single_object['num_points'] < 50:
                            continue
                        if single_object['position'][0] < -1.0:
                            continue
                        all_points_2d, _ = project_all_corners(single_object, self.CAMERA_MATRIX)

                        if 'car' in single_object['class']:
                            color = (255, 0, 0, 0)
                        else:
                            continue
                        # elif 'traffic_light' in single_object['class'] or 'stop' in single_object['class']:
                        #     color = (0, 255, 0, 0)
                        # elif 'landmark' in single_object['class']:
                        #     color = (0, 0, 255, 0)
                        # else:
                        #     color = (0, 0, 0, 0)
                        if all_points_2d is not None:
                            for points_2d in all_points_2d:
                                draw.ellipse((points_2d[0]-5, points_2d[1]-5, points_2d[0]+5, points_2d[1]+5), 
                                                                                                    fill=color)

            # Draw commentary on the image
            text = commentary
            text = textwrap.fill(text, width=80)

            text_draw.text((10, 10), text, fill=(255, 255, 255), font=ImageFont.truetype("arial.ttf", 20))

            # write cause object on the image
            if cause_object is not None:
                if cause_object['class'] == 'car':
                    text = f"Cause object: {cause_object['color_name']}, {cause_object['type_id']}, {cause_object['position']}"
                else:
                    text = f"Cause object: {cause_object['class']}, {cause_object['position']}"
                text_draw.text((10, 100), text, fill=(255, 255, 255), font=ImageFont.truetype("arial.ttf", 20))
                text_draw.text((10, 130), f'Cause object visible in image: {cause_object_visible_in_image}', fill=(255, 255, 255), font=ImageFont.truetype("arial.ttf", 20))

            # Concatenate the text box to the bottom of the image
            concat_image = Image.fromarray(np.concatenate((np.array(image), np.array(text_box)), axis=0))

            # Save concat image
            concat_image.save(f'{path_orgimg}/{frame_number}.jpg')

        if cause_object_str != "" and cause_object_str is not None:
            template = commentary.replace(cause_object_str, '<OBJECT>')
            placeholder = {
                "<OBJECT>": cause_object_str,
            }
        else:
            template = commentary
            placeholder = {}
        if re.search(r'in \d+\.\d+ meters', commentary) is not None:
            template = re.sub(r'in \d+\.\d+ meters', 'in <DISTANCE>', template)
            placeholder['<DISTANCE>'] = re.search(r'in \d+\.\d+ meters', commentary).group(0)
        if re.search(r'in -\d+\.\d+ meters', commentary) is not None:
            template = re.sub(r'in -\d+\.\d+ meters', 'in <DISTANCE>', template)
            placeholder['<DISTANCE>'] = re.search(r'in -\d+\.\d+ meters', commentary).group(0)
        if re.search(r'at \d+\.\d+ meters', commentary) is not None:
            template = re.sub(r'at \d+\.\d+ meters', 'at <DISTANCE>', template)
            placeholder['<DISTANCE>'] = re.search(r'at \d+\.\d+ meters', commentary).group(0)
        if re.search(r'at -\d+\.\d+ meters', commentary) is not None:
            template = re.sub(r'at -\d+\.\d+ meters', 'at <DISTANCE>', template)
            placeholder['<DISTANCE>'] = re.search(r'at -\d+\.\d+ meters', commentary).group(0)

        if template not in self.all_templates:
            self.all_templates.append(template)

        commentary_data = {}
        commentary_data['image'] = relative_image_path
        commentary_data['commentary'] = commentary
        commentary_data['commentary_template'] = template
        commentary_data['cause_object_visible_in_image'] = cause_object_visible_in_image
        commentary_data['cause_object'] = cause_object
        commentary_data['cause_object_string'] = cause_object_str
        commentary_data['scenario_name'] = scenario_name
        commentary_data['placeholder'] = placeholder

        # easier to debug:
        save_dir = (self.output_directory + "/" + path.split("/data/")[1]).replace('boxes', 'commentary')
        # final version
        # save_dir = path.replace('/rgb/', '/commentary/').replace('.jpg', '.json')
        Path(save_dir).parent.mkdir(exist_ok=True, parents=True)
        # json.gz
        with gzip.open(save_dir, 'wt', encoding='utf-8') as f:
            json.dump(commentary_data, f, indent=4)

        # with open(save_dir, 'w', encoding='utf-8') as f:
            # json.dump(commentary_data, f, indent=4)

            
        self.stats['total_frames'] += 1
        if scenario_name not in self.stats['frames_per_scenario']:
            self.stats['frames_per_scenario'][scenario_name] = 1
        else:
            self.stats['frames_per_scenario'][scenario_name] += 1
        
        if cause_object_visible_in_image:
            self.stats['num_visible_objects'] += 1
        else:
            self.stats['num_not_visible_objects'] += 1
    
    def save_stats(self):
        """
        Save stats to a json file
        """
        
        self.stats['num_different_templates'] = len(self.all_templates)
        self.stats['all_templates'] = self.all_templates

        # Save stats
        with open(f'{self.output_directory}/stats.json', 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=4)

        return

    
    def generate_commentary(self, current_boxes, last_measurements_oldest_first, future_measurements, scenario_name):
        """
        """

        augment_commentary = False
        
        # sort boxes by ID
        boxes_by_id = {}
        ego_info_box = None
        dynamic_boxes = []
        walker_boxes = []

        traffic_light = None
        cause_object_visible_in_image = True
        cause_object = None

        current_measurement = last_measurements_oldest_first[-1]
        route_original = current_measurement['route_original']
        route_adjusted = current_measurement['route']

        waypoints = self.get_waypoints(future_measurements,
                                            y_augmentation=0,
                                            yaw_augmentation=0)
        waypoints = np.array(waypoints[1:-1])

        for box in current_boxes:
            if box['class'] == 'ego_info':
                ego_info_box = box
            if box['class'] == 'ego_car':
                ego_box = box

            if 'id' not in box:
                continue

            box['id'] = int(box['id'])
            boxes_by_id[box['id']] = box

            if box['class'] == 'car' or box['class'] == 'walker':
                dynamic_boxes.append(box)
            if box['class'] == 'traffic_light' and box['affects_ego']:
                traffic_light = (box['state'], box['distance'])
            if box['class'] == 'walker':
                walker_boxes.append(box)

        # if the original route is more than 2.5 meters away from the ego vehicle 
        # we assume we are on the adjusted route
        on_adjusted = np.linalg.norm(route_original[0][1]) > 2.5 

        target_speed = current_measurement['target_speed']
        current_speed = current_measurement['speed']
        speed_limit = current_measurement['speed_limit']

        vehicle_hazard = current_measurement['vehicle_hazard']
        vehicle_hazard_id = current_measurement['vehicle_affecting_id']

        # only consider walker hazard if number of lidar hits is above 3
        walker_hazard = False
        if current_measurement['walker_hazard']:
            for walker in walker_boxes:
                if walker['num_points'] > 3:
                    walker_hazard = current_measurement['walker_hazard']

        walker_close_id = current_measurement['walker_close_id']
        walker_close_box = boxes_by_id.get(walker_close_id, None)
        walker_attention = False
        if walker_close_box is not None or len(walker_boxes) > 0:
            if walker_close_box is not None:
                if walker_close_box['distance'] < 15 and walker_close_box['speed'] > 0.5 and walker_close_box['num_points'] > 3:
                    walker_attention = True
            else:
                for walker in walker_boxes:
                    if walker['distance'] < 15 and walker['speed'] > 0.5 and walker['num_points'] > 3:
                        walker_attention = True

        wps = waypoints
        mean_x = np.mean(wps[:, 0])

        # last measurements
        past_changed_route = False
        past_changed_routes = []
        for i in range(self.HISTORY_LEN-1):
            past_changed_routes.append(last_measurements_oldest_first[i]['changed_route'])

        if any(past_changed_routes):
            past_changed_route = True
        
        future_changed_route = False
        future_org_route = False
        future_changed_routes = []
        # check if route changes in the next 10 m
        for i in range(10):
            dist = np.linalg.norm(np.asarray(current_measurement['route'][i]) - np.asarray(current_measurement['route_original'][i]))
            if dist < 0.6 and (current_measurement['changed_route'] or any(future_changed_routes)):
                future_org_route = True
            
            if dist > 0.6:
                future_changed_route = True
        
        future_changed_route_far = False
        for i in range(20):
            dist = np.linalg.norm(np.asarray(current_measurement['route'][i]) - np.asarray(current_measurement['route_original'][i]))
            if dist > 0.6:
                future_changed_route_far = True

        changed_route = current_measurement['changed_route'] or any(future_changed_routes)
        reason_speed = '.'
        action_route = 'Follow the route.'

        distance_to_target_point = np.sqrt(current_measurement['target_point'][0]**2 + current_measurement['target_point'][1]**2)
        y_distance_to_closest_route_point = abs(current_measurement['route'][0][1])
        if scenario_name == 'ParkingExit' and y_distance_to_closest_route_point > 1.5:
            action_route = 'Exit the parking lot.'
            future_changed_route = True
        elif current_measurement['command'] == 5 or current_measurement['command'] == 6:
            if distance_to_target_point < 20:
                action_route = 'Prepare to do a lane change.'
            elif distance_to_target_point < 10:
                action_route = 'Do a lane change.'
        elif current_measurement['command'] == 1:
            if distance_to_target_point < 20:
                action_route = 'Turn left.'
        elif current_measurement['command'] == 2:
            if distance_to_target_point < 20:
                action_route = 'Turn right.'

        active_scenario = scenario_name

        speed_reduced_by_obj_type = current_measurement['speed_reduced_by_obj_type']
        speed_reduced_by_obj_id = current_measurement['speed_reduced_by_obj_id']
        decreased_target_speed_due_to_object_box = boxes_by_id.get(speed_reduced_by_obj_id, None)
        speed_reduced_by_obj_distance = current_measurement['speed_reduced_by_obj_distance']
        object_is_vehicle = False

        if current_measurement['stop_sign_hazard']:
            speed_reduced_by_obj_type = 'traffic.stop'
            speed_reduced_by_obj_id = -1
            decreased_target_speed_due_to_object_box = None
            if ego_info_box['distance_to_junction'] is not None:
                    speed_reduced_by_obj_distance = ego_info_box['distance_to_junction'] - 5
            else:
                speed_reduced_by_obj_distance = None
        elif current_measurement['light_hazard'] and (ego_info_box['distance_to_junction'] is not None and ego_info_box['distance_to_junction'] < 40):
            speed_reduced_by_obj_type = 'traffic_light'
            speed_reduced_by_obj_id = -1
            decreased_target_speed_due_to_object_box = None
            if ego_info_box['distance_to_junction'] is not None:
                    speed_reduced_by_obj_distance = ego_info_box['distance_to_junction'] - 5
            else:
                speed_reduced_by_obj_distance = None

        if speed_reduced_by_obj_distance is not None:
            speed_reduced_by_obj_distance = round(speed_reduced_by_obj_distance, 1)
            if decreased_target_speed_due_to_object_box is None and speed_reduced_by_obj_type is not None:
                if speed_reduced_by_obj_type == 'traffic.stop' or current_measurement['stop_sign_hazard']:
                    object_appearance = 'stop sign'
                elif 'vehicle' in speed_reduced_by_obj_type:
                    object_appearance = 'vehicle'
                elif 'walker' in speed_reduced_by_obj_type:
                    object_appearance = 'walker'
                elif 'traffic_light' in speed_reduced_by_obj_type:
                    object_appearance = 'red traffic light'
                elif 'trafficwarning' in speed_reduced_by_obj_type:
                    object_appearance = 'construction site'
                else:
                    raise ValueError(f"Object type {speed_reduced_by_obj_type} not found.")
            else:
                object_appearance = get_vehicle_appearance_string(decreased_target_speed_due_to_object_box)
                if 'vehicle' in speed_reduced_by_obj_type or 'walker' in speed_reduced_by_obj_type:
                    object_is_vehicle = True
                    cause_object_visible_in_image = is_vehicle_visible_in_image(decreased_target_speed_due_to_object_box, self.MIN_X, self.MAX_X, self.MIN_Y, self.MAX_Y, self.CAMERA_MATRIX) \
                          and decreased_target_speed_due_to_object_box['num_points'] > 3 \
                          and decreased_target_speed_due_to_object_box['position'][0] > -1.5

            if speed_reduced_by_obj_distance > 40:
                speed_reduced_by_obj_type = None
                speed_reduced_by_obj_id = None
                decreased_target_speed_due_to_object_box = None
                speed_reduced_by_obj_distance = None
                object_appearance = ""
        else:
            speed_reduced_by_obj_type = None
            speed_reduced_by_obj_id = None
            decreased_target_speed_due_to_object_box = None
            speed_reduced_by_obj_distance = None
            object_appearance = ""

        # manual check for traffic light as they sometimes get missed
        cause_object_at_traffic_light = False
        for box in current_boxes:
            if box['class'] == 'traffic_light' and box['affects_ego'] and box['distance'] < 40 and box['state'] == 'Red':
                if decreased_target_speed_due_to_object_box is None:
                    speed_reduced_by_obj_type = 'traffic_light'
                    speed_reduced_by_obj_id = box['id']
                    decreased_target_speed_due_to_object_box = box
                    speed_reduced_by_obj_distance = box['distance']
                    object_appearance = 'red traffic light'

                if 'traffic_light_state' in decreased_target_speed_due_to_object_box and decreased_target_speed_due_to_object_box['traffic_light_state'] == 'Red':
                    cause_object_at_traffic_light = True
                break
        
        if speed_reduced_by_obj_distance is not None:
            speed_reduced_by_obj_distance = round(speed_reduced_by_obj_distance, 1)
        cause_object = decreased_target_speed_due_to_object_box

        # get language description of action and reason
        # accelerate vs decelerate
        future_target_speeds = [future_measurements[i]['target_speed'] for i in range(self.FUTURE_LEN)]
        avg_future_target_speeds = np.mean(future_target_speeds[:5])

        maintain = False
        accelerate = False
        decelerate = False
        stopped_ego = False
        if current_speed < 0.2 and abs(current_speed - avg_future_target_speeds) < 0.5:
            if augment_commentary:
                action_speed = random.choice(self.templates['remain_stopped'])
            else:
                action_speed = self.templates['remain_stopped'][0]
            stopped_ego = True
        elif abs(current_speed - avg_future_target_speeds) < 0.5:
            if avg_future_target_speeds < 0.2:
                if augment_commentary:
                    action_speed = random.choice(self.templates['stop_now'])
                else:
                    action_speed = self.templates['stop_now'][0]
            elif avg_future_target_speeds/speed_limit > 0.71:
                maintain = True
                if augment_commentary:
                    action_speed = random.choice(self.templates['maintain_speed'])
                else:
                    action_speed = self.templates['maintain_speed'][0]
            else: 
                maintain = True
                if augment_commentary:
                    action_speed = random.choice(self.templates['maintain_reduced_speed'])
                else:
                    action_speed = self.templates['maintain_reduced_speed'][0]
        elif current_speed < avg_future_target_speeds:
            if augment_commentary:
                action_speed = random.choice(self.templates['accelerate'])
            else:
                action_speed = self.templates['accelerate'][0]
            accelerate = True
        else:
            if augment_commentary:
                action_speed = random.choice(self.templates['decelerate'])
            else:
                action_speed = self.templates['decelerate'][0]
            decelerate = True

        junction_reason = None
        stopped = None
        in_junction = False
        if ego_info_box['distance_to_junction'] is None:
            ego_info_box['distance_to_junction'] = 9999
        if ego_info_box['is_in_junction'] or ego_info_box['is_intersection'] or ego_info_box['distance_to_junction']<15:
            for dynamic_agent in dynamic_boxes:
                # HACK: this does not work e.g. with multiple cameras
                visible = is_vehicle_visible_in_image(dynamic_agent, self.MIN_X, self.MAX_X, self.MIN_Y, self.MAX_Y, self.CAMERA_MATRIX) and dynamic_agent['num_points'] > 3 and dynamic_agent['position'][0] > -1.5
                if not visible:
                    continue
                if 'is_in_junction' in dynamic_agent and dynamic_agent['is_in_junction']:
                    in_junction = True
                if dynamic_agent['class'] == 'walker':
                    continue
                pointing_towards_junction, _ = is_pointing_towards_junction(ego_info_box, dynamic_agent)
                if not pointing_towards_junction:
                    continue
                if dynamic_agent['distance'] > dynamic_agent['speed']*7 and not (dynamic_agent['brake'] > 0.8 or dynamic_agent['speed'] < 1.0):
                    continue
                if pointing_towards_junction and (dynamic_agent['brake'] > 0.8 or dynamic_agent['speed'] < 1.0):
                    stopped = True
                else:
                    stopped = False
            if stopped:
                if not in_junction:
                    junction_reason = 'the other vehicles are stopped at the junction and the junction is clear'
                else:
                    junction_reason = 'the other vehicles are stopped at the junction and the vehicle in the junction is moving away'
            elif stopped is None:
                junction_reason = ''
            else:
                if not in_junction:
                    junction_reason = 'pay attention to the vehicles coming towards the junction'
                else:
                    junction_reason = 'pay attention to the vehicle in the junction'

        if speed_reduced_by_obj_type == 'traffic.stop' and current_speed < target_speed and target_speed > 0.4 and abs(current_speed - target_speed) > 0.5 and speed_reduced_by_obj_distance is not None and speed_reduced_by_obj_distance < 5:
            if augment_commentary:
                cleared_stop = random.choice(self.templates['cleared_stop'])
            else:
                cleared_stop = self.templates['cleared_stop'][0]
            cleared_stop_lower = cleared_stop[0].lower() + cleared_stop[1:]
            if junction_reason == '':
                reason_speed = f' since {cleared_stop_lower}.'
            elif not stopped:
                reason_speed = f' since {cleared_stop_lower} but {junction_reason}.'
            else:
                reason_speed = f' since {cleared_stop_lower} and {junction_reason}.'

        elif vehicle_hazard and (boxes_by_id.get(vehicle_hazard_id, None) is not None and decreased_target_speed_due_to_object_box is not None and boxes_by_id.get(vehicle_hazard_id, None)['distance'] <= (decreased_target_speed_due_to_object_box['distance']-5)):

            vehicle_hazard_box = boxes_by_id.get(vehicle_hazard_id, None)
            object_appearance = get_vehicle_appearance_string(vehicle_hazard_box)
            cause_object_visible_in_image = is_vehicle_visible_in_image(vehicle_hazard_box, self.MIN_X, self.MAX_X, self.MIN_Y, self.MAX_Y, self.CAMERA_MATRIX) \
                          and decreased_target_speed_due_to_object_box['num_points'] > 3 \
                          and decreased_target_speed_due_to_object_box['position'][0] > -1.5
                          
            cause_object = vehicle_hazard_box

            if augment_commentary:
                prevent_collision = random.choice(self.templates['prevent_collision'])
            else:
                prevent_collision = self.templates['prevent_collision'][0]
            prevent_collision_lower = prevent_collision[0].lower() + prevent_collision[1:]
            if vehicle_hazard_box is not None:
                reason_speed = f' to {prevent_collision_lower} the {object_appearance}.'
            else:
                reason_speed = f' to {prevent_collision_lower} a vehicle.'

        elif walker_hazard:
            # if the walker is behind the next junction:
            if ego_info_box['distance_to_junction'] is not None and speed_reduced_by_obj_distance > ego_info_box['distance_to_junction'] + 5 and accelerate:
                reason_speed = f' to drive through the junction as {junction_reason}. Pay attention to the pedestrian on the exit of the junction.'
            else:
                if augment_commentary:
                    cross_path = random.choice(self.templates['cross_path'])
                else:
                    cross_path = self.templates['cross_path'][0]
                cross_path_lower = cross_path[0].lower() + cross_path[1:]
                
                reason_speed = f" due to the {object_appearance} {cross_path_lower}." # (speed:{boxes_by_id[speed_reduced_by_obj_id]['speed']})"

        elif speed_reduced_by_obj_type is not None:
            if augment_commentary:
                stay_behind = random.choice(self.templates['stay_behind'])
            else:
                stay_behind = self.templates['stay_behind'][0]
            stay_behind_lower = stay_behind[0].lower() + stay_behind[1:]
            if augment_commentary:
                drive_closer = random.choice(self.templates['drive_closer'])
            else:
                drive_closer = self.templates['drive_closer'][0]
            drive_closer_lower = drive_closer[0].lower() + drive_closer[1:]
            if 'vehicle' in speed_reduced_by_obj_type:
                if active_scenario is not None and not 'HazardAtSideLane' in active_scenario and decreased_target_speed_due_to_object_box['base_type'] == 'bicycle' and decreased_target_speed_due_to_object_box['role_name'] == 'scenario':
                    reason_speed = f' to avoid the {object_appearance} that is crossing the road.'
                elif cause_object_at_traffic_light and current_speed < 0.2 and not accelerate:
                    if random.random() < 0.5:
                        reason_speed = f' to {stay_behind_lower} the {object_appearance} at {speed_reduced_by_obj_distance} meters that is stopped because of a red traffic light.'
                    else:
                        reason_speed = f' to {stay_behind_lower} the {object_appearance} that is stopped because of a red traffic light.'
                elif cause_object_at_traffic_light and not accelerate:
                    if random.random() < 0.5:
                        reason_speed = f' to {stay_behind_lower} the {object_appearance} at {speed_reduced_by_obj_distance} meters that is slowing down because of a red traffic light.'
                    else:
                        reason_speed = f' to {stay_behind_lower} the {object_appearance} that is slowing down because of a red traffic light.'
                else:
                    if not decelerate and target_speed > 0.5 and decreased_target_speed_due_to_object_box['distance'] < 12 and decreased_target_speed_due_to_object_box['speed'] < 0.5:
                        if random.random() < 0.5:
                            reason_speed = f' to {drive_closer_lower} the stationary {object_appearance} at {speed_reduced_by_obj_distance} meters.'
                        else:
                            reason_speed = f' to {drive_closer_lower} the stationary {object_appearance}.'
                    elif accelerate:
                        if random.random() < 0.5:
                            reason_speed = f' to follow the {object_appearance} in {speed_reduced_by_obj_distance} meters.'
                        else:
                            reason_speed = f' to follow the {object_appearance}.'
                    else:
                        if random.random() < 0.5:
                            reason_speed = f' to {stay_behind_lower} the {object_appearance} in {speed_reduced_by_obj_distance} meters.'
                        else:
                            reason_speed = f' to {stay_behind_lower} the {object_appearance}.'
            else:
                if accelerate:
                    if speed_reduced_by_obj_distance is None:
                        reason_speed = f' to {drive_closer_lower} the {object_appearance}.'
                    elif speed_reduced_by_obj_distance > 5 and speed_reduced_by_obj_distance < 15:
                        if random.random() < 0.5:
                            reason_speed = f' to {drive_closer_lower} the {object_appearance} in {speed_reduced_by_obj_distance} meters.'
                        else:
                            reason_speed = f' to {drive_closer_lower} the {object_appearance}.'
                    else: 
                        reason_speed = ' to reach the speed limit.'
                else:
                    if speed_reduced_by_obj_distance is None:
                        reason_speed = f' due to the {object_appearance}.'
                    elif speed_reduced_by_obj_distance < 20:
                        if random.random() < 0.5:
                            reason_speed = f' due to the {object_appearance} in {speed_reduced_by_obj_distance} meters.'
                        else:
                            reason_speed = f' due to the {object_appearance}.'
                    else:
                        reason_speed = ' to drive according to the speed limit.'
        else:
            if traffic_light is not None and traffic_light[0] == 'Green':
                reason_speed = ' because the traffic light is green'
                if junction_reason is not None:
                    if stopped:
                        reason_speed += f' and {junction_reason}.'
                    else:
                        reason_speed += f' but pay attention to the vehicle coming towards the junction.'
                else:
                    reason_speed += '.'
            else:
                if ego_info_box['is_in_junction']:
                    if not decelerate and not stopped_ego:
                        reason_speed = ' to drive through the junction'
                    else:
                        reason_speed = ''
                else:
                    if maintain:
                        reason_speed = ''
                    else:
                        reason_speed = ' to drive with the target speed'
                if junction_reason is not None and stopped:
                    reason_speed += f' because {junction_reason}.'
                else:
                    reason_speed += '.'

        if changed_route or future_changed_route or future_changed_route_far:
            if active_scenario is not None and "TwoWays" in active_scenario:
                route_change = " to the lane with oncoming traffic"
            else:
                route_change = ""
            
            active_scenario_base = active_scenario.replace('TwoWays','')
            if active_scenario is None:
                reason_route = "."
            elif active_scenario_base in self.templates:
                if augment_commentary:
                    action_route = random.choice(self.templates[active_scenario_base])
                else:
                    if future_changed_route:
                        action_route = self.templates[active_scenario_base][0]
                    elif future_changed_route_far:
                        if cause_object is not None and cause_object['class'] == "car" and cause_object['same_road_as_ego'] and cause_object['lane_relative_to_ego'] == 0:
                            pass
                        else:
                            action_route = self.templates[active_scenario_base][0]
                            #lower case first letter
                            action_route = action_route[0].lower() + action_route[1:]
                            action_route = "Prepare to " + action_route
            else:
                if scenario_name != 'ParkingExit':
                    action_route = ""

            if changed_route or future_changed_route:
                if "TwoWays" in active_scenario:
                    lane_info = 'oncoming lane'
                else:
                    lane_info = 'neighbouring lane'

                if 'InvadingTurn' in active_scenario:
                    if augment_commentary:
                        shift = random.choice(self.templates['shift_right'])
                    else:
                        shift = self.templates['shift_right'][0]
                    route_change_lower = action_route[0].lower() + action_route[1:]
                    action_route = shift + ' to make space for the traffic that invades the lane because of the traffic cones.'
                elif (mean_x < 0.5 or target_speed < 0.2) and (not on_adjusted or not past_changed_route) and future_changed_route:
                    if augment_commentary:
                        wait_gap = random.choice(self.templates['wait_gap'])
                    else:
                        wait_gap = self.templates['wait_gap'][0]
                    action_speed = wait_gap + route_change
                    reason_speed = ''
                
                elif not on_adjusted and not future_org_route:
                    if augment_commentary:
                        gap_big = random.choice(self.templates['gap_big'])
                    else:
                        gap_big = self.templates['gap_big'][0]
                    reason_speed = f' to change to the {lane_info}, {gap_big}'

                elif future_org_route:
                    if augment_commentary:
                        go_back = random.choice(self.templates['go_back'])
                    else:
                        go_back = self.templates['go_back'][0]
                    action_route = go_back
                else:
                    if "TwoWays" in active_scenario:
                        lane_info = 'your current (oncoming) lane'
                    else:
                        lane_info = 'your current lane'
                    action_route_lowercase = action_route[0].lower() + action_route[1:]
                    action_route = f'Stay on {lane_info} to {action_route_lowercase}'

        post_comment = ''
        if walker_attention and (speed_reduced_by_obj_type is None or not 'walker' in speed_reduced_by_obj_type):
            post_comment = ' Pay attention to the walker and brake if necessary.'
        
        commentary = f'{action_route} {action_speed}{reason_speed}{post_comment}.'
        commentary = commentary.replace('..', '.').replace('  ', ' ').replace('...', '.')

        return commentary, cause_object_visible_in_image, cause_object, object_appearance


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
        rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)]])
        translation = np.array([[0.0], [y_augmentation]])
        for waypoint in waypoints:
            pos = np.expand_dims(waypoint, axis=1)
            waypoint_aug = rotation_matrix.T @ (pos - translation)
            waypoints_aug.append(np.squeeze(waypoint_aug))

        return waypoints_aug