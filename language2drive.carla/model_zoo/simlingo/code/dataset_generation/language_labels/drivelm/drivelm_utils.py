import numpy as np
import cv2
import re
import xml.etree.ElementTree as ET
import lingo_pretraining.utils.transfuser_utils as t_u
import json


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def project_points(obj, K):
    # do this check before calling function since it depends on if bike or car etc.
    # if ('num_points' in obj and obj['num_points'] < 5) or obj['position'][0] < -1:
    #     return None


    pos = obj['position']
    if 'extent' not in obj:
        extent = [0.15,0.15,0.15]
    else:
        extent = obj['extent']
    if 'yaw' not in obj:
        yaw = 0
    else:
        yaw = -obj['yaw']
        
    # get bbox corners coordinates
    corners = np.array([[-extent[0], 0, 0.75],
                        [extent[0], 0, 0.75]])
    # rotate bbox
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                [np.sin(yaw), np.cos(yaw), 0],
                                [0, 0, 1]])
    corners = np.dot(corners, rotation_matrix)
    # translate bbox
    corners = corners + np.array(pos)
    all_points_2d = []
    for corner in  corners:
        pos_3d = np.array([corner[1], -corner[2], corner[0]])
        rvec = np.zeros((3, 1), np.float32) 
        tvec = np.array([[0.0, 2.0, 1.5]], np.float32)
        # Define the distortion coefficients 
        dist_coeffs = np.zeros((5, 1), np.float32) 
        points_2d, _ = cv2.projectPoints(pos_3d, 
                            rvec, tvec, 
                            K, 
                            dist_coeffs)
        all_points_2d.append(points_2d[0][0])
        
    return all_points_2d


def light_state_to_word(light_state):
    '''
    0: NONE
    All lights off.
    1: Position
    2: LowBeam
    3: HighBeam
    4: Brake
    5: RightBlinker
    6: LeftBlinker
    7: Reverse
    8: Fog
    9: Interior
    10: Special1
    This is reserved for certain vehicles that can have special lights, like a siren.
    11: Special2
    This is reserved for certain vehicles that can have special lights, like a siren.
    12: All
    All lights on.

    '''
    light_state_dict = {0: 'None', 1: 'position light', 2: 'low beam', 3: 'high beam', 4: 'brake light', 5: 'right blinker',
                        6: 'left blinker', 7: 'reverse light', 8: 'fog light', 9: 'interior light', 10: 'emergency lights', 11: 'emergency lights',
                        12: 'All'}
    # add "the" in front of the light state
    light_state = light_state_dict[light_state]
    light_state = 'the ' + light_state
    return light_state

def logical_xor(str1, str2):
    return bool(str1) ^ bool(str2)
    

def a_or_an(word):
    """
    Returns 'a' or 'an' depending on whether the word starts with a vowel or not.
    :param word: string
    :return: 'a' or 'an'
    """
    vowels = ['a', 'e', 'i', 'o', 'u']
    if word[0].lower() in vowels:
        return 'an'
    else:
        return 'a'

def number_to_word(number):
    """
    Returns the number as a word.
    :param number: int
    :return: string
    """
    number_dict = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                   6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
    return number_dict[number]

def get_scenario_name(measurement_file_current):

    with open(measurement_file_current, 'r') as f:
        current_measurement = json.load(f)

    route_folder = re.search(r'.?_scenario_per_route_.*_routes_per_file_.*_scenarioss', measurement_file_current)
    if route_folder is None:
        route_folder = re.search(r'custom_parkinglane', measurement_file_current)
        if route_folder is None:
            route_folder = re.search(r'pedestrians', measurement_file_current)
            if route_folder is None:
                route_folder = re.search(r'OpensDoor', measurement_file_current)
    route_folder = route_folder.group(0)
    # if route_folder == 'custom_parkinglane':
    seed = re.search(r'random_weather.*_seed_\d+', measurement_file_current).group(0)
    if 'routeall' in measurement_file_current:
        route_file = re.search(rf'routeall_*(\d+)', measurement_file_current).group(1)
        # route_file = re.search(r'routeall_\d+', measurement_file_current).group(0).replace('route', '')
    else:
            # else:
            # seed = re.search(r'random_weather.*_seed_\d+_upsampled', measurement_file_current).group(0)
        route_file = re.search(r'route\d+', measurement_file_current).group(0).replace('route', '')
    train_val = re.search(r'training|validation', measurement_file_current).group(0)

    if route_folder == 'OpensDoor' or route_folder == 'pedestrians' or route_folder == 'custom_parkinglane':
        routefile_path = f'/home/katrinrenz/coding/wayve_carla/data/{route_folder}/routes_{train_val}_{seed}/{route_file}.xml'
    else:
        routefile_path = f'/home/katrinrenz/coding/wayve_carla/data/{route_folder}/routes_{train_val}/{seed}_upsampled/{route_file}.xml'

    try:
        if 'routeall' in measurement_file_current:
            route_number = re.search(rf'routeall_{route_file}_route*(\d+)', measurement_file_current).group(1)
        else:
            route_number = re.search(rf'route{route_file}_route*(\d+)', measurement_file_current).group(1)
    except:
        print(f"Error in {measurement_file_current}")

    # load route file
    if route_folder == 'custom_parkinglane':
        scenario_name = 'turn_in_parkinglane'
    elif route_folder == 'pedestrians':
        scenario_name = 'pedestrians'
    elif route_folder == 'OpensDoor':
        scenario_name = 'VehicleOpensDoor'
    else:
        tree = ET.parse(routefile_path)
        # get route id=route_number
        root = tree.getroot()
        route_id = root.find(f'./route[@id="{route_number}"]')
        # get all information as dict
        route_info = {}
        locs = []
        scenarios = []
        for scenario in route_id.find('scenarios').iter('scenario'):
            p = scenario.find('trigger_point')
            loc = [float(p.attrib['x']), float(p.attrib['y']), float(p.attrib['z'])]
            loc_ego_coords = t_u.inverse_conversion_2d(np.array(loc[:2]), current_measurement['pos_global'], current_measurement['theta'])
            locs.append(loc_ego_coords)
            scenario_name = scenario.attrib['type']
            scenarios.append(scenario_name)

            route_info[scenario_name] = loc

        current_loc = current_measurement['pos_global']
        # find the closest scenario
        # distances = [np.linalg.norm(np.array(loc[:2]) - np.array(current_loc)) for loc in locs]
        behind_or_infront = [loc[0] > 0 for loc in locs]
        closest_behind = [np.linalg.norm(np.array(loc[:2])) if loc[0] < -10 else 999999 for loc in locs ]
        closest_scenario = scenarios[np.argmin(closest_behind)]
        scenario_name = closest_scenario
    
    return scenario_name