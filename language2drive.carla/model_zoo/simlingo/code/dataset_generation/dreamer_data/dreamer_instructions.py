import random
import re

import numpy as np
import torch
import ujson

from dataset_generation.dreamer_data.dreamer_utils import *
from simlingo_training.models.adaptors.adaptors import cross_track_error

def get_info(alternative_trajectories, route_adjusted, route_original, current_measurement, walker_close, ego_info):
    """
    Processes alternative trajectories and generates driving instructions, safety assessments, and route reasoning.
    Parameters:
    alternative_trajectories (list): A list of dictionaries containing alternative trajectory data, including forecasts, route information, and dynamic agent intersections.
    route_adjusted (list): The adjusted route waypoints.
    route_original (list): The original route waypoints.
    current_measurement (dict): Current measurements of the vehicle, such as speed, speed limit, and route change status.
    walker_close (bool): A flag indicating if a pedestrian is close to the vehicle.
    ego_info (dict): Information about the ego vehicle, including lane details and traffic light state.
    Returns:
    dict: A dictionary where keys are instruction modes and values are lists of samples containing waypoints, route reasoning, instructions, safety assessments, and other relevant data.
    """

    route_adjusted = list(map(lambda x: [x[0], x[1]], route_adjusted))
    route_original = list(map(lambda x: [x[0], x[1]], route_original))

    template_file = f"data/augmented_templates/dreamer.json"
    with open(template_file, 'r') as f:
        dreamer_templates = ujson.load(f)

    all_data = []
    for forecast, info, route_tmp in zip(alternative_trajectories[0]['forecasts_ego_adjusted'], alternative_trajectories[0]['forecasts_ego_adjusted_info'], alternative_trajectories[0]['forecasts_ego_adjusted_route']):
        sample = {}
        sample['waypoints'] = forecast
        if route_tmp == 'org':
            sample['route'] = route_tmp # we use the string here to save memory as the expert route is already save in the measurement files
            route_reasoning = 'The predicted route follows the expert route.'
        else:
            route_neg = equal_spacing_route(route_tmp)
            cte = cross_track_error(torch.tensor(route_neg).unsqueeze(0), torch.tensor(route_adjusted).unsqueeze(0))
            max_cte = cte.max().item()
            argmax_cte = cte.argmax().item()
            if max_cte > 5.0:
                continue
            dist_to_max_cte = np.linalg.norm(route_neg[argmax_cte])
            route_reasoning = f'The route deviates from the expert route with at most {max_cte:.2f}m off at a distance of {dist_to_max_cte:.2f}m.'
            route_neg = list(map(lambda x: [x[0], x[1]], route_neg))
            sample['route'] = route_neg
        
        sample['rgb_path'] = alternative_trajectories[0]['path_rgb_image']
        sample['allowed'] = info['allowed']
        sample['mode'] = info['mode']
        sample['info'] = info
        # sample['intersection_with_dynamic_agent_per_timestep'] = alternative_trajectories[0]['intersection_bb_bool']

        dreamer_instruction, instructions_templates, templates_placeholders = get_instructions(sample['mode'], sample['info'], dreamer_templates, ego_info, current_measurement)
        
        dreamer_answer = 'Following the given instruction. Waypoints:'
        safe_to_execute = True
        
        if info['dynamic_crash'] or info['mode'] == 'crash':
            safe_to_execute = False
            dreamer_answer = 'Ignore instruction as it leads to a crash. Waypoints:'
        
        elif info['mode'] == 'target_speed' or info['mode'] == 'stop':
            new_speed = info['target_speed']
            
            # if new target speed is around speed limit we allow
            if walker_close and new_speed > current_measurement['speed']:
                safe_to_execute = False
                dreamer_answer = 'Ignore instruction as it might lead to a dangerous situation because of the pedestrian. Waypoints:'
            elif walker_close and new_speed < current_measurement['speed']:
                safe_to_execute = True
            elif current_measurement['speed_limit']*0.8 < new_speed < current_measurement['speed_limit']*1.0:
                safe_to_execute = True
            elif new_speed > current_measurement['speed_limit']*1.0:
                safe_to_execute = False
                dreamer_answer = 'Ignore instruction as it leads to a speed above the speed limit. Waypoints:'
            # if around IDM target speed we allow
            elif current_measurement['target_speed']*0.8 < new_speed < current_measurement['target_speed']*1.2:
                safe_to_execute = True
            # if new target speed is between current speed and speed limit we allow
            elif current_measurement['speed']*0.8 < new_speed < current_measurement['speed_limit']:
                safe_to_execute = True
            # if ego is standing still we allow
            elif current_measurement['speed'] < 0.5:
                safe_to_execute = True
            else:
                safe_to_execute = False
                dreamer_answer = 'Ignore instruction as it leads to a speed that is too low. Waypoints:'
                
        elif (info['mode'] == 'faster' or info['mode'] == 'slower'):
            if current_measurement['speed'] > current_measurement['speed_limit']*0.9 and info['mode'] == 'faster':
                safe_to_execute = False
                dreamer_answer = 'Ignore instruction as it leads to a speed above the speed limit. Waypoints:'
            elif walker_close and info['mode'] == 'faster':
                safe_to_execute = False
                dreamer_answer = 'Ignore instruction as it might lead to a dangerous situation because of the pedestrian. Waypoints:'
            elif walker_close and info['mode'] == 'slower':
                safe_to_execute = True
            elif current_measurement['speed'] < current_measurement['target_speed']*0.8 and info['mode'] == 'slower':
                safe_to_execute = False
                dreamer_answer = 'Ignore instruction as it leads to a speed that is too low. Waypoints:'
            else:
                safe_to_execute = True
                
        if route_tmp == 'org' and safe_to_execute:
            sample['route'] = route_adjusted
                
        if (info['mode'] == 'lane_change' and not info['allowed']):
            if 'opposite' in info['lane_change_type']:
                dreamer_answer = 'Ignore lane change instruction as the target lane is a lane with oncoming traffic. Waypoints:'
            elif 'sidewalk' in info['lane_change_type']:
                dreamer_answer = 'Ignore lane change instruction as the target lane is a sidewalk. Waypoints:'
            else:
                dreamer_answer = 'Ignore lane change instruction as it is not allowed. Waypoints:'
        elif info['dynamic_crash'] and info['mode'] != 'crash' and info['dynamic_crash_timesteps'][0] < 6:
            dreamer_answer = 'Ignore instruction as it leads to a crash with a dynamic agent. Waypoints:'
        elif info['mode'] == 'crash':
            dreamer_answer = 'Ignore instruction as it leads to a crash. Waypoints:'
        

        sample['route_reasoning'] = route_reasoning
        
        sample['dreamer_instruction'] = dreamer_instruction
        sample['instructions_templates'] = instructions_templates
        sample['templates_placeholders'] = templates_placeholders
        sample['dreamer_answer_safety'] = dreamer_answer
        sample['safe_to_execute'] = safe_to_execute
        
        all_data.append(sample)
    
    # add route negatives
    # 1. if we have a changed route we use the original one as negative
    if current_measurement['changed_route'] and current_measurement['route'] != current_measurement['route_original']:
        if current_measurement['route_original'][0][1] < 0.5: # only if we are on original route
            cte = cross_track_error(torch.tensor(route_original).unsqueeze(0), torch.tensor(route_adjusted).unsqueeze(0))
            max_cte = cte.max().item()
            argmax_cte = cte.argmax().item()
            dist_to_max_cte = np.linalg.norm(route_original[argmax_cte])
            dreamer_instruction = 'Continue driving on your current lane.'
            dreamer_answer = 'Ignore instruction as it leads to a crash with the obstacle on the lane. Waypoints:'
            
            all_data.append(
                {
                'rgb_path': alternative_trajectories[0]['path_rgb_image'],
                'allowed': False,
                'mode': 'route',
                'info': {'allowed': False, 'mode': 'route'},
                'waypoints': 'org',
                'route_reasoning': f'The route does not go around the obstruction and is at most {max_cte:.2f}m off the ground truth route at a distance of {dist_to_max_cte:.2f}m.',
                'dreamer_instruction': dreamer_instruction,
                'instructions_templates': [dreamer_instruction],
                'templates_placeholders': [{}],
                'dreamer_answer_safety': dreamer_answer,
                'safe_to_execute': False,
                'route': route_original, # this is the original route and not the adjusted one
                }
            )

    # create dict of lists with mode as key
    dreamer_dict = {}
    for option in all_data:
        if option['mode'] in dreamer_dict:
            dreamer_dict[option['mode']].append(option)
        else:
            dreamer_dict[option['mode']] = [option]

    return dreamer_dict


def get_instructions(mode, info, dreamer_templates, ego_info, current_measurements):
    """
    Generate driving instructions based on the given mode, information, templates, ego vehicle information, and current measurements.
    Parameters:
    mode (str): The mode of the instruction, e.g., 'lane_change', 'faster', 'slower', 'stop', 'target_speed', 'crash'.
    info (dict): Information related to the current driving scenario, including lane change direction, type, and other relevant details.
    dreamer_templates (dict): A dictionary containing various templates for generating instructions.
    ego_info (dict): Information about the ego vehicle, including the number of lanes in the same and opposite directions, and the current lane number.
    current_measurements (dict): Current measurements of the vehicle, such as speed, speed limit, and other relevant metrics.
    Returns:
    tuple: A tuple containing three elements:
        - instructions (list): A list of generated instructions.
        - instructions_templates (list): A list of templates used for generating the instructions.
        - templates_placeholders (list): A list of dictionaries containing placeholder values used in the templates.
    """
    
    string_mode = ''
    instructions = []
    instructions_templates = []
    templates_placeholders = []
    
    if 'lane_change' in mode:
        string_mode = 'Change'
        
        num_lanes_same_direction = ego_info['num_lanes_same_direction']
        num_lanes_opposite_direction = ego_info['num_lanes_opposite_direction']
        num_lanes_total = num_lanes_same_direction + num_lanes_opposite_direction
        ego_lane_number = ego_info['ego_lane_number']
        ego_lane_number_absolut_left_1 = ego_lane_number + num_lanes_opposite_direction + 1
        ego_lane_number_absolut_right_1 = num_lanes_same_direction - ego_lane_number
        
        lane_change = int(re.search(r'\d+', info['lane_change_direction'])[0])
        if 'left' in info['lane_change_direction']:
            goal_lane_rel_to_ego = -lane_change
            change_side = 'left'
        elif 'right' in info['lane_change_direction']:
            goal_lane_rel_to_ego = lane_change
            change_side = 'right'
        
        goal_lane_absolut_left_1 = ego_lane_number_absolut_left_1 + goal_lane_rel_to_ego
        goal_lane_absolut_right_1 = ego_lane_number_absolut_right_1 + (goal_lane_rel_to_ego * -1)
        if goal_lane_absolut_left_1 > num_lanes_opposite_direction:
            # goal lane is on the same side
            goal_lane_absolut_on_lane_type_left_1 = goal_lane_absolut_left_1 - num_lanes_opposite_direction
            goal_lane_absolut_on_lane_type_right_1 = goal_lane_absolut_right_1
            num_lanes_lane_type = num_lanes_same_direction
        else:
            goal_lane_absolut_on_lane_type_left_1 = goal_lane_absolut_left_1
            goal_lane_absolut_on_lane_type_right_1 = goal_lane_absolut_right_1 - num_lanes_same_direction
            num_lanes_lane_type = num_lanes_opposite_direction
        
        if 'opposite' in info['lane_change_type']:
            lane_type = 'opposite direction'
        elif 'driving' in info['lane_change_type']:
            lane_type = 'same direction'
        elif 'parking' in info['lane_change_type']:
            lane_type = 'parking'
        elif 'sidewalk' in info['lane_change_type']:
            lane_type = 'sidewalk'
        else:
            lane_type = 'unknown'
        
        # regex to get the number of the lane change
        # num to string
        lane_lanes = 'lanes'
        if lane_change == 1:
            lane_change = 'one'
            lane_lanes = 'lane'
        elif lane_change == 2:
            lane_change = 'two'
        elif lane_change == 3:
            lane_change = 'three'
        elif lane_change == 4:
            lane_change = 'four'
        else:
            lane_change = 'unknown'
            
        # 1st Option: relative lane number e.g. 2 lanes to the left
        # 2nd Option: absolute lane number on lanes of same/direction or opposite direction or total e.g. 3rd lane from the left on the lanes going in the same direction
        # 3rd Option: lane type e.g. parking lane, sidewalk
        
        # 1st Option: relative lane number
        answers = dreamer_templates['lanechange_rel']
        template = dreamer_templates['lanechange_rel'][0]
        placeholder_values = {
            '<LANE_CHANGE_SIDE>': change_side,
            '<LANE_NUMBERS_REL>': lane_change,
            '<LANE_OR_LANES>': lane_lanes
        }
        answer_1 = random.choice(answers).replace('<LANE_CHANGE_SIDE>', change_side).replace('<LANE_NUMBERS_REL>', lane_change).replace('<LANE_OR_LANES>', lane_lanes)
        instructions.append(answer_1)
        instructions_templates.append(template)
        templates_placeholders.append(placeholder_values)
        
        # 2nd Option: absolute lane number
        if goal_lane_absolut_on_lane_type_left_1 == 1 and random.random() < 0.3:
            lane_desc = 'leftmost lane'
        elif goal_lane_absolut_on_lane_type_left_1 == num_lanes_lane_type and random.random() < 0.3:
            lane_desc = 'rightmost lane'
            assert goal_lane_absolut_on_lane_type_right_1 == 1
        else:
            if random.random() < 0.5:
                if random.random() < 0.5:
                    goal_lane_tmp = goal_lane_absolut_on_lane_type_left_1
                else:
                    goal_lane_tmp = goal_lane_absolut_left_1
                dir_tmp = 'left'
            else:
                if random.random() < 0.5:
                    goal_lane_tmp = goal_lane_absolut_on_lane_type_right_1
                else:
                    goal_lane_tmp = goal_lane_absolut_right_1
                dir_tmp = 'right'
            if goal_lane_tmp == 1:
                lane_desc = f'1st lane from the {dir_tmp}'
            elif goal_lane_tmp == 2:
                lane_desc = f'2nd lane from the {dir_tmp}'
            elif goal_lane_tmp == 3:
                lane_desc = f'3rd lane from the {dir_tmp}'
            else:
                lane_desc = f'{goal_lane_tmp}th lane from the {dir_tmp}'

        if random.random() < 0.5:
            answer_2 = random.choice(dreamer_templates['lanechange_abs']).replace('<LANE_NUM>', lane_desc).replace('<LANE_TYPE>', lane_type)
            template = dreamer_templates['lanechange_abs'][0]
            placeholder_values = {
                '<LANE_NUM>': lane_desc,
                '<LANE_TYPE>': lane_type
            }
        else:
            answer_2 = random.choice(dreamer_templates['lanechange_abs_all']).replace('<LANE_NUM>', lane_desc)
            template = dreamer_templates['lanechange_abs_all'][0]
            placeholder_values = {
                '<LANE_NUM>': lane_desc
            }
        instructions.append(answer_2)
        instructions_templates.append(template)
        templates_placeholders.append(placeholder_values)


        # 3rd Option: lane change transition
        if 'left' in info['lane_change_direction']:
            string_mode = 'left'
        elif 'right' in info['lane_change_direction']:
            string_mode = 'right'
        else:
            string_mode = 'left or right'
        
        answer_3 = None
        if 'parking' in mode and random.random() < 0.5:
            answer_3 = random.choice(dreamer_templates['parking']).replace('<SIDE>', string_mode)
            template = dreamer_templates['parking'][0]
            placeholder_values = {
                '<SIDE>': string_mode
            }
        elif 'sidewalk' in mode and random.random() < 0.5:
            answer_3 = random.choice(dreamer_templates['sidewalk']).replace('<SIDE>', string_mode)
            template = dreamer_templates['sidewalk'][0]
            placeholder_values = {
                '<SIDE>': string_mode
            }
        
        if answer_3 is not None:
            instructions.append(answer_3)
            instructions_templates.append(template)
            templates_placeholders.append(placeholder_values)
        
        start, transition, amount = info['lane_change_in_transition_amount_meters']
        answer_4 = random.choice(dreamer_templates['lane_change_transition']).replace('<SIDE>', string_mode).replace('<LANE_NUMS>', lane_change).replace('<LANE_OR_LANES>', lane_lanes)
        answer_4 = answer_4.replace('<START>', str(start)).replace('<TRANSITION>', str(transition))
        template = dreamer_templates['lane_change_transition'][0]
        placeholder_values = {
            '<SIDE>': string_mode,
            '<LANE_NUMS>': lane_change,
            '<LANE_OR_LANES>': lane_lanes,
            '<START>': str(start),
            '<TRANSITION>': str(transition),
        }
        instructions.append(answer_4)
        instructions_templates.append(template)
        templates_placeholders.append(placeholder_values)
            
            
    elif 'faster' in mode:
        if info['dynamic_crash']:
            if random.random() < 0.5:
                instruction = random.choice(dreamer_templates['faster_crash'])
                template = dreamer_templates['faster_crash'][0]
            else:
                instruction = random.choice(dreamer_templates['faster'])
                template = dreamer_templates['faster'][0]
        elif current_measurements['speed_reduced_by_obj_type'] is not None and ('light' in current_measurements['speed_reduced_by_obj_type'] and current_measurements['speed'] < 4 and current_measurements['speed_reduced_by_obj_distance'] < 10) or ego_info['traffic_light_state'] == 'red' and ego_info['distance_to_junction'] < 8:
            if random.random() < 0.5:
                instruction = random.choice(dreamer_templates['redlight'])
                template = dreamer_templates['redlight'][0]
            else:
                instruction = random.choice(dreamer_templates['faster'])
                template = dreamer_templates['faster'][0]
        else:
            instruction = random.choice(dreamer_templates['faster'])
            template = dreamer_templates['faster'][0]

        instructions.append(instruction)
        instructions_templates.append(template)
        templates_placeholders.append({})
    
        
    elif 'slower' in mode:
        instruction = random.choice(dreamer_templates['slower'])
        template = dreamer_templates['slower'][0]
        instructions.append(instruction)
        instructions_templates.append(template)
        templates_placeholders.append({})
    
    elif 'stop' in mode:
        instruction = random.choice(dreamer_templates['stop_now'])
        template = dreamer_templates['stop_now'][0]
        instructions.append(instruction)
        instructions_templates.append(template)
        templates_placeholders.append({})
        
    elif 'target_speed' in mode:
        instruction = random.choice(dreamer_templates['target_speed'])
        template = dreamer_templates['target_speed'][0]
        target_speed_ms = info['target_speed']
        target_speed_kmh = round(target_speed_ms * 3.6, 1)
        if random.random() < 0.5:
            instruction = instruction.replace('<TARGET_SPEED>', f'{str(target_speed_kmh)} km/h')
            placeholder_values = {
                '<TARGET_SPEED>': f'{str(target_speed_kmh)} km/h'
            }
        else:
            instruction = instruction.replace('<TARGET_SPEED>', f'{str(target_speed_ms)} m/s')
            placeholder_values = {
                '<TARGET_SPEED>': f'{str(target_speed_ms)} m/s'
            }
        instructions.append(instruction)
        instructions_templates.append(template)
        templates_placeholders.append(placeholder_values)
            
        
    elif 'crash' in mode: # and not 'vqa' in info['type']:
        object_type = info['type']
        if 'Line' in object_type or 'Stencil' in object_type:
            # drive over instead of crash into
            if 'stopline' in object_type.lower():
                object_type = 'stop line'
            elif 'Stencil_STOP' in object_type:
                object_type = 'written STOP on the street'
            instruction = random.choice(dreamer_templates['driveover']).replace('<OBJECT>', object_type)
            template = dreamer_templates['driveover'][0]
            placeholder_values = {
                '<OBJECT>': object_type
            }
        elif 'walker' in object_type:
            instruction = random.choice(dreamer_templates['walker'])
            template = dreamer_templates['walker'][0]
            placeholder_values = {}
        else:
            if random.random() < 0.15:
                object_type = 'object'
                object_position = info['crash_position']
                object_position = f"x: {object_position[0]}m, y: {object_position[1]}m"
                instruction = random.choice(dreamer_templates['crash_loc']).replace('<OBJECT>', object_type).replace('<LOC>', object_position)
                template = dreamer_templates['crash_loc'][0]
                placeholder_values = {
                    '<OBJECT>': object_type,
                    '<LOC>': object_position
                }
            else:
                object_type = object_type.replace('_vqa', '').replace('crash_', '').replace('_', ' ').replace('static.prop.', 'the ').replace('.', ' ')
                if 'constructioncone' in object_type:
                    object_type = 'construction cone'
                elif 'warningconstruction' in object_type:
                    object_type = 'construction warning sign'
                elif 'warningaccident' in object_type:
                    object_type = 'accident warning sign'
                elif 'police' in object_type:
                    object_type = 'police car'
                elif 'Sign_Yield' in object_type:
                    object_type = 'yield sign'
                elif 'haybalelb' in object_type:
                    object_type = 'hay bale'
                elif 'busstoplb' in object_type:
                    object_type = 'bus stop'
                    
                instruction = random.choice(dreamer_templates['crash']).replace('<OBJECT>', object_type)
                template = dreamer_templates['crash'][0]
                placeholder_values = {
                    '<OBJECT>': object_type
                }
        instructions.append(instruction)
        instructions_templates.append(template)
        templates_placeholders.append(placeholder_values)
        
    else:
        raise ValueError(f"Unexpected mode: {mode}") # raise an error for any unknown mode

    return instructions, instructions_templates, templates_placeholders
