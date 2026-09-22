import argparse
import json
import os
import pathlib
import random
import re
import shutil
import tqdm

import carla
import lxml.etree as ET
import matplotlib.pyplot as plt
import numpy as np
from xml.dom import minidom

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
from srunner.tools.scenario_helper import get_same_dir_lanes

"""
This script is used to split the long routes provided by CARLA Leaderboard 2.0 into smaller ones,
which only contain a predefined number of scenarios per route.
Adapted from https://github.com/autonomousvision/carla_garage
changes: how we generate the weather parameters, added functionality of having multiple routes in one file
"""
argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', default=1, type=int, help='Seed for random number generator.')
argparser.add_argument('--path-in', default='leaderboard/data/parking_lane/Town12_short.xml', type=str, help='Input path of the route that should be split.')
argparser.add_argument('--save-path', default='data/simlingo/training_parking_lane', type=str, help='Output directory of the files.')
argparser.add_argument('--max-scenarios', default=1, type=int, help='Maximum number of scenarios per xml file.')
argparser.add_argument('--routes-per-file', default=1, type=int, help='Maximum number of routes per xml file.')
argparser.add_argument('--only-waypoints', default=False, action='store_true', help='Controls if scenarios are included or only waypoints.')
argparser.add_argument('--port', default=6222, type=int, help='Port of the CARLA server.')
argparser.add_argument('--easy-weather', default=False, action='store_true', help='Use easy weather.')
argparser.add_argument('--exclude-scenarios', default=[], type=str, nargs='+', help='Specify, which scenarios should be excluded.')
# argparser.add_argument('--exclude-scenarios', default=['Accident', 'AccidentTwoWays', 'ConstructionObstacle', 'ConstructionObstacleTwoWays', 'HazardAtSideLane', 'HazardAtSideLaneTwoWays', 'InvadingTurn', 'ParkedObstacle', 'ParkedObstacleTwoWays', 'VehicleOpensDoorTwoWays', 'YieldToEmergencyVehicle'], type=str, nargs='+', help='Specify, which scenarios should be excluded.')
args = argparser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
random_weather = True
easy_weather = args.easy_weather

print('Seed:', args.seed)
print(f"easy_weather: {easy_weather}")

if easy_weather:
    easy_weather_str = 'easy_weather_'
else:
    easy_weather_str = ''
if random_weather:
    weather_str = f'random_weather_{easy_weather_str}seed_{args.seed}'
else:
    weather_str = 'org_weather'

train_val_test = args.path_in.split('/')[-1].split('.')[0]

path_xml = f"{args.save_path}/{train_val_test}/{weather_str}"
os.path.join(args.save_path, train_val_test)
possible_scenario_types = []
# 'Accident', 'AccidentTwoWays', 'BlockedIntersection', 'ConstructionObstacle', 'ConstructionObstacleTwoWays', 'ControlLoss',
# 'CrossingBicycleFlow', 'DynamicObjectCrossing', 'EnterActorFlow', 'EnterActorFlowV2', 'HardBreakRoute', 'HazardAtSideLane', 'HazardAtSideLaneTwoWays', 
# 'HighwayCutIn', 'HighwayExit', 'InterurbanActorFlow', 'InterurbanAdvancedActorFlow', 'InvadingTurn', 'MergerIntoSlowTraffic', 'MergerIntoSlowTrafficV2', 
# 'NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionRightTurn', 'OppositeVehicleRunningRedLight', 'OppositeVehicleTakingPriority', 'ParkedObstacle',
# 'ParkedObstacleTwoWays', 'ParkingCrossingPedestrian', 'ParkingCutIn', 'ParkingExit', 'PedestrianCrossing', 'PriorityAtJunction', 'SignalizedJunctionLeftTurn', 
# 'SignalizedJunctionRightTurn', 'StaticCutIn', 'VehicleOpensDoorTwoWays', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian', 'YieldToEmergencyVehicle']

pathlib.Path(path_xml).mkdir(parents=True, exist_ok=True)

#################################################################################################################################################
# Client and general setup
#################################################################################################################################################

client=carla.Client('localhost', args.port)
client.set_timeout(240)

tree = ET.parse(args.path_in)
town = list(tree.iter("route"))[0].attrib['town']

print('Loading town:', town)

world = client.get_world()
print("World loaded.")
if town not in world.get_map().name:
    world = client.load_world(town)

print("Getting map.")
carla_map = world.get_map()
print("Map loaded.")
grp_1 = GlobalRoutePlanner(carla_map, 1.0)

#################################################################################################################################################
# Class that saves all information about a route 
# Scenarios and waypoints are sorted in trace, trace_type, trace_elem 
#################################################################################################################################################

class Route():
    weather_params = ["route_percentage", "cloudiness", "precipitation", "precipitation_deposits", "wetness", "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle", "fog_density"]
    if random_weather:
        # random weather values
        # route_percentage 0,100, cloudiness 0-100, 
        # precipitation 0-100, precipitation_deposits 0-100, 
        # wetness 0-100, wind_intensity 0-100, sun_azimuth_angle 0-360,
        # sun_altitude_angle -90 - 90, fog_density 0-100

        if easy_weather:
            ranges_per_param = {
                'cloudiness': [0.0, 2.0, 5.0, 10.0, 15.0, 20.0],
                'precipitation': [0.0, 2.0, 4.0, 6.0],
                'precipitation_deposits': [0.0, 4.0, 8.0],
                'wetness': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                'wind_intensity': [5.0, 10.0],
                'sun_azimuth_angle': [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0],
                'sun_altitude_angle': [10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 70.0, 80.0, 90.0],
                'fog_density': [0.0, 0.0, 1.0, 1.0, 2.0]
            }
        else:
            ranges_per_param = {
                'cloudiness': [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 40.0, 50.0, 60.0, 80.0, 100.0],
                'precipitation': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0],
                'precipitation_deposits': [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0],
                'wetness': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0],
                'wind_intensity': [5.0, 10.0, 25.0, 30.0, 50.0, 60.0, 80.0, 100.0],
                'sun_azimuth_angle': [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0],
                'sun_altitude_angle': [-90.0, -45.0, -30.0, -10, -15.0, 5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 70.0, 80.0, 90.0],
                'fog_density': [0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 40.0, 70.0, 100.0]
            }
        weather_values_begin = [0]
        weather_values_end = [100]
        for param in weather_params[1:]:
            ranges = ranges_per_param[param]
            weather_values_begin.append(random.choice(ranges))
            weather_values_end.append(random.choice(ranges))


    def __init__(self, route_tree):
        self.parse_route_tree(route_tree)
        self.generate_trace_from_route()
        self.sort_scenarios_in()
        
        self.is_junction = np.array([carla_map.get_waypoint(carla.Location(x=x[0], y=x[1], z=x[2])).is_junction for x in self.trace])
        
    def parse_route_tree(self, route_tree):
        route_town = route_tree.attrib['town']

        weathers = []
        for weather in route_tree.find('weathers').iter('weather'):
            weather_tmp = []
            for param in self.weather_params:
                weather_tmp.append(float(weather.attrib[param]))
            weathers.append(weather_tmp)

        if random_weather:
            weathers = [self.weather_values_begin, self.weather_values_begin]
        else:
            self.weather_values_begin = weathers[0]
            self.weather_values_end = weathers[1]

        waypoints = []
        
        for waypoint in route_tree.find('waypoints').iter('position'):
            loc = [waypoint.attrib['x'], waypoint.attrib['y'], waypoint.attrib['z']]
            waypoints.append(loc)
            
        waypoints = np.array(waypoints).astype('float')

        scenarios = []
        scenario_trigger_points = []
        for scenario in route_tree.find('scenarios').iter('scenario'):
            p = scenario.find('trigger_point')
            loc = [p.attrib['x'], p.attrib['y'], p.attrib['z']]
            
            scenario_trigger_points.append(loc)
            scenarios.append(scenario)   
            
        scenario_trigger_points = np.array(scenario_trigger_points).astype('float')
        
        self.route_town = route_town
        self.waypoints = waypoints
        self.scenarios = scenarios
        self.scenario_trigger_points = scenario_trigger_points
        
    def generate_trace_from_route(self):
        trace = []
        trace_type = []
        trace_elem = []
        trace_cmds = []
        
        for i in range(len(self.waypoints)-1):
            p = self.waypoints[i]
            p_next = self.waypoints[i+1]
            
            waypoint = carla.Location(x=p[0], y=p[1], z=p[2])
            waypoint_next = carla.Location(x=p_next[0], y=p_next[1], z=p_next[2])
            interpolated_trace = grp_1.trace_route(waypoint, waypoint_next)

            trace_cmd = [x[1] for x in interpolated_trace]
            interpolated_trace = [x[0].transform.location for x in interpolated_trace]
            
            interpolated_trace = [[x.x, x.y, x.z] for x in interpolated_trace]
            
            trace += [p] + interpolated_trace
            trace_type += ['waypoint'] + ['trace'] * len(interpolated_trace)
            trace_elem += [None] + [None] * len(interpolated_trace)
            trace_cmds += [RoadOption.LANEFOLLOW] + trace_cmd
            
        trace.append(self.waypoints[-1])
        trace_type.append('waypoint')
        trace_elem.append(None)
        trace_cmds.append(RoadOption.LANEFOLLOW)
        
        self.trace = np.array(trace)
        self.trace_type = np.array(trace_type)
        self.trace_elem = trace_elem
        self.trace_cmds = trace_cmds
        
    def sort_scenarios_in(self):
        for scenario, scenario_location in zip(self.scenarios, self.scenario_trigger_points):
            diff = np.linalg.norm(self.trace - scenario_location[None], axis=1)
            diff[self.trace_type == 'waypoint'] = 1e9
            min_idx = np.argmin(diff)

            self.trace_cmds = np.concatenate([self.trace_cmds[:min_idx], [RoadOption.LANEFOLLOW], self.trace_cmds[min_idx:]])
            self.trace = np.concatenate([self.trace[:min_idx], [scenario_location], self.trace[min_idx:]])
            self.trace_type = np.concatenate([self.trace_type[:min_idx], ['scenario'], self.trace_type[min_idx:]])
            self.trace_elem = self.trace_elem[:min_idx] + [scenario] + self.trace_elem[min_idx:]

#################################################################################################################################################
# Read in the routes and save them in objects of Route 
#################################################################################################################################################

tree = ET.parse(args.path_in)

print("Parsing routes.")
l_routes = []
for i, route_tree in tqdm.tqdm(enumerate(tree.iter("route"))):
    l_routes.append(Route(route_tree))

print("Routes parsed.")
#################################################################################################################################################
# plot a route 
#################################################################################################################################################

# route = l_routes[0]
# trace = route.trace[route.trace_type == 'trace']
# is_junction = route.is_junction[route.trace_type == 'trace']
# scenario = route.trace[route.trace_type == 'scenario']
# waypoint = route.trace[route.trace_type == 'waypoint']

# waypoint_list = carla_map.generate_waypoints(5.0)
# waypoint_list = [x.transform.location for x in waypoint_list]
# waypoint_list = [[x.x, x.y, x.z] for x in waypoint_list]
# waypoint_list = np.array(waypoint_list)

# plt.figure(figsize=(8, 8))
# plt.scatter(waypoint_list[:, 0], waypoint_list[:, 1], c='lightgray', s=1)
# plt.scatter(trace[:, 0], trace[:, 1], c=np.where(is_junction, 'red', 'black'), s=1)
# plt.scatter(scenario[:, 0], scenario[:, 1], c='orange', s=20)
# plt.scatter(waypoint[:, 0], waypoint[:, 1], c='green', s=40, marker='x')
# plt.xlim(trace[:,0].min()-100, trace[:,0].max()+100)
# plt.ylim(trace[:,1].min()-100, trace[:,1].max()+100)
# plt.show()
# plt.clf()

#################################################################################################################################################
# Individual distances before and last scenario trigger location
#################################################################################################################################################

def get_previous_distance(scenario_type):
    if scenario_type == 'HardBreakRoute' or scenario_type == 'HighwayExit':
        return 50
    else:
        return 20

#################################################################################################################################################
# calculate waypoints x meters before first scenario trigger location (that vehicle can speed up)
#################################################################################################################################################

def get_previous_waypoints(route_idx, first_scenario_idx):
    route = l_routes[route_idx]
    scenario_type = route.trace_elem[first_scenario_idx].get('type')
    min_distance = get_previous_distance(scenario_type)
    
    trace = route.trace[:first_scenario_idx+1, :2]
    diff = np.linalg.norm(np.diff(trace, axis=0), axis=1)
    distances = np.cumsum(diff[::-1])[::-1]
    
    l_indices = []
    junction_cooldown = 0 # keeps the starting location 20m away from a junction
    for i in range(first_scenario_idx-1, -1, -1):
        if route.trace_type[i] == 'waypoint':
            l_indices.append(i)
           
        if distances[i] > min_distance:
            location = carla.Location(x=route.trace[i][0], y=route.trace[i][1], z=route.trace[i][2])
            if route.is_junction[i]:
                junction_cooldown = 20
            else:
                if junction_cooldown == 0:
                    l_indices.append(i)
                    break
                else:
                    junction_cooldown -= 1
                
    return l_indices[::-1]


# try to set the distance 10m more than the scenario lasts
distance_after = {
    'Accident': 86,                      
    'AccidentTwoWays': 86,                
    'BlockedIntersection': 20,            
    'ConstructionObstacle': 70,          
    'ConstructionObstacleTwoWays': 70,   
    'ControlLoss': 130,                   
    'CrossingBicycleFlow': 25,           
    'DynamicObjectCrossing': 70,          
    'EnterActorFlow': 110,                
    'EnterActorFlowV2': 110,              
    'HardBreakRoute': 65,                 
    'HazardAtSideLane': 150,              
    'HazardAtSideLaneTwoWays': 150,       
    'HighwayCutIn': 200,                  
    'HighwayExit': 30,                   
    'InterurbanActorFlow': 30,           
    'InterurbanAdvancedActorFlow': 50,   
    'InvadingTurn': 50,                  
    'MergerIntoSlowTraffic': 250,         
    'MergerIntoSlowTrafficV2': 250,       
    'NonSignalizedJunctionLeftTurn': 30,  
    'NonSignalizedJunctionRightTurn': 30, 
    'OppositeVehicleRunningRedLight': 30,
    'OppositeVehicleTakingPriority': 30,  
    'ParkedObstacle': 70,                 
    'ParkedObstacleTwoWays': 70,          
    'ParkingCrossingPedestrian': 60,      
    'ParkingCutIn': 85,                   
    'ParkingExit': 50,                    
    'PedestrianCrossing': 30,             
    'PriorityAtJunction': 30,             
    'SignalizedJunctionLeftTurn': 30,     
    'SignalizedJunctionRightTurn': 30,    
    'StaticCutIn': 80,                   
    'VehicleOpensDoorTwoWays': 40,        
    'VehicleTurningRoute': 70,            
    'VehicleTurningRoutePedestrian': 70,   
    'YieldToEmergencyVehicle': 260        
}


#################################################################################################################################################
# calcualtes waypoints x meters after the last scenario trigger location (to make sure the scenario ended properly)
#################################################################################################################################################

def get_value_parameter(scenario_elem, name, default):
    if scenario_elem.find(name) is None:
        return default
    else:
        return int(scenario_elem.find(name).get('value'))

def get_distance_to_end_of_next_junction(route, scenario_idx):
    is_junction = route.is_junction[scenario_idx:]
    first_idx_of_next_junction = np.argmax(is_junction)
    
    end_idx_of_next_junction = first_idx_of_next_junction
    while scenario_idx+end_idx_of_next_junction < len(is_junction) and is_junction[end_idx_of_next_junction]:
        end_idx_of_next_junction += 1
        
    differences = np.diff(route.trace[scenario_idx:scenario_idx+end_idx_of_next_junction, :2], axis=0)
    distance = np.linalg.norm(differences, axis=1).sum() + 25 # plus 20 to make sure we are outside the junction 
    
    return distance

def get_distance_to_end_of_next_junction_with_traffic_lights(route, scenario_idx):
    is_junction = route.is_junction[scenario_idx:]
    first_idx_of_next_junction = np.argmax(is_junction)
    
    end_idx_of_next_junction = first_idx_of_next_junction
    while scenario_idx + end_idx_of_next_junction < len(is_junction):
        if is_junction[end_idx_of_next_junction]:
            loc = route.trace[scenario_idx + end_idx_of_next_junction]
            wp = carla_map.get_waypoint(carla.Location(x=loc[0], y=loc[1], z=loc[2]))
            if wp.is_junction and len(world.get_traffic_lights_in_junction(wp.junction_id)) > 0:
                break

        end_idx_of_next_junction += 1
        
    differences = np.diff(route.trace[scenario_idx:scenario_idx+end_idx_of_next_junction, :2], axis=0)
    distance = np.linalg.norm(differences, axis=1).sum() + 25 # plus 25 to make sure we are outside the junction 
    
    return distance

def get_distance_to_lane_change(route, scenario_idx):
    idx = scenario_idx
    
    while True:
        if idx >= len(route.trace)-1 or \
                route.trace_cmds[idx] == RoadOption.CHANGELANELEFT or \
                route.trace_cmds[idx] == RoadOption.CHANGELANERIGHT:
            break
        idx += 1
    
    differences = np.diff(route.trace[scenario_idx:idx, :2], axis=0)
    distance = np.linalg.norm(differences, axis=1).sum()
    
    return distance
    
def get_distance_between_overlapping_routes(route, scenario_idx, max_distance):
    scenario_elem = route.trace_elem[scenario_idx]
    start_actor_flow, end_actor_flow = scenario_elem.find('start_actor_flow'), scenario_elem.find('end_actor_flow')
        
    source_wp = carla_map.get_waypoint(carla.Location(x=float(start_actor_flow.attrib['x']), y=float(start_actor_flow.attrib['y']), z=float(start_actor_flow.attrib['z'])))
    sink_wp = carla_map.get_waypoint(carla.Location(x=float(end_actor_flow.attrib['x']), y=float(end_actor_flow.attrib['y']), z=float(end_actor_flow.attrib['z'])))

    source_wps = get_same_dir_lanes(source_wp)
    sink_wps = get_same_dir_lanes(sink_wp)
    
    interpolated_traces = [grp_1.trace_route(x.transform.location, y.transform.location) for (x, y) in zip(source_wps, sink_wps)]
    interpolated_traces = np.vstack(interpolated_traces)
    interpolated_traces = [x[0].transform.location for x in interpolated_traces]
    interpolated_traces = np.array([[x.x, x.y, x.z] for x in interpolated_traces])
    
    idx = scenario_idx
    distances = np.array([100])
    min_distance = 1e9
    while True:
        current_distance = distances.min()
        min_distance = min(min_distance, current_distance)
        if idx >= len(route.trace) or current_distance<max_distance or min_distance+20 < current_distance:
            break
        distances = np.linalg.norm(route.trace[idx][None, :] - interpolated_traces, axis=1)
        idx += 1
        
    differences = np.diff(route.trace[scenario_idx:idx, :2], axis=0)
    distance = np.linalg.norm(differences, axis=1).sum()
    
    return distance
    
def get_distance_between_overlapping_routes_v2(route, scenario_idx, max_distance):
    # also uses connected lanes, not only the lane from source to sink point
    scenario_elem = route.trace_elem[scenario_idx]
    start_actor_flow, end_actor_flow = scenario_elem.find('start_actor_flow'), scenario_elem.find('end_actor_flow')
        
    source_loc = carla.Location(x=float(start_actor_flow.attrib['x']), y=float(start_actor_flow.attrib['y']), z=float(start_actor_flow.attrib['z']))
    sink_loc = carla.Location(x=float(end_actor_flow.attrib['x']), y=float(end_actor_flow.attrib['y']), z=float(end_actor_flow.attrib['z']))
       
    interpolated_trace = grp_1.trace_route(source_loc, sink_loc)
    interpolated_trace = [x[0].transform.location for x in interpolated_trace]
    interpolated_trace = np.array([[x.x, x.y, x.z] for x in interpolated_trace])
    
    idx = scenario_idx
    distances = np.array([100])
    min_distance = 1e9
    while True:
        current_distance = distances.min()
        min_distance = min(min_distance, current_distance)
        if idx >= len(route.trace) or current_distance<max_distance or min_distance+20 < current_distance:
            break
        distances = np.linalg.norm(route.trace[idx][None, :] - interpolated_trace, axis=1)
        idx += 1
        
    differences = np.diff(route.trace[scenario_idx:idx, :2], axis=0)
    distance = np.linalg.norm(differences, axis=1).sum()
    
    return distance
    
def get_distance_till_right_lane_change_is_possible(route, scenario_idx):
    idx = scenario_idx
        
    while True:
        wp = carla_map.get_waypoint(carla.Location(x=route.trace[idx, 0], y=route.trace[idx, 1], z=route.trace[idx, 2]))
        if idx >= len(route.trace) or wp.lane_change == carla.LaneChange.Right or wp.lane_change == carla.LaneChange.Both:
            break
        idx += 1

    differences = np.diff(route.trace[scenario_idx:idx, :2], axis=0)
    distance = np.linalg.norm(differences, axis=1).sum()

    return distance
    
def get_succeeding_distance(route, scenario_idx):
    scenario_elem = route.trace_elem[scenario_idx]
    scenario_type = scenario_elem.get('type')
    end_distance = distance_after[scenario_type]
    additional_distance = 0

    if scenario_type == 'Accident' or scenario_type == 'AccidentTwoWays':
        additional_distance = get_value_parameter(scenario_elem, 'distance', 120)
    elif scenario_type == 'ParkedObstacle' or scenario_type == 'ParkedObstacleTwoWays':
        additional_distance = get_value_parameter(scenario_elem, 'distance', 120)
    elif scenario_type == 'HazardAtSideLane' or scenario_type == 'HazardAtSideLaneTwoWays':
        additional_distance = get_value_parameter(scenario_elem, 'distance', 100)
        additional_distance += get_value_parameter(scenario_elem, 'bicycle_drive_distance', 50)
    elif scenario_type == 'ConstructionObstacle' or scenario_type == 'ConstructionObstacleTwoWays':
        additional_distance = get_value_parameter(scenario_elem, 'distance', 100)
    elif scenario_type == 'InvadingTurn':
        additional_distance = get_value_parameter(scenario_elem, 'distance', 100)
    elif scenario_type == 'CrossingBicycleFlow':
        additional_distance = get_distance_to_end_of_next_junction(route, scenario_idx)
    elif scenario_type == 'HighwayExit':
        additional_distance = get_distance_to_end_of_next_junction(route, scenario_idx)
    elif scenario_type == 'NonSignalizedJunctionLeftTurn':
        additional_distance = get_distance_to_end_of_next_junction(route, scenario_idx)
    elif scenario_type == 'NonSignalizedJunctionRightTurn':
        additional_distance = get_distance_to_end_of_next_junction(route, scenario_idx)
    elif scenario_type == 'OppositeVehicleRunningRedLight':
        additional_distance = get_distance_to_end_of_next_junction_with_traffic_lights(route, scenario_idx)
    elif scenario_type == 'OppositeVehicleTakingPriority':
        additional_distance = get_distance_to_end_of_next_junction(route, scenario_idx)
    elif scenario_type == 'ParkingCrossingPedestrian':
        additional_distance = get_value_parameter(scenario_elem, 'distance', 12)
    elif scenario_type == 'PedestrianCrossing':
        additional_distance = get_distance_to_end_of_next_junction(route, scenario_idx)
    elif scenario_type == 'PriorityAtJunction':
        additional_distance = get_distance_to_end_of_next_junction(route, scenario_idx)
    elif scenario_type == 'SignalizedJunctionLeftTurn':
        additional_distance = get_distance_to_end_of_next_junction(route, scenario_idx)
    elif scenario_type == 'SignalizedJunctionRightTurn':
        additional_distance = get_distance_to_end_of_next_junction(route, scenario_idx)
    elif scenario_type == 'StaticCutIn':
        additional_distance = get_value_parameter(scenario_elem, 'distance', 100)
    elif scenario_type == 'VehicleOpensDoorTwoWays':
        additional_distance = get_value_parameter(scenario_elem, 'distance', 50)
    elif scenario_type == 'VehicleTurningRoute':
        additional_distance = get_distance_to_end_of_next_junction(route, scenario_idx)
    elif scenario_type == 'VehicleTurningRoutePedestrian':
        additional_distance = get_distance_to_end_of_next_junction(route, scenario_idx)
    elif scenario_type == 'YieldToEmergencyVehicle':
        additional_distance = get_value_parameter(scenario_elem, 'distance', 140)
    elif scenario_type == 'InterurbanActorFlow':
        additional_distance = get_distance_to_end_of_next_junction(route, scenario_idx)
    elif scenario_type == 'InterurbanAdvancedActorFlow':
        additional_distance = get_distance_to_end_of_next_junction(route, scenario_idx)
    elif scenario_type == 'EnterActorFlow' or scenario_type == 'EnterActorFlowV2':
        additional_distance = get_distance_between_overlapping_routes(route, scenario_idx, 2) # lane width is often 3.5m and distance between following waypoints is 1m
    elif scenario_type == 'MergerIntoSlowTraffic' or scenario_type == 'MergerIntoSlowTrafficV2':
        additional_distance = get_distance_between_overlapping_routes_v2(route, scenario_idx, 1.5)  # lane width is often 3.5 and distance between following waypoints is 1m
    elif scenario_type == 'HighwayCutIn':
        additional_distance = get_distance_till_right_lane_change_is_possible(route, scenario_idx)
    elif scenario_type == 'DynamicObjectCrossing':
        additional_distance = get_value_parameter(scenario_elem, 'distance', 12)
    elif scenario_type == 'BlockedIntersection':
        additional_distance = get_distance_to_end_of_next_junction(route, scenario_idx)
                
    return additional_distance + end_distance

def get_succeeding_waypoints(route_idx, last_scenario_idx):    
    route = l_routes[route_idx]
    # scenario_type = route.trace_elem[last_scenario_idx].get('type')
    min_distance = get_succeeding_distance(route, last_scenario_idx) # distance_after[scenario_type]
    
    trace = route.trace[last_scenario_idx:, :2]
    diff = np.linalg.norm(np.diff(trace, axis=0), axis=1)
    distances = np.cumsum(diff)
    
    l_indices = []
    junction_cooldown = 0 # keeps the end location 20m away from a junction
    for i in range(last_scenario_idx+1, len(route.trace)):
        if route.trace_type[i] == 'waypoint':
            l_indices.append(i)
           
        if distances[i - last_scenario_idx - 1] > min_distance:
            location = carla.Location(x=route.trace[i][0], y=route.trace[i][1], z=route.trace[i][2])
            if carla_map.get_waypoint(location).is_junction:
                junction_cooldown = 20
            else:
                if junction_cooldown == 0:
                    l_indices.append(i)
                    break
                else:
                    junction_cooldown -= 1
                
    return l_indices

#################################################################################################################################################
# xml-file stuff to save the file and add the elements
######################################################################################:scenario_idx+5###########################################################

def save_file(n_file, data):
    b_xml = ET.tostring(data)
    # remove \n and \t
    b_xml = b_xml.replace(b'\n', b'').replace(b'\t', b'')
    # replace spaces between tags
    b_xml = re.sub(b'>\s+<', b'><', b_xml)


    xmlstr = minidom.parseString(b_xml).toprettyxml(indent="   ")       
    path_out = os.path.join(path_xml, "{}.xml".format(n_file))
    with open(path_out, "w") as f:
        f.write(xmlstr)
        
def write_begin_of_file(data, route, n_route, num_route_in_file):
    route_elem = ET.SubElement(data, 'route')
    route_elem.set('id', '{}'.format(num_route_in_file))
    route_elem.set('town', '{}'.format(route.route_town))
    
    weathers_elem = ET.SubElement(route_elem, 'weathers')
    weather_elem_begin = ET.SubElement(weathers_elem, 'weather')
    weather_elem_end = ET.SubElement(weathers_elem, 'weather')

    if random_weather:
        weather_values_begin = [0]
        weather_values_end = [100]
        for param in route.weather_params[1:]:
            ranges = route.ranges_per_param[param]
            weather_values_begin.append(random.choice(ranges))
            weather_values_end.append(random.choice(ranges))
    else:
        weather_values_begin = route.weather_values_begin
        weather_values_end = route.weather_values_end

    for param_name, value_begin, value_end in zip(route.weather_params, weather_values_begin, weather_values_end):
        weather_elem_begin.set(param_name, '{:.1f}'.format(value_begin))
        weather_elem_end.set(param_name, '{:.1f}'.format(value_end))
        
    waypoints_elem = ET.SubElement(route_elem, 'waypoints')
    scenarios_elem = ET.SubElement(route_elem, 'scenarios')
    
    return waypoints_elem, scenarios_elem

#################################################################################################################################################
# Adds the waypoints and the scenarios of a route to the file
#################################################################################################################################################

def add_waypoints_to_file(waypoints_elem, l_indices, route_idx):
    route = l_routes[route_idx]
    if l_indices is None:
        use_range = range(len(route.trace))
    else:
        use_range = l_indices
    for idx in use_range:
        loc = route.trace[idx]
        if route.trace_type[idx] == 'waypoint' or route.trace_type[idx] == 'trace':            
            pos_elem = ET.SubElement(waypoints_elem, 'position')
            pos_elem.set('x', '{:.1f}'.format(loc[0]))
            pos_elem.set('y', '{:.1f}'.format(loc[1]))
            pos_elem.set('z', '{:.1f}'.format(loc[2]))
        elif route.trace_type[idx] == 'scenario':
            pos_elem = ET.SubElement(waypoints_elem, 'position')
            pos_elem.set('x', '{:.1f}'.format(loc[0]))
            pos_elem.set('y', '{:.1f}'.format(loc[1]))
            pos_elem.set('z', '{:.1f}'.format(loc[2]))
        else:
            raise NotImplementedError()

def add_scenarios_to_file(scenarios_elem, l_indices, route_idx):
    if not args.only_waypoints:
        route = l_routes[route_idx]
        for idx in l_indices:
            if route.trace_type[idx] == 'scenario':
                elem = route.trace_elem[idx]
                scenarios_elem.append(elem)
            else:
                pass
                # raise NotImplementedError()
            
#################################################################################################################################################
# saves a route given where it's gonna get split
#################################################################################################################################################

def save_scenario(route_idx, l_scenario_indices, n_file, data, num_route_in_file):
    route = l_routes[route_idx]
    if l_scenario_indices is not None:
        first_scenario_idx = l_scenario_indices[0]
        last_scenario_idx = l_scenario_indices[-1]
        
        l_indices_before = get_previous_waypoints(route_idx, first_scenario_idx)
        l_indices_after = get_succeeding_waypoints(route_idx, last_scenario_idx)
            
        l_indices_scenarios, l_indices_waypoints = [], []
        for i in range(first_scenario_idx, last_scenario_idx+1):
            if route.trace_type[i] == 'scenario' and i in l_scenario_indices:
                l_indices_scenarios.append(i)
            elif route.trace_type[i] == 'waypoint':
                l_indices_waypoints.append(i)
        
        l_indices_waypoints = l_indices_before + l_indices_waypoints + l_indices_after
    
        waypoints_elem, scenarios_elem = write_begin_of_file(data, route, route_idx, num_route_in_file)
        add_waypoints_to_file(waypoints_elem, l_indices_waypoints, route_idx)
        add_scenarios_to_file(scenarios_elem, l_indices_scenarios, route_idx)
        return l_indices_waypoints[0], l_indices_waypoints[-1], data
    else:
        waypoints_elem, scenarios_elem = write_begin_of_file(data, route, route_idx, num_route_in_file)
        add_waypoints_to_file(waypoints_elem, None, route_idx)
        return data
    

#################################################################################################################################################
# Split the routes and save them into xml-files
#################################################################################################################################################

n_file = 0
l_scenario_indices = []
l_routes_delimiters = []
l_routes_scenarios = [[]]

routes_per_file = 0
data = ET.Element('routes')
print("Splitting routes.")
for route_idx, route in tqdm.tqdm(enumerate(l_routes)):
    for idx, trace_type in enumerate(route.trace_type):
        # if args.max_scenarios == 0:
        #     data = save_scenario(route_idx, None, n_file, data, routes_per_file)
        #     routes_per_file += 1
        #     if routes_per_file >= args.routes_per_file:
        #         save_file(n_file, data)
        #         data = ET.Element('routes')
        #         routes_per_file = 0
        #         n_file += 1
        # else:
        if args.max_scenarios != 0:
            if trace_type == 'scenario' and route.trace_elem[idx].attrib['type'] not in args.exclude_scenarios:
                l_scenario_indices.append(idx)
                l_routes_scenarios[-1].append([route_idx, idx])
                
            if len(l_scenario_indices) == args.max_scenarios: # save the scenarios
                from_idx, to_idx, data = save_scenario(route_idx, l_scenario_indices, n_file, data, routes_per_file)
                routes_per_file += 1
                if routes_per_file >= args.routes_per_file:
                    save_file(n_file, data)
                    data = ET.Element('routes')
                    routes_per_file = 0
                    n_file += 1            
                    
                l_routes_delimiters.append((route_idx, from_idx, to_idx))
                l_scenario_indices = []
                l_routes_scenarios.append([])
    if args.max_scenarios == 0:
        data = save_scenario(route_idx, None, n_file, data, routes_per_file)
        routes_per_file += 1
        if routes_per_file >= args.routes_per_file:
            save_file(n_file, data)
            data = ET.Element('routes')
            routes_per_file = 0
            n_file += 1
    else:
        if l_scenario_indices: # save the scenarios
            from_idx, to_idx, data = save_scenario(route_idx, l_scenario_indices, n_file, data, routes_per_file)
            routes_per_file += 1
            if routes_per_file >= args.routes_per_file:
                save_file(n_file, data)
                data = ET.Element('routes')
                routes_per_file = 0
                n_file += 1
            l_routes_delimiters.append((route_idx, from_idx, to_idx))
            l_scenario_indices = []
            l_routes_scenarios.append([])
    
if l_scenario_indices: # save the scenarios
    from_idx, to_idx, data = save_scenario(len(l_routes)-1, l_scenario_indices, n_file, data, routes_per_file)
    routes_per_file += 1
    if routes_per_file >= args.routes_per_file:
        save_file(n_file, data)
        data = ET.Element('routes')
        routes_per_file = 0
        n_file += 1
    l_routes_delimiters.append((len(l_routes)-1, from_idx, to_idx))
    l_scenario_indices = []

if not l_routes_scenarios[-1]:
    l_routes_scenarios = l_routes_scenarios[:-1]

#################################################################################################################################################
# Adds the route lengths into a list to create a histogram of route lengths later
#################################################################################################################################################

route_lengths = []
route_lengths_per_scenario = dict()
if args.max_scenarios == 0:
    for route_idx, route in tqdm.tqdm(enumerate(l_routes)):
        trace = route.trace[:, :2]
        diff = np.linalg.norm(np.diff(trace, axis=0), axis=1)
        total_route_length = diff.sum()
        route_lengths.append(total_route_length)

else:
    for idx, (route_idx, from_idx, to_idx) in enumerate(l_routes_delimiters):
        route = l_routes[route_idx]
        
        trace = route.trace[from_idx:to_idx+1, :2]
        diff = np.linalg.norm(np.diff(trace, axis=0), axis=1)

        total_route_length = diff.sum()
        route_lengths.append(total_route_length)

        for (route_idx, scenario_idx) in l_routes_scenarios[idx]:
            scenario_type = l_routes[route_idx].trace_elem[scenario_idx].get('type')
            if scenario_type not in route_lengths_per_scenario:
                route_lengths_per_scenario[scenario_type] = []
            
            route_lengths_per_scenario[scenario_type].append(total_route_length)

#################################################################################################################################################
# make histogram of route lengths and the average route distance per scenario type
#################################################################################################################################################

fig = plt.figure(figsize=(8, 8))
plt.suptitle('Route lengths for {} scenarios per route (number of routes: {})'.format(args.max_scenarios, len(route_lengths)))
plt.title('Median route length: {:.0f} m'.format(np.median(route_lengths)))
plt.ylabel('#routes')
plt.xlabel('Route length [m]')
plt.hist(route_lengths, bins=25)
plt.savefig(os.path.join(args.save_path, 'histogram.png'))
# plt.show()
fig.clear()

if route_lengths_per_scenario:
    for key in route_lengths_per_scenario.keys():
        route_lengths_per_scenario[key] = np.mean(route_lengths_per_scenario[key])

    with open(os.path.join(args.save_path, 'average_route_lenghts_per_scenario.json'), 'w', encoding="utf8") as fp:
        json.dump(route_lengths_per_scenario, fp)  # encode dict into JSON

#################################################################################################################################################
# Copy files from the ALL_ROUTES folder into folders of the folder of scenario x, if they contain scenario x at least ones
#################################################################################################################################################

if not args.only_waypoints:
    all_files = [os.path.join(path_xml, file) for file in os.listdir(path_xml)]

    for scenario_type in possible_scenario_types:
        pth = pathlib.Path(args.save_path) / scenario_type
        pth.mkdir(parents=True, exist_ok=True)
        
        file_n = 0
        for file in all_files:
            with open(file, 'r') as f:
                content = f.read()
                
                if "type=\"{}\">".format(scenario_type) in content:
                    file_name = file.split('/')[-1]
                    shutil.copy(file, str(pth / file_name))
                    file_n +=1 