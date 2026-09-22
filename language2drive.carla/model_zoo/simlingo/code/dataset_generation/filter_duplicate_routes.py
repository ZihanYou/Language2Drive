import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import carla
import os
import shutil
import glob
import pathlib
from agents.navigation.global_route_planner import GlobalRoutePlanner
import argparse
import time
from pathlib import Path
from IPython.display import display, clear_output
import tqdm

#
# This scipt is used to filter duplicate routes from our dataset
# such that all routes have their start and end points at least min_dist meters apart
# You need to have CARLA running on the port given below.
# To start CARLA, cd $CARLA_ROOT and use ./CarlaUE4.sh --world-port=4542 -RenderOffScreen -nosound
# Use the garage environment to execute the script.
#

TOWN = 'Town13'
MAIN_DIR = 'data/longxall_val/routes_validation'   # without trailing slash
MAIN_DESTINATION = 'data/benchmarks/longxall_train_filtered'
Path(MAIN_DESTINATION).mkdir(parents=True, exist_ok=True)
min_dist = 0.2

client=carla.Client('localhost', 2500)
client.set_timeout(240)
world = client.load_world(TOWN)

carla_map = world.get_map()
grp_1 = GlobalRoutePlanner(carla_map, 1.0)
print('loaded.')

# **Classes that are handy later**


class Route():
    weather_params = ["route_percentage", "cloudiness", "precipitation", "precipitation_deposits", "wetness", "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle", "fog_density"]
    weather_values_begin = [0, 5.0, 0.0, 0.0, 0.0, 10.0, -1.0, 45.0, 2.0]
    weather_values_end = [100, 5.0, 0.0, 0.0, 0.0, 10.0, -1.0, 45.0, 2.0]
    
    def __init__(self, route_tree):
        self.parse_route_tree(route_tree)
        self.create_trace()
        self.sort_scenarios_in()
        
    def parse_route_tree(self, route_tree):
        self.route_town = route_tree.attrib['town']
        self.waypoints = []
        
        for waypoint in route_tree.find('waypoints').iter('position'):
            loc = [waypoint.attrib['x'], waypoint.attrib['y'], waypoint.attrib['z']]
            self.waypoints.append(loc)
            
        self.waypoints = np.array(self.waypoints).astype('float')

        self.scenarios = []
        self.scenario_locations = []
        for scenario in route_tree.find('scenarios').iter('scenario'):
            p = scenario.find('trigger_point')
            loc = [p.attrib['x'], p.attrib['y'], p.attrib['z']]
            
            self.scenario_locations.append(loc)
            self.scenarios.append(scenario)   
            
        self.scenario_locations = np.array(self.scenario_locations).astype('float')
        
    def create_trace(self):
        self.trace = []
        self.trace_type = []
        self.trace_elem = []
        
        for i in range(len(self.waypoints)-1):
            p = self.waypoints[i]
            p_next = self.waypoints[i+1]
            
            waypoint = carla.Location(x=p[0], y=p[1], z=p[2])
            waypoint_next = carla.Location(x=p_next[0], y=p_next[1], z=p_next[2])
            interpolated_trace = grp_1.trace_route(waypoint, waypoint_next)

            interpolated_trace = [x[0].transform.location for x in interpolated_trace]
            interpolated_trace = [[x.x, x.y, x.z] for x in interpolated_trace]
            
            self.trace += [p] + interpolated_trace
            self.trace_type += ['waypoint'] + ['trace'] * len(interpolated_trace)
            self.trace_elem += [None] + [None] * len(interpolated_trace)
            
        self.trace.append(p_next)
        self.trace_type.append('waypoint')
        self.trace_elem.append(None)
        
        self.trace = np.array(self.trace)
        self.trace_type = np.array(self.trace_type)
        
    def sort_scenarios_in(self):
        for scenario, scenario_location in zip(self.scenarios, self.scenario_locations):
            diff = self.trace - scenario_location[None]
            diff = np.linalg.norm(diff, axis=1)
            diff[self.trace_type == 'waypoint'] = 1e9
            min_idx = np.argmin(diff)

            self.trace = np.concatenate([self.trace[:min_idx], [scenario_location], self.trace[min_idx:]])
            self.trace_type = np.concatenate([self.trace_type[:min_idx], ['scenario'], self.trace_type[min_idx:]])
            self.trace_elem = self.trace_elem[:min_idx] + [scenario] + self.trace_elem[min_idx:]

# generate map backgroudn waypoints
waypoint_list = carla_map.generate_waypoints(5.0)
waypoint_list = [x.transform.location for x in waypoint_list]
waypoint_list = [[x.x, x.y, x.z] for x in waypoint_list]
waypoint_list = np.array(waypoint_list)

dirs = glob.glob(MAIN_DIR + '/*/')
for DIR in  dirs:
    if "allroutes" in DIR:
        continue
    destination = MAIN_DESTINATION + '/' + DIR.split("/")[-2]
    Path(destination).mkdir(parents=True, exist_ok=True)

    l_routes = []
    l_routes_idx = []

    all_xml_paths = glob.glob(f'{DIR}/**/*.xml', recursive=True)

    for pth in tqdm.tqdm(all_xml_paths):
        tree = ET.parse(pth)

        for i, route_tree in enumerate(tree.iter("route")):
            l_routes.append(Route(route_tree))
            l_routes_idx.append(pth.split('/')[-1].split('.')[0])

    # plot map
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(waypoint_list[:, 0], waypoint_list[:, 1], c='lightgray', s=1)

    saved_wps = []
    saved_wps_idx = []

    for i, route in tqdm.tqdm(enumerate(l_routes)):
        trace = route.trace[route.trace_type == 'trace']
        scenario = route.trace[route.trace_type == 'scenario']
        waypoint = route.trace[route.trace_type == 'waypoint']

        # check if route at similar location already exists
        # check 1. wp with 1. wp, last wp with last wp, 1. wp with last wp, last wp with 1. wp
        duplicate = False
        for saved_route in saved_wps:
            if np.linalg.norm(saved_route[0] - waypoint[0]) < min_dist and np.linalg.norm(saved_route[-1] - waypoint[-1]) < min_dist:
                duplicate = True
                break
            #elif np.linalg.norm(saved_route[-1] - waypoint[-1]) < min_dist:
            #    duplicate = True
            #    break
            # elif np.linalg.norm(saved_route[0] - waypoint[-1]) < min_dist:
            #     duplicate = True
            #     break
            # elif np.linalg.norm(saved_route[-1] - waypoint[0]) < min_dist:
            #     duplicate = True
            #     break
        if duplicate:
            continue
        saved_wps.append(waypoint)
        saved_wps_idx.append(l_routes_idx[i])
        # ax.cla()
        ax.scatter(trace[:, 0], trace[:, 1], c='black', s=1)
        ax.scatter(scenario[:, 0], scenario[:, 1], c='orange', s=20)
        ax.scatter(waypoint[:, 0], waypoint[:, 1], c='green', s=40, marker='x')
        ax.annotate(str(l_routes_idx[i]), (waypoint[0, 0]+40, waypoint[0, 1]+40))
        
        #display(fig)    
        clear_output(wait = True)
        # plt.pause(0.1)
        # plt.waitforbuttonpress()
    plt.savefig(f"{destination}/{DIR.split('/')[-2]}_{min_dist}.png")
    #plt.waitforbuttonpress()   
    #plt.show()

    # copy routes to new folder
    for routes in saved_wps_idx:
        # shutil.copy(os.path.join(DIR, routes+'.xml'), os.path.join(destination, DIR.split('/')[-1]))
        shutil.copy(os.path.join(DIR, routes+'.xml'), destination)

    print(f"Number of selected routes: {len(saved_wps_idx)}")
    print(f"Number of scenarios: {len(l_routes_idx)}")

    plt.clf()