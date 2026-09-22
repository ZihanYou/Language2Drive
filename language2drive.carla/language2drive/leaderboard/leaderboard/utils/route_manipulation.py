#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Labs.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Module to manipulate the routes, by making then more or less dense (Up to a certain parameter).
It also contains functions to convert the CARLA world location do GPS coordinates.
"""

import math
import xml.etree.ElementTree as ET
import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption


def _location_to_gps(lat_ref, lon_ref, location):
    """
    Convert from world coordinates to GPS coordinates
    :param lat_ref: latitude reference for the current map
    :param lon_ref: longitude reference for the current map
    :param location: location to translate
    :return: dictionary with lat, lon and height
    """

    EARTH_RADIUS_EQUA = 6378137.0   # pylint: disable=invalid-name
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    mx += location.x
    my -= location.y

    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    z = location.z

    return {'lat': lat, 'lon': lon, 'z': z}


def location_route_to_gps(route, lat_ref, lon_ref):
    """
        Locate each waypoint of the route into gps, (lat long ) representations.
    :param route:
    :param lat_ref:
    :param lon_ref:
    :return:
    """
    gps_route = []

    for transform, connection in route:
        gps_point = _location_to_gps(lat_ref, lon_ref, transform.location)
        gps_route.append((gps_point, connection))

    return gps_route


def _get_latlon_ref(world):
    """
    Convert from waypoints world coordinates to CARLA GPS coordinates
    :return: tuple with lat and lon coordinates
    """
    xodr = world.get_map().to_opendrive()
    tree = ET.ElementTree(ET.fromstring(xodr))

    # default reference
    lat_ref = 42.0
    lon_ref = 2.0

    for opendrive in tree.iter("OpenDRIVE"):
        for header in opendrive.iter("header"):
            for georef in header.iter("geoReference"):
                if georef.text:
                    str_list = georef.text.split(' ')
                    for item in str_list:
                        if '+lat_0' in item:
                            lat_ref = float(item.split('=')[1])
                        if '+lon_0' in item:
                            lon_ref = float(item.split('=')[1])
    return lat_ref, lon_ref


def downsample_route(route, sample_factor):
    """
    Downsample the route by some factor.
    :param route: the trajectory , has to contain the waypoints and the road options
    :param sample_factor: Maximum distance between samples
    :return: returns the ids of the final route that can
    """

    ids_to_sample = []
    prev_option = None
    dist = 0

    for i, point in enumerate(route):
        curr_option = point[1]

        # At the beginning
        if prev_option is None:
            ids_to_sample.append(i)
            dist = 0

        # Lane changing
        elif curr_option in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            ids_to_sample.append(i)
            dist = 0

        # When entering or exitting intersections
        elif prev_option != curr_option and prev_option not in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            ids_to_sample.append(i)
            dist = 0

        # After a certain max distance
        elif dist > sample_factor:
            ids_to_sample.append(i)
            dist = 0

        # At the end
        elif i == len(route) - 1:
            ids_to_sample.append(i)
            dist = 0

        # Compute the distance traveled
        else:
            curr_location = point[0].location
            prev_location = route[i-1][0].location
            dist += curr_location.distance(prev_location)

        prev_option = curr_option

    return ids_to_sample


def interpolate_trajectory(waypoints_trajectory, hop_resolution=1.0):
    """
    Given some raw keypoints interpolate a full dense trajectory to be used by the user.
    returns the full interpolated route both in GPS coordinates and also in its original form.
    
    Args:
        - waypoints_trajectory: the current coarse trajectory
        - hop_resolution: distance between the trajectory's waypoints
    """

    grp = GlobalRoutePlanner(CarlaDataProvider.get_map(), hop_resolution)
    # Obtain route plan
    lat_ref, lon_ref = _get_latlon_ref(CarlaDataProvider.get_world())

    route = []
    gps_route = []

    for i in range(len(waypoints_trajectory) - 1):

        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        interpolated_trace = grp.trace_route(waypoint, waypoint_next)
        for wp, connection in interpolated_trace:
            route.append((wp.transform, connection))
            gps_coord = _location_to_gps(lat_ref, lon_ref, wp.transform.location)
            gps_route.append((gps_coord, connection))

    return gps_route, route


def direct_trajectory_from_oracle_path(oracle_path):
    """Compile a validated L2D sidecar polyline without a second GRP pass.

    Language2Drive's private ``oracle_path`` is already a directed, sampled
    legal witness (and may deliberately terminate in a confirmed semantic or
    parking polygon). Feeding its sparse samples through GlobalRoutePlanner a
    second time can select another arm, overshoot a parking slot, or fail on
    the topology-free ``road1`` control map. This compiler preserves every
    supplied coordinate and assigns an explicit RoadOption to every sample.

    The returned structure is evaluator/scenario data. Standardized agents are
    never given it by the leaderboard evaluator.
    """

    if not isinstance(oracle_path, list) or len(oracle_path) < 2:
        raise ValueError("private oracle_path requires at least two points")
    points = []
    explicit_yaws = []
    for index, point in enumerate(oracle_path):
        if not isinstance(point, dict):
            raise ValueError("private oracle_path point {} is not an object".format(index))
        try:
            values = (
                float(point["x"]), float(point["y"]), float(point["z"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "private oracle_path point {} is invalid".format(index)
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "private oracle_path point {} is not finite".format(index)
            )
        if points and math.hypot(
            values[0] - points[-1][0], values[1] - points[-1][1]
        ) <= 1e-6:
            raise ValueError(
                "private oracle_path has duplicate consecutive point {}".format(index)
            )
        points.append(values)
        try:
            yaw = float(point["yaw_deg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "private oracle_path point {} requires yaw_deg".format(index)
            ) from exc
        if not math.isfinite(yaw):
            raise ValueError(
                "private oracle_path point {} yaw_deg is not finite".format(index)
            )
        explicit_yaws.append(yaw)

    headings = [
        math.atan2(second[1] - first[1], second[0] - first[0])
        for first, second in zip(points, points[1:])
    ]

    def normalize_angle(value):
        return (value + math.pi) % (2.0 * math.pi) - math.pi

    options = []
    for index in range(len(points)):
        if index == 0 or index == len(points) - 1:
            options.append(RoadOption.LANEFOLLOW)
            continue
        deviation = normalize_angle(headings[index] - headings[index - 1])
        if abs(deviation) < math.radians(20.0):
            options.append(RoadOption.LANEFOLLOW)
        # CARLA's GlobalRoutePlanner convention maps a negative z cross
        # product to LEFT and positive to RIGHT.
        elif deviation < 0.0:
            options.append(RoadOption.LEFT)
        else:
            options.append(RoadOption.RIGHT)

    lat_ref, lon_ref = _get_latlon_ref(CarlaDataProvider.get_world())
    route = []
    gps_route = []
    for index, ((x_value, y_value, z_value), option) in enumerate(
        zip(points, options)
    ):
        heading_degrees = explicit_yaws[index]
        transform = carla.Transform(
            carla.Location(x=x_value, y=y_value, z=z_value),
            carla.Rotation(yaw=heading_degrees),
        )
        route.append((transform, option))
        gps_route.append(
            (_location_to_gps(lat_ref, lon_ref, transform.location), option)
        )
    return gps_route, route
