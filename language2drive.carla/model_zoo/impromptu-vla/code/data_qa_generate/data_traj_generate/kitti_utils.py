"""Provides helper methods for loading and parsing KITTI data."""
"""Thanks the code:https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py"""
from collections import namedtuple  
import numpy as np
from PIL import Image  

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"


OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +         # Latitude, Longitude, Altitude
                        'roll, pitch, yaw, ' +      # Roll, Pitch, Yaw angles (in radians)
                        'vn, ve, vf, vl, vu, ' +    # North, East, Forward, Left, Up velocity (m/s)
                        'ax, ay, az, af, al, au, ' + # Triaxial acceleration (m/s²)
                        'wx, wy, wz, wf, wl, wu, ' + # Triaxial angular velocity (rad/s)
                        'pos_accuracy, vel_accuracy, ' + # Position and velocity accuracy estimates
                        'navstat, numsats, ' +      # Navigation status, Number of satellites
                        'posmode, velmode, orimode') # Position mode, Velocity mode, Orientation mode

# Contains raw data packets and corresponding transformation matrices (from IMU coordinate system to world coordinate system)
OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')

def subselect_files(files, indices):
    try:
        files = [files[i] for i in indices]
    except Exception:
        pass
    return files

def rotx(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def transform_from_rot_trans(R, t):
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def read_calib_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                key, value = line.split(':', 1)
            except ValueError:
                key, value = line.split(' ', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def pose_from_oxts_packet(packet, scale):
    er = 6378137. 
  
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))
    return R, t

def load_oxts_packets_and_poses(oxts_files):
    scale = None
    origin = None
    oxts = []

    for filename in oxts_files:
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]
                packet = OxtsPacket(*line)
                if scale is None:
                    scale = np.cos(packet.lat * np.pi / 180.)
                R, t = pose_from_oxts_packet(packet, scale)
                if origin is None:
                    origin = t
                T_w_imu = transform_from_rot_trans(R, t - origin)
                oxts.append(OxtsData(packet, T_w_imu))
    return oxts

def load_image(file, mode):
    return Image.open(file).convert(mode)

def yield_images(imfiles, mode):
    for file in imfiles:
        yield load_image(file, mode)

def load_velo_scan(file):
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))

def yield_velo_scans(velo_files):
    for file in velo_files:
        yield load_velo_scan(file)



