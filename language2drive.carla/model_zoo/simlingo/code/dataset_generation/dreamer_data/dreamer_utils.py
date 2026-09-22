import numpy as np
from scipy.interpolate import interp1d

def equal_spacing_route(points):
    route = np.concatenate((np.zeros_like(points[:1]),  points)) # Add 0 to front
    shift = np.roll(route, 1, axis=0) # Shift by 1
    shift[0] = shift[1] # Set wraparound value to 0

    dists = np.linalg.norm(route-shift, axis=1)
    dists = np.cumsum(dists)
    dists += np.arange(0, len(dists))*1e-4 # Prevents dists not being strictly increasing

    x = np.arange(0, 20, 1)
    interp_points = np.array([np.interp(x, dists, route[:, 0]), np.interp(x, dists, route[:, 1])]).T

    return interp_points