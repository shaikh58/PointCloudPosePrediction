import numpy as np
from torch import zeros

def invert_pose(pose, is_torch=False):
    p_inv = np.zeros((4,4))
    if is_torch:
        p_inv = zeros((4,4))
    p_inv[:3,:3] = pose[:3,:3].T
    p_inv[:3,3] = -pose[:3,:3].T @ pose[:3,3]
    p_inv[3,:3] = 0
    p_inv[3,3] = 1
    return p_inv
