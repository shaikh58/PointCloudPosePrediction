import numpy as np

def invert_pose(pose):
    p_inv = np.zeros((4,4))
    p_inv[:3,:3] = pose[:3,:3].T
    p_inv[:3,3] = -pose[:3,:3].T @ pose[:3,3]
    p_inv[3,:3] = 0
    p_inv[3,3] = 1
    return p_inv