import numpy as np
from enum import Enum
import open3d as o3d
from typing import Callable 

def invert_pose(pose):
    p_inv = np.zeros((4,4))
    p_inv[:3,:3] = pose[:3,:3].T
    p_inv[:3,3] = -pose[:3,:3].T @ pose[:3,3]
    p_inv[3,:3] = 0
    p_inv[3,3] = 1
    return p_inv

class KeyPointDetectorType(Enum):
    ORB = 1
    SIFT = 2
    FPFH = 3
    RANDOM = 4

class KeyPointDetector:
    def __init__(self) -> None:
        self.router : dict = {"ORB":self.orb, "SIFT":self.sift, "FPFH":self.fpfh, "RANDOM":self.random}

    def process(self, keyPointAlgo : KeyPointDetectorType, img=None, pcd=None) -> Callable:
        """Takes in one of img or pcd depending on which detector to use. Routes to appropriate function"""
        if img:
            return self.router[keyPointAlgo.name](img)
        if pcd:
            return self.router[keyPointAlgo.name](pcd)
        
    def orb(self, img : np.ndarray):
        """need to project pixel frame keypoints onto point cloud (intrinsics)"""
        # orb = cv2.ORB_create()
        # kp, des = orb.detectAndCompute(img,None)
        # return kp
        return
    
    def sift(self, img : np.ndarray):
        # surf = cv2.SIFT_create() 
        # kp, des = surf.detectAndCompute(img,None)
        # return kp
        return
    
    def fpfh(self, pcd : o3d.geometry.PointCloud):
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=100))
        return pcd_fpfh
    
    def random(self, pcd : o3d.geometry.PointCloud):
        keep_inds = np.random.choice(len(pcd.points)-1, len(pcd.points)//100)
        return np.asarray(pcd.points)[keep_inds]
    

keypt_enum_map = {"orb":KeyPointDetectorType.ORB,
                "fpfh":KeyPointDetectorType.FPFH,
                "sift":KeyPointDetectorType.SIFT,
                "random":KeyPointDetectorType.RANDOM}