import numpy as np
from enum import Enum
import open3d as o3d
from typing import Callable 
import cv2
from functools import singledispatchmethod


class PointCloudProcessor:
    def __init__(self) -> None:
        pass

    def pixel2optical(self, arr_uv, arr_d, K) -> np.ndarray:
        fsu, fsv = K[0,0], K[1,1]
        c_u, c_v = K[0,2], K[1,2]
        x = arr_d*(arr_uv[:,0]-c_u)/fsu
        y = arr_d*(arr_uv[:,1]-c_v)/fsv
        return np.stack((x,y,arr_d,np.ones_like(x)))

    def optical2world(self, pose, coord_optical) -> np.ndarray:
        opt2reg = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
        R = pose[:3,:3]
        p = pose[:3,3]
        extrinsics = np.zeros_like(pose)
        extrinsics[:3,:3] = opt2reg @ R.T
        extrinsics[:3,3] = -opt2reg @ R.T @ p
        extrinsics[3,3] = 1
        coord_world = np.linalg.pinv(extrinsics) @ coord_optical
        return coord_world

    def pt_from_rgbd(self, arr_uv, arr_d, K, pose) -> np.ndarray:
        arr_coord_opt = self.pixel2optical(arr_uv,arr_d,K)
        arr_coord_world = self.optical2world(pose, arr_coord_opt)
        return arr_coord_world[:-1,...].T
    
    def normalize(self, pcd) -> np.ndarray:
        pcd = pcd - np.expand_dims(np.mean(pcd, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(pcd ** 2, axis=1)), 0)
        pcd = pcd / dist  # scale
        return pcd



class KeyPointDetectorType(Enum):
    ORB = 1
    SIFT = 2
    FPFH = 3
    RANDOM = 4


class KeyPointDetector:
    def __init__(self, pcd_num_pts) -> None:
        self.router : dict = {"ORB":self.orb, "SIFT":self.sift, "FPFH":self.fpfh, "RANDOM":self.random}
        self.pcd_num_pts = pcd_num_pts

    def process(self, keyPointAlgo : KeyPointDetectorType, data) -> Callable:
        """Takes in img or pcd depending on the keypt detection method being used; argument must match method"""
        return self.router[keyPointAlgo.name](data)
        
    def orb(self, img : np.ndarray) -> list:
        """need to project pixel frame keypoints onto point cloud (intrinsics)"""
        _kp = []
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(img,None)
        for data in kp:
            _kp.append(data.pt)
        return _kp
    
    def sift(self, img : np.ndarray) -> list:
        _kp = []
        sift = cv2.SIFT_create() 
        kp, des = sift.detectAndCompute(img,None)
        for data in kp:
            _kp.append(data.pt)
        return _kp
    
    def fpfh(self, pcd : o3d.geometry.PointCloud):
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=100))
        return pcd_fpfh
    
    @singledispatchmethod
    def random(self, arg):
        return

    @random.register(np.ndarray)
    def _(self, arg : np.ndarray) -> np.ndarray:
        print(arg.shape)
        keep_inds = np.random.choice(arg.shape[0]-1, self.pcd_num_pts)
        return arg[keep_inds]
    
    @random.register(o3d.geometry.PointCloud)
    def _(self, arg : o3d.geometry.PointCloud) -> np.ndarray:
        keep_inds = np.random.choice(len(arg.points)-1, self.pcd_num_pts)
        return np.asarray(arg.points)[keep_inds]
    

    
keypt_enum_map = {"orb":KeyPointDetectorType.ORB,
                "fpfh":KeyPointDetectorType.FPFH,
                "sift":KeyPointDetectorType.SIFT,
                "random":KeyPointDetectorType.RANDOM}