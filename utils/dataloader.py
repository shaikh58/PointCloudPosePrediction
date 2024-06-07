import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
# import open3d as o3d
import cv2
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from typing import Any, Callable, Optional, Tuple
import utils.utils
import utils.vision


class EdenDataset(Dataset):
    def __init__(self, data_dir, keypt_method:str, pcd_num_pts:int) -> None:
        
        self.data_dir = data_dir
        self.sequences = sorted(os.listdir(os.path.join(data_dir,"rgb")))[1:] # 0001, 0003, etc.
        self.dict_seq_len = {seq:len(os.listdir(os.path.join(data_dir, "poses", seq, "clear"))) 
                             for seq in self.sequences}
        self.seq_len = 500  # Eden dataset has 500 data points per sequence (sequence is "0001" etc)
        self.dset_len = 0
        for k,v in self.dict_seq_len.items():
            self.dset_len += v
        
        self.cam_width, self.cam_height = 640, 480
        self.K = np.array([[640,0,320],[0,640,240],[0,0,1]])

        self.pcd_num_pts = pcd_num_pts
        self.keypt_method = utils.vision.keypt_enum_map[keypt_method]

        self.KeyPointDetector = utils.vision.KeyPointDetector(pcd_num_pts)
        self.PointCloudProcessor = utils.vision.PointCloudProcessor()

    def get_corr_depth(self, x) -> float:
            """Gets depth value corresponding to array of pixel positions (u,v)"""
            # keypt detection returns x,y i.e. (640,480) but depth img is (480,640)
            return self.depth[x[:,1].astype(int), x[:,0].astype(int)]

    # recall: this method overloads the square bracket operator (__setitem__ is setter)
    def __getitem__(self, index):

        raw_seq, pos = divmod(index, self.seq_len)
        seq = self.sequences[raw_seq]
        rgb_dir_path = os.path.join(self.data_dir, "rgb", seq, "clear")
        seg_dir_path = os.path.join(self.data_dir, "segmentation", seq, "clear")
        depth_dir_path = os.path.join(self.data_dir, "depth", seq, "clear")
        pose_dir_path = os.path.join(self.data_dir, "poses", seq, "clear")

        rgb_file_list = sorted(os.listdir(rgb_dir_path))
        rgb_file = rgb_file_list[pos*2] # rgb dir has L/R pairs i.e. double the number of images
        
        self.rgb = cv2.imread(os.path.join(rgb_dir_path, rgb_file), cv2.IMREAD_GRAYSCALE)
        self.seg = cv2.imread(os.path.join(seg_dir_path, rgb_file[:-4] + "_Sem.png"), cv2.IMREAD_GRAYSCALE)
        self.pose = loadmat(os.path.join(pose_dir_path, rgb_file[:-6] + ".mat"))["RT_L"]
        self.pose = np.concatenate((self.pose,np.array([[0,0,0,1]])))
        self.depth = cv2.imread(os.path.join(depth_dir_path, rgb_file[:-4] + "_vis.png"), cv2.IMREAD_GRAYSCALE)

        # now get the previous set of images etc. for the point cloud registration
        # note 100 data pts per trajectory so reset poses every 100
        if pos == 0 or pos == 100:
            self.pose_prev = self.pose
            self.rgb_prev = self.rgb
            self.depth_prev = self.depth
            self.seg_prev = self.seg
        else:
            self.pose_prev = loadmat(os.path.join(pose_dir_path, rgb_file_list[pos*2 - 2][:-6] + ".mat"))["RT_L"]
            self.pose_prev = np.concatenate((self.pose_prev,np.array([[0,0,0,1]])))
            self.rgb_prev = cv2.imread(os.path.join(rgb_dir_path, rgb_file_list[pos*2 - 2]), cv2.IMREAD_GRAYSCALE)
            self.depth_prev = cv2.imread(os.path.join(depth_dir_path, rgb_file_list[pos*2 - 2][:-4] + "_vis.png"), cv2.IMREAD_GRAYSCALE)
            self.seg_prev = cv2.imread(os.path.join(seg_dir_path, rgb_file_list[pos*2 - 2][:-4] + "_Sem.png"), cv2.IMREAD_GRAYSCALE)

        # get relative pose; t_T_t+1 i.e. next to current - inside brackets is tTw @ w_T_t-1 = t_T_t-1
        rel_pose = np.dot(utils.utils.invert_pose(self.pose), self.pose_prev)
        # convert pose matrix to quaternion/position/orientation
        quat = R.from_matrix(rel_pose[:3,:3]).as_quat()
        translation =  rel_pose[:3,3]

        # detect key points from original input images
        list_kp = self.KeyPointDetector.process(keyPointAlgo=self.keypt_method, data=self.rgb)
        list_kp_prev = self.KeyPointDetector.process(keyPointAlgo=self.keypt_method, data=self.rgb_prev)

        arr_kp = np.zeros((self.pcd_num_pts,2))
        arr_kp_prev = np.zeros((self.pcd_num_pts,2))

        # create point cloud array; could have less key points than specified in config
        pcd = np.zeros((self.pcd_num_pts,3))
        pcd_prev = np.zeros((self.pcd_num_pts,3))

        # stop loop at shorter keypt list; assume successive images have similar number of keypts
        for i in range(min(len(list_kp),len(list_kp_prev),self.pcd_num_pts)):
            arr_kp[i] = list_kp[i]
            arr_kp_prev[i] = list_kp_prev[i]

        # fill empty rows with first value (arbitrary) in case there aren't enough pts from keypt detection
        arr_kp[np.all(arr_kp==0, axis=1)] = arr_kp[0]
        arr_kp_prev[np.all(arr_kp_prev==0, axis=1)] = arr_kp_prev[0]

        # get depth values corresponding to keypoint pixel indexes
        arr_kp_depth = self.get_corr_depth(arr_kp)
        arr_kp_prev_depth = self.get_corr_depth(arr_kp_prev)

        # project pixel frame u,v coordinates to world frame 3D point
        pcd = self.PointCloudProcessor.pt_from_rgbd(arr_kp, arr_kp_depth, self.K, self.pose)
        pcd_prev = self.PointCloudProcessor.pt_from_rgbd(arr_kp_prev, arr_kp_prev_depth, self.K, self.pose_prev)

        # normalize the point cloud
        pcd = self.PointCloudProcessor.normalize(pcd)
        pcd_prev = self.PointCloudProcessor.normalize(pcd_prev)

        return {"pcd1":torch.from_numpy(pcd_prev).float(),\
                "pcd2":torch.from_numpy(pcd).float(),\
                "rel_pose":torch.from_numpy(rel_pose).float(),\
                "translation":torch.from_numpy(translation).float(),\
                "quat":torch.from_numpy(quat).float(),\
                "pose":torch.from_numpy(self.pose).float()
        }
    
    def __len__(self):
        return self.dset_len
    

# original loader using open3d functions
# class EdenDataset(Dataset):
#     def __init__(self, data_dir, use_keypt_downsample : bool, keypt_method : str) -> None:
        
#         self.data_dir = data_dir
#         self.sequences = sorted(os.listdir(os.path.join(data_dir,"rgb")))[1:] # 0001, 0003, etc.
#         self.dict_seq_len = {seq:len(os.listdir(os.path.join(data_dir, "poses", seq, "clear"))) 
#                              for seq in self.sequences}
#         self.seq_len = 500  # assume Eden dataset has 500 data points per sequence
#         self.dset_len = 0
#         for k,v in self.dict_seq_len.items():
#             self.dset_len += v
        
#         self.cam_width, self.cam_height = 640, 480
#         self.cam_instrinsics = np.array([[640,0,320],[0,640,240],[0,0,1]])

#         self.use_keypt_downsample = use_keypt_downsample
#         self.keypt_method = utils.vision.keypt_enum_map[keypt_method]

# #         if self.use_keypt_downsample:
#         self.keypt_detector = utils.vision.KeyPointDetector()
    
#     def pcd_from_rgb_depth(self, rgb, depth) -> o3d.geometry.PointCloud:
#         # o3d assumes depth scale = 1000 i.e. depth values given in mm. Eden has depth in m
#         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1)
#         pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,
#         o3d.camera.PinholeCameraIntrinsic(self.cam_width, self.cam_height, self.cam_instrinsics))
#         # Flip it, otherwise the pointcloud will be upside down
#         pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#         return pcd
    
#     def process_pcd(self, pcd) -> np.ndarray:
#         pcd = pcd - np.expand_dims(np.mean(pcd, axis=0), 0)  # center
#         dist = np.max(np.sqrt(np.sum(pcd ** 2, axis=1)), 0)
#         pcd = pcd / dist  # scale
#         return pcd

#     # recall: this method overloads the square bracket operator (_setitem_ is setter)
#     def __getitem__(self, index):

#         raw_seq, pos = divmod(index, self.seq_len)
#         seq = self.sequences[raw_seq]
#         rgb_dir_path = os.path.join(self.data_dir, "rgb", seq, "clear")
#         seg_dir_path = os.path.join(self.data_dir, "segmentation", seq, "clear")
#         depth_dir_path = os.path.join(self.data_dir, "depth", seq, "clear")
#         pose_dir_path = os.path.join(self.data_dir, "poses", seq, "clear")

#         rgb_file_list = sorted(os.listdir(rgb_dir_path))
#         rgb_file = rgb_file_list[pos*2] # rgb dir has L/R pairs i.e. double the number of images
        
#         self.rgb = o3d.io.read_image(os.path.join(rgb_dir_path, rgb_file))
#         self.seg = o3d.io.read_image(os.path.join(seg_dir_path, rgb_file[:-4] + "_Sem.png"))
#         self.pose = loadmat(os.path.join(pose_dir_path, rgb_file[:-6] + ".mat"))["RT_L"]
#         self.pose = np.concatenate((self.pose,np.array([[0,0,0,1]])))
#         self.depth = o3d.io.read_image(os.path.join(depth_dir_path, rgb_file[:-4] + "_vis.png"))

#         # now get the previous set of images etc. for the point cloud registration
#         if pos == 0 or pos == 100:
#             self.pose_prev = self.pose
#             self.rgb_prev = self.rgb
#             self.depth_prev = self.depth
#             self.seg_prev = self.seg
#         else:
#             self.pose_prev = loadmat(os.path.join(pose_dir_path, rgb_file_list[pos*2 - 2][:-6] + ".mat"))["RT_L"]
#             self.pose_prev = np.concatenate((self.pose_prev,np.array([[0,0,0,1]])))
#             self.rgb_prev = o3d.io.read_image(os.path.join(rgb_dir_path, rgb_file_list[pos*2 - 2]))
#             self.depth_prev = o3d.io.read_image(os.path.join(depth_dir_path, rgb_file_list[pos*2 - 2][:-4] + "_vis.png"))
#             self.seg_prev = o3d.io.read_image(os.path.join(seg_dir_path, rgb_file_list[pos*2 - 2][:-4] + "_Sem.png"))

#         # get relative pose; recall transpose of rotation matrix is inverse rotation; t2 to world @ world to t1
#         rel_pose = np.dot(utils.utils.invert_pose(self.pose), self.pose_prev)

#         # convert pose matrix to quaternion/position/orientation
#         quat = R.from_matrix(rel_pose[:3,:3]).as_quat()
#         translation =  rel_pose[:3,3]

#         # create Open3D RGBD image and point cloud
#         pcd = self.pcd_from_rgb_depth(self.rgb, self.depth)
#         pcd_prev = self.pcd_from_rgb_depth(self.rgb_prev, self.depth_prev)

#         # downsample using keypoint detection
# #         if self.use_keypt_downsample:
#         pcd = self.keypt_detector.process(keyPointAlgo=self.keypt_method, data=pcd)
#         pcd_prev = self.keypt_detector.process(keyPointAlgo=self.keypt_method, data=pcd_prev)
# #         else:
# #             # keypoint detectors return point cloud as ndarray; code below expects ndnarray
# #             pcd, pcd_prev = np.asarray(pcd), np.asarray(pcd_prev)

#         # normalize the point cloud
#         # pcd = self.process_pcd(pcd)
#         # pcd_prev = self.process_pcd(pcd_prev)

#         return {"pcd1":torch.from_numpy(pcd).double(),\
#                 "pcd2":torch.from_numpy(pcd_prev).double(),\
#                 "rel_pose":torch.from_numpy(rel_pose).double(),\
#                 "translation":torch.from_numpy(translation).double(),
#                 "quat":torch.from_numpy(quat).double(),
#                 "pose":torch.from_numpy(self.pose).double()
#         }
    
#     def __len__(self):
#         return self.dset_len