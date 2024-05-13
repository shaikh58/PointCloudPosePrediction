import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import open3d as o3d
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from typing import Any, Callable, Optional, Tuple
from utils.utils import invert_pose


class EdenDataset(Dataset):
    def __init__(self, data_dir,
                 train: bool = True, transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None) -> None:
        
        self.data_dir = data_dir
        self.sequences = sorted(os.listdir(os.path.join(data_dir,"rgb"))[1:]) # 0001, 0003, etc.
        self.dict_seq_len = {seq:len(os.listdir(os.path.join(data_dir, "poses", seq, "clear"))) 
                             for seq in self.sequences}
        self.seq_len = 500  # assume Eden dataset has 500 data points per sequence
        self.dset_len = 0
        for k,v in self.dict_seq_len.items():
            self.dset_len += v
        
        self.cam_width, self.cam_height = 640, 480
        self.cam_instrinsics = np.array([[640,0,320],[0,640,240],[0,0,1]])
        self.transform = transform
        self.target_transform = target_transform
    
    def pcd_from_rgb_depth(self, rgb, depth) -> o3d.geometry.PointCloud:
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,
        o3d.camera.PinholeCameraIntrinsic(self.cam_width, self.cam_height, self.cam_instrinsics))
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        return pcd
    
    def process_pcd(self, pcd) -> np.ndarray:
        pcd = pcd - np.expand_dims(np.mean(pcd, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(pcd ** 2, axis=1)), 0)
        pcd = pcd / dist  # scale
        return pcd

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
        
        self.rgb = o3d.io.read_image(os.path.join(rgb_dir_path, rgb_file))
        self.seg = o3d.io.read_image(os.path.join(seg_dir_path, rgb_file[:-4] + "_Sem.png"))
        self.pose = loadmat(os.path.join(pose_dir_path, rgb_file[:-6] + ".mat"))["RT_L"]
        self.pose = np.concatenate((self.pose,np.array([[0,0,0,1]])))
        self.depth = o3d.io.read_image(os.path.join(depth_dir_path, rgb_file[:-4] + "_vis.png"))

        # now get the previous set of images etc. for the point cloud registration
        if pos == 0:
            self.pose_prev = self.pose
            self.rgb_prev = self.rgb
            self.depth_prev = self.depth
            self.seg_prev = self.seg
        else:
            self.pose_prev = loadmat(os.path.join(pose_dir_path, rgb_file_list[pos*2 - 2][:-6] + ".mat"))["RT_L"]
            self.pose_prev = np.concatenate((self.pose_prev,np.array([[0,0,0,1]])))
            self.rgb_prev = o3d.io.read_image(os.path.join(rgb_dir_path, rgb_file_list[pos*2 - 2]))
            self.depth_prev = o3d.io.read_image(os.path.join(depth_dir_path, rgb_file_list[pos*2 - 2][:-4] + "_vis.png"))
            self.seg_prev = o3d.io.read_image(os.path.join(seg_dir_path, rgb_file_list[pos*2 - 2][:-4] + "_Sem.png"))

        # get relative pose; recall transpose of rotation matrix is inverse rotation; t2 to world @ world to t1
        rel_pose = invert_pose(self.pose) @ self.pose_prev
        # convert pose matrix to quaternion/position/orientation
        quat = R.from_matrix(rel_pose[:3,:3]).as_quat()
        translation =  rel_pose[:3,3]

        # create Open3D RGBD image and point cloud
        pcd = self.pcd_from_rgb_depth(self.rgb, self.depth)
        pcd_prev = self.pcd_from_rgb_depth(self.rgb_prev, self.depth_prev)
        
        # normalize the point cloud
        pcd = self.process_pcd(np.asarray(pcd.points))
        pcd_prev = self.process_pcd(np.asarray(pcd_prev.points))

        return {"pcd1":torch.from_numpy(pcd.astype(np.float32)),\
                "pcd2":torch.from_numpy(pcd_prev.astype(np.float32)),\
                "translation":torch.from_numpy(translation.astype(np.float32)),
                "quat":torch.from_numpy(quat.astype(np.float32))
        }
    
    def __len__(self):
        return self.dset_len