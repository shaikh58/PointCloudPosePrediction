import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from core.pointnet import PointNetFeatureExtractor
torch.set_default_dtype(torch.float64)


def PoseLoss(pred, target):
    """Expects a length 7 vector, upper 3 elements are relative position, 
    lower 4 are relative rotation in quaternion form"""
    return torch.linalg.norm(pred[:3] - target[:3]) + torch.linalg.norm(target[3:] - pred[3:]/np.linalg.norm(pred[3:]))


class PCA(nn.Module):
    def __init__(self, num_pc):
        super().__init()
        self.num_pc = num_pc

    def forward(self, x):
        U,S,V = torch.pca_lowrank(x)
        projected = torch.matmul(x, V[:,:,:self.num_pc])
        return projected


class DeepPoseEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        # separate learnable feature extractors for each input point cloud
        self.pcd_feat1 = PointNetFeatureExtractor(feature_transform=True)
        self.pcd_feat2 = PointNetFeatureExtractor(feature_transform=True)
        self.pca = PCA(num_pc=5)
        self.pca.requires_grad_(False) # frozen layer

    def forward(self, pcd1, pcd2):
        pcd1_feat, trans_feat_1 = self.pcd_feat1(pcd1)
        pcd2_feat, trans_feat_2 = self.pcd_feat2(pcd2)
        pcd1_pca = self.pca(pcd1_feat)
        pcd2_pca = self.pca(pcd2_feat)
        pose_vec = nn.Linear()
        return pose_vec, trans_feat_1, trans_feat_2
