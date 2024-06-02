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
    return torch.linalg.norm(pred[:3] - target[:3]) + torch.linalg.norm(pred[3:] - pred[3:])

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
        self.pcd_feat = PointNetFeatureExtractor(feature_transform=True)
        self.pca = PCA(num_pc=5)
        self.pca.requires_grad_(False)

    def forward(self, pcd1, pcd2):
        pcd1_feat = self.pcd_feat(pcd1)
        pcd2_feat = self.pcd_feat(pcd2)
        pcd1_pca = self.pca(pcd1_feat)
        pcd2_pca = self.pca(pcd2_feat)

        return pcd1_feat
