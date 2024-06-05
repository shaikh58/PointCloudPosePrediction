import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from core.pointnet import PointNetFeatureExtractor
torch.set_default_dtype(torch.float32)


def PoseLoss(pred, target):
    """Expects a length 7 vector; upper 3 elements are relative translation, 
    lower 4 are relative rotation in quaternion form"""
    
    pos_loss = torch.linalg.norm(target[:,:3] - pred[:,:3])
    # normalize predicted rotation vector to ensure valid quaternion; clamp to avoid zero div
    q_norm = torch.linalg.norm(pred[:,3:], dim=1).view(-1,1)
    rot_loss = torch.linalg.norm(target[:,3:] - (pred[:,3:]/torch.clamp(q_norm, min=1e-5)))
    
    return pos_loss + rot_loss
            

class PCA(nn.Module):
    def __init__(self, num_pc):
        super().__init__()
        self.num_pc = num_pc

    def forward(self, x):
        # TODO: set seed to ensure axes are the same every time
        U,S,V = torch.pca_lowrank(x)
        projected = torch.matmul(x, V[:,:,:self.num_pc])
        return projected


class DeepPoseEstimator(nn.Module):
    def __init__(self, num_pts):
        super().__init__()
        self.num_pc = 5
        self.num_pts = num_pts
        # separate learnable feature extractors for each input point cloud
        self.pcd_feat1 = PointNetFeatureExtractor(feature_transform=True)
        self.pcd_feat2 = PointNetFeatureExtractor(feature_transform=True)
        self.pca = PCA(self.num_pc)
        self.pca.requires_grad_(False) # frozen layer
        self.fc = nn.Linear(2*self.num_pc*self.num_pts,7,bias=True, dtype=torch.float32)

    def forward(self, pcd1, pcd2):
        # input pcd : batch_size x 3 x num_pts (transposed before input)
        pcd1_feat, trans_feat_1 = self.pcd_feat1(pcd1)
        pcd2_feat, trans_feat_2 = self.pcd_feat2(pcd2)
        # after feature extraction : batch_size x num_pts x 128
    
        pcd1_feat, pcd2_feat = pcd1_feat.permute((0,2,1)), pcd2_feat.permute((0,2,1))
        pcd1_feat = self.pca(pcd1_feat).permute(0,2,1)
        pcd2_feat = self.pca(pcd2_feat).permute(0,2,1)
        # after pca : batch_size x num_pc x num_pts
        
        # concatenate per-point feature vectors into single vector
        combined_pcd = torch.cat((pcd1_feat, pcd2_feat), dim=1).permute(0,2,1)
        # batch_size x num_pts x 2*num_pc
        combined_pcd = torch.flatten(combined_pcd, start_dim=1)
        # batch_size x num_pts*2*num_pc
        
        pose_vec = self.fc(combined_pcd)
        # after fully connected layer : batch_size x 7
        
        return pose_vec, trans_feat_1, trans_feat_2
