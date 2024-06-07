import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from core.pointnet import PointNetFeatureExtractor
from core.utils import PCA
torch.set_default_dtype(torch.float32)
            

# version: run 1,2: 2 pointnet FE + linear layer
# class DeepPoseEstimator(nn.Module):
#     def __init__(self, num_pts, use_pca):
#         super().__init__()
#         self.num_pc = 0
#         self.use_pca = use_pca
#         if self.use_pca:
#             self.num_pc = 5
#         self.num_pts = num_pts
#         # separate learnable feature extractors for each input point cloud
#         self.pcd_feat1 = PointNetFeatureExtractor(feature_transform=True)
#         self.pcd_feat2 = PointNetFeatureExtractor(feature_transform=True)
#         self.pca = PCA(self.num_pc, self.use_pca)
#         self.pca.requires_grad_(False) # frozen layer
#         self.fc = nn.Linear(2*self.num_pc*self.num_pts,7,bias=True, dtype=torch.float32)

#     def forward(self, pcd1, pcd2):
#         # input pcd : batch_size x 3 x num_pts (transposed before input)
#         pcd1_feat, trans_feat_1 = self.pcd_feat1(pcd1)
#         pcd2_feat, trans_feat_2 = self.pcd_feat2(pcd2)
#         # after feature extraction : batch_size x num_pts x 128
    
#         pcd1_feat, pcd2_feat = pcd1_feat.permute((0,2,1)), pcd2_feat.permute((0,2,1))
#         pcd1_feat = self.pca(pcd1_feat).permute(0,2,1)
#         pcd2_feat = self.pca(pcd2_feat).permute(0,2,1)
#         # after pca : batch_size x num_pc x num_pts
        
#         # concatenate per-point feature vectors into single vector
#         combined_pcd = torch.cat((pcd1_feat, pcd2_feat), dim=1).permute(0,2,1)
#         # batch_size x num_pts x 2*num_pc
#         combined_pcd = torch.flatten(combined_pcd, start_dim=1)
#         # batch_size x num_pts*2*num_pc
        
#         pose_vec = self.fc(combined_pcd)
#         # after fully connected layer : batch_size x 7
        
#         return pose_vec, trans_feat_1, trans_feat_2


# version: run 3,4,5,6: single pointnet FE + single 1x1 conv + linear layer
class DeepPoseEstimator(nn.Module):
    def __init__(self, num_pts, use_pca):
        super().__init__()
        
        self.use_pca = use_pca
        if self.use_pca:
            self.num_pc = 5
        else:
            self.num_pc = 128
        self.num_pts = num_pts
        
        # single learnable feature extractors for both input point clouds
        self.pcd_feat = PointNetFeatureExtractor(feature_transform=True)
        self.pca = PCA(self.num_pc, self.use_pca)
        self.pca.requires_grad_(False) # frozen layer
        
        # 1x1 conv on 5 num input channels x num_pts vector; conv fully connected layer
        self.out_channels_1 = 1
        self.conv1 = nn.Conv1d(self.num_pc, self.out_channels_1, 1)
        self.bn1 = nn.BatchNorm1d(self.out_channels_1)
        self.relu = nn.ReLU()
        
        # final regression layer
        self.fc = nn.Linear(2*self.num_pts,7,bias=True, dtype=torch.float32)

    def forward(self, pcd1, pcd2):
        
        # input pcd : batch_size x 3 x num_pts (transposed before input)
        pcd1_feat, trans_feat_1 = self.pcd_feat(pcd1)
        pcd2_feat, trans_feat_2 = self.pcd_feat(pcd2)
        
        # after feature extraction : batch_size x 128 x num_pts
        pcd1_feat, pcd2_feat = pcd1_feat.permute((0,2,1)), pcd2_feat.permute((0,2,1))
        pcd1_feat = self.pca(pcd1_feat).permute(0,2,1)
        pcd2_feat = self.pca(pcd2_feat).permute(0,2,1)
        # after pca : batch_size x num_pc x num_pts
        
        pcd1_feat = self.relu(self.bn1(self.conv1(pcd1_feat)))
        pcd2_feat = self.relu(self.bn1(self.conv1(pcd2_feat)))
        # after 1x1 conv : batch_size x 1 x num_pts (each point is represented by a scalar)

        # concatenate per-point scalar into single vector
        combined_pcd = torch.cat((pcd1_feat, pcd2_feat), dim=2).permute(0,2,1)

        # batch_size x 2*num_pts x 1
        combined_pcd = torch.flatten(combined_pcd, start_dim=1)
        # remove redundant dim at the end; batch_size x 2*num_pts
        
        pose_vec = self.fc(combined_pcd)
        # after fully connected layer : batch_size x 7
        
        return pose_vec, trans_feat_1, trans_feat_2
    


# version: dummy model for debugging
# class DeepPoseEstimator(nn.Module):
#     def __init__(self, num_pts, use_pca):
#         super().__init__()
        
#         self.use_pca = use_pca
#         if self.use_pca:
#             self.num_pc = 5
#         else:
#             self.num_pc = 128
#         self.num_pts = num_pts
        
# #         # single learnable feature extractors for both input point clouds
# #         self.pcd_feat = PointNetFeatureExtractor(feature_transform=True)
# #         self.pca = PCA(self.num_pc, self.use_pca)
# #         self.pca.requires_grad_(False) # frozen layer
        
# #         # 1x1 conv on 5 num input channels x num_pts vector; conv fully connected layer
# #         self.out_channels_1 = 1
# #         self.conv1 = nn.Conv1d(self.num_pc, self.out_channels_1, 1)
# #         self.bn1 = nn.BatchNorm1d(self.out_channels_1)
# #         self.relu = nn.ReLU()
        
# #         # final regression layer
#         self.fc = nn.Linear(2*self.num_pts,7,bias=True, dtype=torch.float32)

#     def forward(self, pcd1, pcd2):
#         N = pcd1.shape[0]
#         return torch.ones((N,7),device='cuda',requires_grad=True), torch.ones((N,64,64),device='cuda',requires_grad=True), torch.ones((N,64,64),device='cuda',requires_grad=True)