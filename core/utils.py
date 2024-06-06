import torch
import torch.nn as nn


class PCA(nn.Module):
    def __init__(self, num_pc : int, use_pca : bool):
        super().__init__()
        self.num_pc = num_pc
        self.use_pca = use_pca

    def forward(self, x):
        if self.use_pca:
            U,S,V = torch.pca_lowrank(x)
            projected = torch.matmul(x, V[:,:,:self.num_pc])
            return projected
        else:
            return x.permute(0,2,1)