import torch
import torch.nn as nn
import numpy as np

# from escnn import gspaces
# from escnn import nn as enn
# from escnn import group

from .external.siren_pytorch import SirenNet

from .so3_decomposition import so3_point_group_operations
from pytorch3d.transforms import so3_rotation_angle, random_rotations, so3_relative_angle

class SymmetrizedFeature(torch.nn.Module):
    def __init__(self, point_group):
        super().__init__()
        self.register_buffer('symm_ops', so3_point_group_operations(point_group))
        
        # self.embed_fc = torch.nn.Sequential(
        #     torch.nn.Linear(3, 64),
        #     torch.nn.SiLU(),
        #     torch.nn.Linear(64, 3)
        # )
        self.embed_fc = SirenNet(
            dim_in=3,
            dim_hidden=32,
            dim_out=3,
            num_layers=2,
        )
        
    def forward(self, x):
        x_expanded = torch.einsum('gij, bj -> gbi', self.symm_ops, x)
        x_embeded  = self.embed_fc(x_expanded)
        x_symmetry = x_embeded.mean(dim=0)
        return x_symmetry
    

from pytorch3d.transforms import rotation_6d_to_matrix
class LearnableSymmetrizedFeature(torch.nn.Module):
    def __init__(self, N_ops):
        super().__init__()
        self.N_ops = N_ops
        self.register_parameter('symm_ops_6d', torch.nn.Parameter(torch.randn(self.N_ops, 6)))

        self.embed_fc = SirenNet(
            dim_in=3,
            dim_hidden=32,
            dim_out=3,
            num_layers=2,
        )
        
    def forward(self, x):
        symm_ops = rotation_6d_to_matrix(self.symm_ops_6d)
        x_expanded = torch.einsum('gij, bj -> gbi', symm_ops, x)
        x_embeded  = self.embed_fc(x_expanded)
        x_symmetry = x_embeded.mean(dim=0)
        return x_symmetry

class RotationFolding(torch.nn.Module):
    def __init__(self, point_group):
        super().__init__()
        self.register_buffer('symm_ops', so3_point_group_operations(point_group))
        
    def forward(self, rotations):
        expanded_rotations = torch.einsum('gij, bjk -> bgik', self.symm_ops, rotations)
        shape = expanded_rotations.shape[:2]
        
        # print(self.symm_ops.det().min(), rotations.det().min())
        # print(expanded_rotations.reshape(-1,3,3).det().min())
        angles = so3_rotation_angle(expanded_rotations.reshape(-1,3,3), eps=1e-2).view(shape)
        
        min_encoding_indices = torch.argmin(angles, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], len(self.symm_ops)).to(rotations.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        folded_rotations = torch.einsum('bg, bgik -> bik', min_encodings, expanded_rotations)
        
        folded_rotations = rotations + (folded_rotations - rotations).detach()
        
        return folded_rotations