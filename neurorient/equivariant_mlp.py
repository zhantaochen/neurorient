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
    
# class IcosahedralMLP(torch.nn.Module):
    
#     def __init__(self, ):
#         super().__init__()
        
#         # the model is equivariant to the icosahedral group
#         self.G = group.ico_group()
#         # self.G = group.dihedral_group(5)
#         # self.G = group.cyclic_group(5)
        
#         # since we are building an MLP, there is no base-space
#         self.gspace = gspaces.no_base_space(self.G)
        
#         try:
#             self.G.standard_representation()
#         except TypeError:
#             pass
        
#         # Define input type; using regular representation for simplicity
#         self.in_type = self.gspace.type(self.G._representations['standard'])
        
#         out_type_1 = self.gspace.type(self.G._representations['regular'])
#         self.linear_1 = enn.Linear(self.in_type, self.in_type, bias=True)
#         # self.activation_1 = enn.ReLU(out_type_1)
        
#         # self.linear_2 = enn.Linear(self.in_type, out_type_1, bias=True)
#         # self.activation_2 = enn.ReLU(out_type_1)
        
#         # self.linear_3 = enn.Linear(self.in_type, out_type_1, bias=True)
#         # self.activation_3 = enn.ReLU(out_type_1)
        
#         self.out_invariant_map = enn.NormPool(out_type_1)
        
#         # self.out_linear = nn.Sequential(
#         #     nn.Linear(3, 128, bias=True),
#         #     nn.SiLU(),
#         #     nn.Linear(128, 128, bias=True),
#         #     nn.SiLU(),
#         #     nn.Linear(128, 1, bias=True),
#         # )
        
#         self.out_linear = SirenNet(
#             dim_in=3,
#             dim_hidden=128,
#             dim_out=1,
#             num_layers=3,
#             final_activation=torch.nn.ReLU(),
#         )

#     def forward(self, x):
        
#         if not isinstance(x, enn.GeometricTensor):
#             x = self.in_type(x)
        
#         y1 = self.linear_1(x)
#         # y1 = self.activation_1(y1)
#         # y1 = self.out_invariant_map(y1).tensor
        
#         # y2 = self.linear_2(x)
#         # y2 = self.activation_2(y2)
#         # y2 = self.out_invariant_map(y2).tensor
        
#         # y3 = self.linear_3(x)
#         # y3 = self.activation_3(y3)
#         # y3 = self.out_invariant_map(y3).tensor
        
#         # y = torch.cat([y1, y2, y3], dim=-1)
#         y = self.out_linear(y1.tensor)
        
#         return y
