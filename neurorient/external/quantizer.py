"""
https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from pytorch3d.transforms import 

from .image2sphere.so3_utils import so3_healpix_grid
from ..so3_relative_angle import so3_relative_angle
from pytorch3d.transforms import euler_angles_to_matrix


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, beta, rec_level=2):
        super().__init__()
        self.beta = beta

        euler_yxy = so3_healpix_grid(rec_level).T
        # self.register_parameter('embedding', torch.nn.Parameter(euler_angles_to_matrix(euler_yxy, convention='YXY')))
        self.register_buffer('embedding', euler_angles_to_matrix(euler_yxy, convention='YXY'))
        
        self.n_e = self.embedding.shape[0]
        
        # self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def distance_func(self, z_flattened):
        return so3_relative_angle(z_flattened, self.embedding)
    
    def distance_func_l2(self, z_flattened):
        z_flattened = z_flattened.view(z_flattened.shape[0], -1)
        z_embedding = self.embedding.view(self.embedding.shape[0], -1)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(z_embedding**2, dim=1) - 2 * \
            torch.matmul(z_flattened, z_embedding.t())
        return d
    
    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        
        d = self.distance_func(z)
        
        # print(d.shape)
        
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        # z_q = torch.matmul(min_encodings, self.embedding).view(z.shape)
        z_q = torch.einsum('bn, nij->bij', min_encodings, self.embedding)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        weights = F.gumbel_softmax(100 * d, dim=-1)
        # print(weights.max())
        # perplexity
        e_mean = torch.mean(weights, dim=0)
        e_mean = e_mean / e_mean.sum()
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
    
    
    
    

# class VectorQuantizer(nn.Module):
#     """
#     Discretization bottleneck part of the VQ-VAE.

#     Inputs:
#     - n_e : number of embeddings
#     - e_dim : dimension of embedding
#     - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
#     """

#     def __init__(self, beta, rec_level=2):
#         super().__init__()
#         self.beta = beta

#         euler_yxy = so3_healpix_grid(rec_level).T
#         # self.register_parameter('embedding', torch.nn.Parameter(euler_angles_to_matrix(euler_yxy, convention='YXY')))
#         self.register_buffer('embedding', euler_angles_to_matrix(euler_yxy, convention='YXY'))
        
#         self.n_e = self.embedding.shape[0]
        
#         # self.embedding = nn.Embedding(self.n_e, self.e_dim)
#         # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

#     def distance_func(self, z_flattened):
#         return so3_relative_angle(z_flattened, self.embedding)
    
#     def forward(self, z):
#         """
#         Inputs the output of the encoder network z and maps it to a discrete 
#         one-hot vector that is the index of the closest embedding vector e_j

#         z (continuous) -> z_q (discrete)

#         z.shape = (batch, channel, height, width)

#         quantization pipeline:

#             1. get encoder input (B,C,H,W)
#             2. flatten input to (B*H*W,C)

#         """
        
#         d = self.distance_func(z)
        
#         # print(d.shape)
        
#         # find closest encodings
#         min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
#         min_encodings = torch.zeros(
#             min_encoding_indices.shape[0], self.n_e).to(z.device)
#         min_encodings.scatter_(1, min_encoding_indices, 1)

#         # get quantized latent vectors
#         # z_q = torch.matmul(min_encodings, self.embedding).view(z.shape)
#         z_q = torch.einsum('bn, nij->bij', min_encodings, self.embedding)

#         # compute loss for embedding
#         loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
#             torch.mean((z_q - z.detach()) ** 2)

#         # preserve gradients
#         z_q = z + (z_q - z).detach()

#         # perplexity
#         e_mean = torch.mean(min_encodings, dim=0)
#         perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

#         return loss, z_q, perplexity, min_encodings, min_encoding_indices