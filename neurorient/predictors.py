import re
import torch
import time
from torch import Tensor
import numpy as np
import torch.nn as nn
import e3nn
from e3nn import o3
import warnings
import torchvision.models.resnet as resnet
from pytorch3d.transforms import rotation_6d_to_matrix

from .encoders.encoders import ImageEncoder, SteerableCNN
from .encoders.e2wrn import Wide_ResNet

from .external.image2sphere import so3_utils
from .external.image2sphere import (
    SpatialS2Projector,
    HarmonicS2Projector,
    SpatialS2Features,
    HarmonicS2Features,
    ResNet,
    SO3Convolution,
)

from .utils_transform import weighted_average_matrices

class BaseSO3Predictor(nn.Module):
    def __init__(self,
                 num_classes: int=1,
                 encoder: str='resnet18',
                 encoder_input_shape: tuple=(3, 224, 224),
                 pool_features: bool=False,
                 *args,
                 **kwargs
                ):
        super().__init__()
        self.num_classes = num_classes

        if encoder.find('resnet') > -1:
            pretrained = encoder.find('pretrained') > -1
            size = int(re.findall('\d+', encoder)[0])
            self.encoder = ResNet(size, pretrained, pool_features)
        elif encoder.find('swin') > -1:
            self.encoder = ImageEncoder()
        elif encoder.find('steerablecnn') > -1:
            self.encoder = SteerableCNN(input_shape=encoder_input_shape)
        elif encoder.find('e2wrn') > -1:
            self.encoder = Wide_ResNet(
                16, 1, 0.0, num_classes=-1, initial_stride=1, N=8, f=False, r=0
            ) 

        dummy_input = torch.zeros((1,) + tuple(encoder_input_shape))
        self.encoder.output_shape = self.encoder(dummy_input).shape[1:]
        
    def save(self, path):
        torch.save(self.state_dict(), path)


class ResNet2Rotmat(nn.Module):
    def __init__(self, size=50, pretrained=False, pool_features=False):
        super().__init__()
        weights = 'DEFAULT' if pretrained else None
        self.resnet = eval(f'resnet.resnet{size}')(weights=weights)

        # remove pool and linear
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 6)
        
    
    def forward(self, img):
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        embed = self.resnet(img)
        rotmat = rotation_6d_to_matrix(embed)
        return rotmat
        
class I2S(BaseSO3Predictor):
    def __init__(self,
                 num_classes: int=1,
                 sphere_fdim: int=512,
                 encoder: str='resnet50_pretrained',
                 encoder_input_shape: tuple=(3, 224, 224),
                 projection_mode='spatialS2',
                 feature_sphere_mode='harmonicS2',
                 lmax: int=6,
                 f_hidden: int=8,
                 train_grid_rec_level: int=3,
                 train_grid_n_points: int=4096,
                 train_grid_include_gt: bool=False,
                 train_grid_mode: str='healpix',
                 eval_grid_rec_level: int=5,
                 eval_use_gradient_ascent: bool=False,
                 eval_wigners_file: str=None,
                 include_class_label: bool=False,
                ):
        super().__init__(num_classes, encoder, encoder_input_shape, pool_features=False)

        proj_input_shape = list(self.encoder.output_shape)
        self.include_class_label = include_class_label
        if self.include_class_label:
            proj_input_shape[0] += num_classes

        #projection stuff
        self.projector = {
            'spatialS2' : SpatialS2Projector,
            'harmonicS2' : HarmonicS2Projector,
        }[projection_mode](proj_input_shape, sphere_fdim, lmax)

		#spherical conv stuff
        self.feature_sphere = {
            'spatialS2' : SpatialS2Features,
            'harmonicS2' : HarmonicS2Features,
        }[feature_sphere_mode](sphere_fdim, lmax, f_out=f_hidden)

        self.lmax = lmax
        irreps_in = so3_utils.s2_irreps(lmax)
        self.o3_conv = o3.Linear(irreps_in, so3_utils.so3_irreps(lmax),
                                 f_in=sphere_fdim, f_out=f_hidden, internal_weights=False)

        self.so3_activation = e3nn.nn.SO3Activation(lmax, lmax, torch.relu, 10)
        so3_grid = so3_utils.so3_near_identity_grid()
        self.so3_conv = SO3Convolution(f_hidden, 1, lmax, so3_grid)

        # output rotations for training and evaluation
        self.train_grid_rec_level = train_grid_rec_level
        self.train_grid_n_points = train_grid_n_points
        self.train_grid_include_gt = train_grid_include_gt
        self.train_grid_mode = train_grid_mode
        self.eval_grid_rec_level = eval_grid_rec_level
        self.eval_use_gradient_ascent = eval_use_gradient_ascent

        output_xyx = so3_utils.so3_healpix_grid(rec_level=train_grid_rec_level)
        self.register_buffer(
            "output_wigners", so3_utils.flat_wigner(lmax, *output_xyx).transpose(0,1)
        )
        self.register_buffer(
            "output_rotmats", o3.angles_to_matrix(*output_xyx)
        )

        output_xyx = so3_utils.so3_healpix_grid(rec_level=eval_grid_rec_level)
        if eval_wigners_file is not None:
            self.eval_wigners = torch.load(eval_wigners_file)
        else:
            if eval_grid_rec_level >= 5:
                warnings.warn('eval_grid_rec_level >= 5 is very slow, consider using eval_wigners_file')
            self.eval_wigners = so3_utils.flat_wigner(lmax, *output_xyx).transpose(0,1)

        self.eval_rotmats = o3.angles_to_matrix(*output_xyx)

    def forward(self, x, o):
        x = self.encoder(x)
        if self.include_class_label:
            o_oh = nn.functional.one_hot(o.squeeze(1), num_classes=self.num_classes)
            o_oh_fmap = o_oh.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.size(-2), x.size(-1))
            x = torch.cat((x, o_oh_fmap), dim=1)

        x = self.projector(x)

        weight, _ = self.feature_sphere()
        x = self.o3_conv(x, weight=weight)

        x = self.so3_activation(x)

        x = self.so3_conv(x)

        return x

    def query_train_grid(self, x, gt_rot=None):
        '''x is signal over fourier basis'''
        if self.train_grid_mode == 'random':
            idx = torch.randint(len(self.output_rotmats), (self.train_grid_n_points,))

            wigners = self.output_wigners[:,idx]
            rotmats = self.output_rotmats[idx]
            if self.train_grid_include_gt:
                # creating wigners is slightly faster on cpu
                try:
                    abg = o3.matrix_to_angles(gt_rot.cpu())
                    wigners[:,:gt_rot.size(0)] = so3_utils.flat_wigner(self.lmax, *abg).transpose(0,1).to(x.device)
                    rotmats[:gt_rot.size(0)] = gt_rot
                except AssertionError:
                    # sometimes dataloader generates invalid rot matrix according to o3
                    pass

        elif self.train_grid_mode == 'healpix':
            wigners = self.output_wigners
            rotmats = self.output_rotmats

        return torch.matmul(x, wigners).squeeze(1), rotmats

    def predict(self, x, o, lr=1e-3, n_iters=10):
        with torch.no_grad():
            fourier = self.forward(x, o)
            fourier = fourier.cpu()
            probs = torch.matmul(fourier, self.eval_wigners).squeeze(1)
            pred_id = probs.max(dim=1)[1]
            rots = self.eval_rotmats[pred_id]

        if self.eval_use_gradient_ascent:
            a,b,g = o3.matrix_to_angles(rots)
            a.requires_grad = True
            b.requires_grad = True
            g.requires_grad = True
            for _ in range(n_iters):
                wigners = so3_utils.flat_wigner(self.lmax, a,b,g).transpose(0,1)
                val = torch.diagonal(torch.matmul(fourier, wigners).squeeze(1))
                da, db, dg = torch.autograd.grad(val.mean(), (a, b, g))
                a = a + lr * da
                b = b + lr * db
                g = g + lr * dg
            rots = o3.angles_to_matrix(a, b, g).detach()

        return rots

    def compute_average_rotmats(self, x, o):
        ''' compute probabilities over eval grid'''
        harmonics = self.forward(x, o)
        probs = torch.matmul(harmonics, o).squeeze(1)
        probs = nn.Softmax(dim=1)(probs)
        
        # output_rotmats: [n_points, 3, 3]
        # probs: [batch_size, n_points]
        avg_rotmat = weighted_average_matrices(self.output_rotmats, probs)
        
        return avg_rotmat
    
    @torch.no_grad()
    def compute_probabilities(self, x, o, use_o_for_eval=False):
        ''' compute probabilities over eval grid'''
        harmonics = self.forward(x, o)

        # move to cpu to avoid memory issues, at expense of speed
        harmonics = harmonics.cpu()

        if use_o_for_eval:
            probs = torch.matmul(harmonics, o.cpu()).squeeze(1)
        else:
            probs = torch.matmul(harmonics, self.eval_wigners.cpu()).squeeze(1)

        return nn.Softmax(dim=1)(probs)

    def compute_loss(self, img, cls, rot):
        x = self.forward(img, cls)
        grid_signal, rotmats = self.query_train_grid(x, rot)

        rot_id = so3_utils.nearest_rotmat(rot, rotmats)
        loss = nn.CrossEntropyLoss()(grid_signal, rot_id)

        with torch.no_grad():
            pred_id = grid_signal.max(dim=1)[1]
            pred_rotmat = rotmats[pred_id]
            acc = so3_utils.rotation_error(rot, pred_rotmat)

        return loss, acc.cpu().numpy()
