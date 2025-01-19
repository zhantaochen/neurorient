import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
import torchvision.models.resnet as resnet

from tqdm import tqdm

import scipy

import os

import lightning as L
import numpy as np
from torchkbnufft import KbNufft
from pathlib import Path
from copy import deepcopy

from .image_encoder import ImageEncoder
from .bifpn import DepthwiseSeparableConv2d, BiFPN
from pytorch3d.transforms import rotation_6d_to_matrix, euler_angles_to_matrix, random_rotations

from .external.siren_pytorch import SirenNet
from .external.image2sphere.so3_utils import so3_healpix_grid
from .reconstruction.slicing import get_real_mesh, gen_nonuniform_normalized_positions
from .utils_visualization import display_images_in_parallel, display_volumes, save_mrc
from .lr_scheduler import CosineLRScheduler
from .so3_decomposition import so3_point_group_operations
from .equivariant_mlp import SymmetrizedFeature, RotationFolding

from .external.quantizer import VectorQuantizer

class KbNufftRealView(KbNufft):
    def __init__(self, im_size, grid_size = None):
        super(KbNufftRealView, self).__init__(im_size, grid_size)

        # Convert all buffers to real view
        for name, buf in self.named_buffers():
            if (buf.dtype != torch.complex128) and (buf.dtype != torch.complex64): continue
            real_view_buf = torch.view_as_real(buf)
            self.register_buffer(name, real_view_buf)

class Slice2RotMat(nn.Module):
    def __init__(self, size=18, pretrained=False):
        super().__init__()
        weights = 'DEFAULT' if pretrained else None
        self.resnet = eval(f'resnet.resnet{size}')(weights=weights)

        # Average the weights in the input channels...
        conv1_weight = self.resnet.conv1.weight.data.mean(dim = 1, keepdim = True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.conv1.weight.data = conv1_weight
        # Output 6D rotation matrix
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 6)
    
    def forward(self, img):
        if img.ndim == 3:
            img = img.unsqueeze(1)
        embed = self.resnet(img)
        rotmat = rotation_6d_to_matrix(embed)
        return rotmat

import trimesh
from pytorch3d.transforms import axis_angle_to_matrix
def so3_grid_by_icosphere(icosphere_subdivisions=3, angle_samples=120):
    grid_axis = torch.tensor(trimesh.creation.icosphere(subdivisions=icosphere_subdivisions, radius=1.0).vertices.tolist()).float()
    grid_angles = (torch.linspace(0, np.pi, angle_samples+1)[:-1] + torch.linspace(0, np.pi, angle_samples+1)[1:]) / 2
    print(f'number of axes: {grid_axis.size(0)}, number of angles: {grid_angles.size(0)}')
    axis_angle = grid_axis.repeat_interleave(grid_angles.size(0), dim=0) * grid_angles.tile(grid_axis.size(0)).unsqueeze(-1)
    return axis_angle_to_matrix(axis_angle)

class Slice2RotMat_Logits(nn.Module):
    def __init__(self, size=18, pretrained=False, icosphere_subdivisions=3, angle_samples=120):
        super().__init__()

        # Create a grid of rotations
        self.register_buffer('output_rotmats', so3_grid_by_icosphere(icosphere_subdivisions=icosphere_subdivisions, angle_samples=angle_samples))
        self.num_output_rotmats = self.output_rotmats.size(0)

        # Define the resnet
        weights = 'DEFAULT' if pretrained else None
        self.resnet = eval(f'resnet.resnet{size}')(weights=weights)
        conv1_weight = self.resnet.conv1.weight.data.mean(dim = 1, keepdim = True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.conv1.weight.data = conv1_weight
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_output_rotmats)
    
    def forward(self, img):
        if img.ndim == 3:
            img = img.unsqueeze(1)
        logits = self.resnet(img)
        return logits
    
class FluctuationPredictor(nn.Module):
    def __init__(self, size=18, pretrained=False):
        super().__init__()
        weights = 'DEFAULT' if pretrained else None
        self.resnet = eval(f'resnet.resnet{size}')(weights=weights)

        # Average the weights in the input channels...
        conv1_weight = self.resnet.conv1.weight.data.mean(dim = 1, keepdim = True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.conv1.weight.data = conv1_weight
        # Output 6D rotation matrix
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        self.output_relu = nn.ReLU()
    
    def forward(self, img):
        if img.ndim == 3:
            img = img.unsqueeze(1)
        output = self.resnet(img)
        output = self.output_relu(output)
        return output
    
class IntensityNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net_mag = SirenNet(*args, **kwargs)

    def forward(self, x):
        return self.net_mag(x) + self.net_mag(-x)

# class IntensityNet(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.sym_net = SymmetrizedFeature('I')
#         self.net_mag = SirenNet(*args, **kwargs)

#     def forward(self, x):
#         x_symm = self.sym_net(x)
#         y_symm = self.net_mag(x_symm)

#         return y_symm 

from .encoder_i2s import I2S
import time
class NeurOrient(nn.Module):
    def __init__(self, 
                 pixel_position_reciprocal, 
                 over_sampling=1,
                 photons_per_pulse=1e13,
                 use_bifpn=False,
                 rec_level=4,
                 use_fluctuation_predictor=True,
                 config_slice2rotmat={'size': 18, 'pretrained': True},
                 config_intensitynet={'dim_hidden': 256, 'num_layers': 5},):
        super().__init__()

        self.register_buffer('pixel_position_reciprocal', 
                             pixel_position_reciprocal if isinstance(pixel_position_reciprocal, torch.Tensor) 
                                                       else torch.from_numpy(pixel_position_reciprocal)
                            )
        # only works for square images for now
        self.register_buffer('image_dimension', torch.tensor(self.pixel_position_reciprocal.shape[1]))
        # register real and reciprocal mesh
        real_mesh, reciprocal_mesh = get_real_mesh(self.image_dimension, self.pixel_position_reciprocal.max(), return_reciprocal=True)
        self.register_buffer('grid_position_reciprocal', reciprocal_mesh)
        self.register_buffer('grid_position_real', real_mesh)
        del real_mesh, reciprocal_mesh

        self.over_sampling = over_sampling
            
        # self.orientation_predictor = Slice2RotMat_Logits(**config_slice2rotmat)
        self.orientation_predictor = I2S(rec_level=3, input_size=self.pixel_position_reciprocal.shape[1])
        if use_fluctuation_predictor:
            self.fluctuation_predictor = FluctuationPredictor(config_slice2rotmat['size'], config_slice2rotmat['pretrained'])
        else:
            self.fluctuation_predictor = None

        # setup volume predictor
        self.volume_predictor = IntensityNet(
            dim_in=3,
            dim_hidden=config_intensitynet['dim_hidden'],
            dim_out=1,
            num_layers=config_intensitynet['num_layers'],
            # final_activation=torch.nn.ReLU(),
        )

        self.photons_per_pulse = photons_per_pulse
        self.loss_scale_factor = 1e14 / self.photons_per_pulse

    def image_to_orientation(self, x):
        rotmats = self.orientation_predictor(x)
        return rotmats

    def predict_intensity(self, grid_position_reciprocal):
        if grid_position_reciprocal.ndim > 2 and grid_position_reciprocal.shape[-1] == 3:
            out_shape = grid_position_reciprocal.shape[:-1]
            grid_position_reciprocal = grid_position_reciprocal.view(-1, 3)
            intensity = self.volume_predictor(grid_position_reciprocal).view(out_shape)
        else:
            intensity = self.volume_predictor(grid_position_reciprocal)
        return intensity

    def predict_slice(self, grid_position_reciprocal):
        if grid_position_reciprocal.ndim == 2 and grid_position_reciprocal.shape[0] == 3:
            grid_position_reciprocal = grid_position_reciprocal.T
        slices = self.volume_predictor(grid_position_reciprocal)
        return slices


    def estimate(self, batch, return_reconstruction=False):
        slices_input = torch.log(batch['input_mask'] * batch['image'] * self.loss_scale_factor + 1e-12)

        # predict orientations from images
        orientations = self.image_to_orientation(slices_input)
        if not return_reconstruction:
            return orientations
        else:
            # get reciprocal positions based on orientations
            # HKL has shape (3, num_qpts)
            HKL = gen_nonuniform_normalized_positions(
                orientations, self.pixel_position_reciprocal, self.over_sampling)
            # predict slices from HKL
            slices_pred = self.predict_slice(HKL).view((-1, 1,) + (self.image_dimension,)*2)

            return orientations, torch.exp(slices_pred) / self.loss_scale_factor


    def forward(self, x):
        slices_true = x

        slices_true = slices_true * self.loss_scale_factor + 1.
        slices_input = torch.log(slices_true)

        # predict orientations from images
        orientations = self.image_to_orientation(slices_input)
        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations, self.pixel_position_reciprocal, self.over_sampling)
        # predict slices from HKL
        slices_pred = self.predict_slice(HKL).view((-1, 1,) + (self.image_dimension,)*2)

        return slices_pred

scheduler_dict = {
    'CosineLRScheduler': CosineLRScheduler,
}

class NeurOrientLightning(L.LightningModule):
    
    def __init__(self, 
                 pixel_position_reciprocal, 
                 over_sampling=1,
                 photons_per_pulse=1e13,
                 use_bifpn=False,
                 use_fluctuation_predictor=True,
                 config_slice2rotmat={'size': 18, 'pretrained': True},
                 config_intensitynet={'dim_hidden': 256, 'num_layers': 3},
                 config_optimization={'lr': 1e-3, 'weight_decay': 1e-4, 'loss_func': 'MSELoss'}):
        super().__init__()
        self.save_hyperparameters()
        
        
        
        self.model = NeurOrient(
            pixel_position_reciprocal, 
            over_sampling=over_sampling,
            photons_per_pulse=photons_per_pulse,
            use_bifpn=use_bifpn,
            use_fluctuation_predictor=use_fluctuation_predictor,
            config_slice2rotmat=config_slice2rotmat,
            config_intensitynet=config_intensitynet,
        )
        
        self.configure_optimization = config_optimization
        if config_optimization['loss_func'] != 'PoissonNLLLoss':
            self.loss_func = eval(f"torch.nn.{config_optimization['loss_func']}()")
            self.log_transform = True
        else:
            self.loss_func = torch.nn.PoissonNLLLoss(log_input=False, full=True)
            self.log_transform = False
    
    @torch.no_grad()
    def generate_reference_slices(self, ):
        with torch.no_grad():
            # print(HKL.shape)
            rotation_dset = TensorDataset(self.model.orientation_predictor.output_rotmats)
            rotation_dloader = DataLoader(rotation_dset, batch_size=75, shuffle=False)
            _ref_images = []
            for batch in tqdm(rotation_dloader, miniters=int(len(rotation_dloader)/10)):
                _rotations = batch[0].to(self.device)
                HKL = gen_nonuniform_normalized_positions(
                    _rotations, self.model.pixel_position_reciprocal, self.model.over_sampling).T
                _intens = self.model.volume_predictor(HKL)
                _ref_images.append(_intens.view((-1, ) + (self.model.image_dimension,)*2))
        self.ref_images = torch.cat(_ref_images, dim=0)
        print(f'\nReference images generated')
    
    @torch.no_grad()
    def compute_pearson_correlation(self, slices_pred, slices_target):
        slices_pred = slices_pred.view(slices_pred.size(0), -1)
        slices_target = slices_target.view(slices_target.size(0), -1)

        slices_pred = slices_pred - slices_pred.mean(dim=-1, keepdim=True)
        slices_target = slices_target - slices_target.mean(dim=-1, keepdim=True)

        numerator = torch.einsum('bi, ni -> bn', slices_pred, slices_target)
        divisor   = torch.einsum('b, n -> bn', slices_pred.norm(dim=-1).pow(2), 
                                               slices_target.norm(dim=-1).pow(2)).sqrt()

        return numerator / (divisor + 1e-12)
    
    def normalize_to_range(self, x, min_val=0.0, max_val=2*np.pi):
        return (x - x.amin(dim=-1, keepdim=True)) / (x.amax(dim=-1, keepdim=True) - x.amin(dim=-1, keepdim=True) + 1e-8) * (max_val - min_val) + min_val
    
    @torch.no_grad()
    def compute_probs_based_on_correlation(self, slices_pred, slices_target, tau=1.):
        pearson_correlation = self.compute_pearson_correlation(slices_pred, slices_target)
        # probs = torch.softmax(pearson_correlation / tau, dim=-1)
        dist_norm = self.normalize_to_range(1.0 - pearson_correlation, 0.0, tau)
        probs = nn.functional.softmax(- dist_norm, dim=-1)
        return probs

    def get_dist_max(self, x):
        return 100. - 99.9 * np.exp(-0.1 * x)

    def on_train_epoch_start(self, *args, **kwargs):
        self.generate_reference_slices()

    def get_tau(self, x):
        return min(10, max(0.01, 10 - 0.1 * x))

    def gumbel_softmax_temperature(self, x):
        return min(2, max(0.1, 2 - 0.02 * x))
        
    def training_step(self, batch, batch_idx):
        
        task_type = 'train'
        
        if isinstance(batch, dict):
            slices_true = batch['image'].to(self.dtype)
            input_mask  = batch['input_mask'].bool()
            general_mask = batch['general_mask'].bool()
        else:
            slices_true = batch[0].to(self.dtype)
            input_mask = torch.ones_like(slices_true).bool().bool()
            general_mask = torch.ones_like(slices_true).bool().bool()

        # Apply input and general masks and loss scale factor to get input slices.
        slices_input  = torch.log(input_mask * slices_true * self.model.loss_scale_factor + 1e-12)
        if self.model.fluctuation_predictor is not None:
            slices_target  = torch.log(general_mask * slices_true * self.model.loss_scale_factor + 1e-12)
        else:
            slices_target  = general_mask * slices_true * self.model.loss_scale_factor

        # predict orientations from images
        orientation_logits = self.model.orientation_predictor.compute_logits(slices_input)
        
        gt_orientation_probs = self.compute_probs_based_on_correlation(slices_target, self.ref_images, tau=self.get_dist_max(self.current_epoch))
        if batch_idx >= 0:
            self.log(f"{task_type}/probs_max", gt_orientation_probs.max().item(), sync_dist=True)
            self.log(f"{task_type}/probs_min", gt_orientation_probs.min().item(), sync_dist=True)
        gt_orientation_indices = torch.multinomial(gt_orientation_probs, 1).squeeze(-1)
        loss_orientation = torch.nn.functional.cross_entropy(orientation_logits, gt_orientation_probs)
        self.log(f"{task_type}/loss_orientation", loss_orientation.item(), sync_dist=True)

        orientation_probs = nn.functional.softmax(orientation_logits, dim=-1)
        orientation_onehot = nn.functional.one_hot(gt_orientation_indices, self.model.orientation_predictor.output_rotmats.size(0)).to(self.model.orientation_predictor.output_rotmats)
        orientation_probs = orientation_onehot.detach() + orientation_probs - orientation_probs.detach()
        # orientation_probs  = nn.functional.gumbel_softmax(orientation_logits, dim=-1, tau=self.gumbel_softmax_temperature(self.current_epoch), hard=True) # one-hot of shape (batch_size, num_rotations)
        orientations = torch.einsum('bn, nij -> bij', orientation_probs, self.model.orientation_predictor.output_rotmats)

        # start_time = time.time()
        # orientation_onehot = nn.functional.one_hot(gt_orientation_indices, self.model.orientation_predictor.output_rotmats.size(0)).to(self.model.orientation_predictor.output_rotmats)
        # print(f'one hot time: {time.time() - start_time}')
        # orientations = torch.einsum('bn, nij -> bij', orientation_onehot, self.model.orientation_predictor.output_rotmats)
        # print(f'einsum time: {time.time() - start_time}')

        # orientation_onehot = nn.functional.one_hot(gt_orientation_indices, self.model.orientation_predictor.output_rotmats.size(0)).float()
        # print(gt_orientation_indices.shape)
        # print(orientation_onehot.shape)
        # print(self.model.orientation_predictor.output_rotmats.shape)
        # orientations = torch.einsum('bn, nij -> bij', orientation_onehot, self.model.orientation_predictor.output_rotmats)

        orientations_shape = orientations.shape[:-2]
        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations.view(-1,3,3), self.model.pixel_position_reciprocal, self.model.over_sampling).T

        # predict slices from HKL
        _slices_pred = self.model.predict_slice(HKL).view(orientations_shape + (1, self.model.image_dimension,self.model.image_dimension)).clamp(np.log(1e-12), np.log(2500 * self.model.loss_scale_factor))
        if self.model.fluctuation_predictor is not None:
            slices_scale_factor = self.model.fluctuation_predictor(slices_input).unsqueeze(-1).unsqueeze(-1)
            # slices_target  = torch.log(general_mask * slices_true * self.model.loss_scale_factor + 1e-12)
            slices_pred    = torch.log(general_mask * torch.exp(_slices_pred) * slices_scale_factor + 1e-12)
        # else:
        #     slices_target  = general_mask * slices_true * self.model.loss_scale_factor
            
        loss_reconst = self.loss_func(slices_pred[general_mask.bool()].cpu(), 
                              slices_target[general_mask.bool()].cpu())
        self.log(f"{task_type}/loss_reconst", loss_reconst.item(), sync_dist=True)
        loss = loss_reconst + loss_orientation
        
        self.log(f"{task_type}/loss", loss.item(), prog_bar=True, sync_dist=True)

        # display_volumes(rho, save_to=f'{self.path}/rho.png')
        if self.global_step % 10 == 0:
            self.get_figure_save_dir()
            num_figs = min(10, slices_true.shape[0])
            if self.log_transform:
                slice_disp = torch.exp(_slices_pred[:num_figs]) / self.model.loss_scale_factor
            else:
                slice_disp = _slices_pred[:num_figs] / self.model.loss_scale_factor
            display_images_in_parallel(slice_disp, slices_true[:num_figs], save_to=f'{self.fig_path}/version_{self.logger.version}_{task_type}.png', closefig=True)
            display_images_in_parallel(_slices_pred[:num_figs], slices_input[:num_figs], save_to=f'{self.fig_path}/version_{self.logger.version}_{task_type}_raw.png', closefig=True)
            
            reciprocal_volume = self.predict_reciprocal_volume()
            try:
                display_volumes(reciprocal_volume, closefig=True, cmap='gray',
                                vmax=1e-3 * reciprocal_volume.max(),
                                save_to=f'{self.fig_path}/version_{self.logger.version}_{task_type}_reciprocal_vol.png')
                display_volumes(np.log(reciprocal_volume.clip(1e-12, None)) - np.log(1e-12), closefig=True, cmap='gray',
                                vmax=1e-3 * reciprocal_volume.max(),
                                save_to=f'{self.fig_path}/version_{self.logger.version}_{task_type}_reciprocal_vol_log.png')
                save_mrc(f'{self.fig_path}/version_{self.logger.version}_{task_type}_reciprocal_vol.mrc', reciprocal_volume)
                save_mrc(
                    f'{self.fig_path}/version_{self.logger.version}_{task_type}_reciprocal_vol_log.mrc', 
                    np.log(reciprocal_volume.clip(1e-12, None)) - np.log(1e-12)
                )
            except ValueError:
                pass

        return loss


    def validation_step(self, batch, batch_idx):
        task_type = 'val'
        
        if isinstance(batch, dict):
            slices_true = batch['image'].to(self.dtype)
            input_mask  = batch['input_mask'].bool()
            general_mask = batch['general_mask'].bool()
        else:
            slices_true = batch[0].to(self.dtype)
            input_mask = torch.ones_like(slices_true).bool().bool()
            general_mask = torch.ones_like(slices_true).bool().bool()

        # Apply input and general masks and loss scale factor to get input slices.
        slices_input  = torch.log(input_mask * slices_true * self.model.loss_scale_factor + 1e-12)
        if self.model.fluctuation_predictor is not None:
            slices_target  = torch.log(general_mask * slices_true * self.model.loss_scale_factor + 1e-12)
        else:
            slices_target  = general_mask * slices_true * self.model.loss_scale_factor

        # predict orientations from images
        orientation_logits = self.model.orientation_predictor.compute_logits(slices_input)
        
        if not hasattr(self, 'ref_images'):
            self.ref_images = torch.randn(self.model.orientation_predictor.output_rotmats.size(0), 1, self.model.image_dimension, self.model.image_dimension).to(self.device)

        gt_orientation_probs = self.compute_probs_based_on_correlation(slices_target, self.ref_images, tau=self.get_dist_max(self.current_epoch))
        if batch_idx >= 0:
            self.log(f"{task_type}/probs_max", gt_orientation_probs.max().item(), sync_dist=True)
            self.log(f"{task_type}/probs_min", gt_orientation_probs.min().item(), sync_dist=True)

        gt_orientation_indices = torch.multinomial(gt_orientation_probs, 1).squeeze(-1)
        loss_orientation = torch.nn.functional.cross_entropy(orientation_logits, gt_orientation_probs)
        self.log(f"{task_type}/loss_orientation", loss_orientation.item(), sync_dist=True)

        orientation_probs = nn.functional.softmax(orientation_logits, dim=-1)
        orientation_onehot = nn.functional.one_hot(gt_orientation_indices, self.model.orientation_predictor.output_rotmats.size(0)).to(self.model.orientation_predictor.output_rotmats)
        orientation_probs = orientation_onehot.detach() + orientation_probs - orientation_probs.detach()
        # orientation_probs  = nn.functional.gumbel_softmax(orientation_logits, dim=-1, tau=self.gumbel_softmax_temperature(self.current_epoch), hard=True) # one-hot of shape (batch_size, num_rotations)
        orientations = torch.einsum('bn, nij -> bij', orientation_probs, self.model.orientation_predictor.output_rotmats)

        # orientations = torch.index_select(self.model.orientation_predictor.output_rotmats, 0, gt_orientation_indices.long())
        # orientations = self.model.orientation_predictor.output_rotmats[gt_orientation_indices]

        orientations_shape = orientations.shape[:-2]
        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations.view(-1,3,3), self.model.pixel_position_reciprocal, self.model.over_sampling).T

        # predict slices from HKL
        _slices_pred = self.model.predict_slice(HKL).view(orientations_shape + (1, self.model.image_dimension,self.model.image_dimension)).clamp(np.log(1e-12), np.log(2500 * self.model.loss_scale_factor))
        if self.model.fluctuation_predictor is not None:
            slices_scale_factor = self.model.fluctuation_predictor(slices_input).unsqueeze(-1).unsqueeze(-1)
            # slices_target  = torch.log(general_mask * slices_true * self.model.loss_scale_factor + 1e-12)
            slices_pred    = torch.log(general_mask * torch.exp(_slices_pred) * slices_scale_factor + 1e-12)
        # else:
        #     slices_target  = general_mask * slices_true * self.model.loss_scale_factor
            
        loss_reconst = self.loss_func(slices_pred[general_mask.bool()].cpu(), 
                              slices_target[general_mask.bool()].cpu())
        self.log(f"{task_type}/loss_reconst", loss_reconst.item(), sync_dist=True)
        loss = loss_reconst + loss_orientation
        
        self.log(f"{task_type}/loss", loss.item(), prog_bar=True, sync_dist=True)

        # display_volumes(rho, save_to=f'{self.path}/rho.png')
        if batch_idx == 0:
            self.get_figure_save_dir()
            num_figs = min(10, slices_true.shape[0])
            if self.log_transform:
                slice_disp = torch.exp(_slices_pred[:num_figs]) / self.model.loss_scale_factor
            else:
                slice_disp = _slices_pred[:num_figs] / self.model.loss_scale_factor
            display_images_in_parallel(slice_disp, slices_true[:num_figs], save_to=f'{self.fig_path}/version_{self.logger.version}_{task_type}.png', closefig=True)
            display_images_in_parallel(_slices_pred[:num_figs], slices_input[:num_figs], save_to=f'{self.fig_path}/version_{self.logger.version}_{task_type}_raw.png', closefig=True)

           
    def _step(self, batch, batch_idx):
        
        task_type = 'step'
        
        if isinstance(batch, dict):
            slices_true = batch['image'].to(self.dtype)
            input_mask  = batch['input_mask'].bool()
            general_mask = batch['general_mask'].bool()
        else:
            slices_true = batch[0].to(self.dtype)
            input_mask = torch.ones_like(slices_true).bool().bool()
            general_mask = torch.ones_like(slices_true).bool().bool()

        # Apply input and general masks and loss scale factor to get input slices.
        slices_input  = torch.log(input_mask * slices_true * self.model.loss_scale_factor + 1e-12)

        # predict orientations from images
        orientation_logits = self.model.orientation_predictor.compute_logits(slices_input)
        orientation_probs = nn.functional.softmax(orientation_logits, dim=-1)
        # orientation_probs  = nn.functional.gumbel_softmax(orientation_logits, dim=-1, tau=self.gumbel_softmax_temperature(self.current_epoch), hard=False)
        num_samples = min(75 * slices_input.size(0), 70) // slices_input.size(0)
        orientation_indices = torch.multinomial(orientation_probs, num_samples, replacement=True)
        orientations = self.model.orientation_predictor.output_rotmats[orientation_indices]

        orientations_shape = orientations.shape[:-2]
        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations.view(-1,3,3), self.model.pixel_position_reciprocal, self.model.over_sampling).T

        # predict slices from HKL
        _slices_pred = self.model.predict_slice(HKL).view(orientations_shape + (1, self.model.image_dimension,self.model.image_dimension)).clamp(np.log(1e-12), np.log(2500 * self.model.loss_scale_factor))
        if self.model.fluctuation_predictor is not None:
            slices_scale_factor = self.model.fluctuation_predictor(slices_input).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            slices_target  = torch.log(general_mask * slices_true * self.model.loss_scale_factor + 1e-12)
            slices_pred    = torch.log(general_mask.unsqueeze(1).repeat_interleave(num_samples, dim=1) * torch.exp(_slices_pred) * slices_scale_factor + 1e-12)
        else:
            slices_target  = general_mask * slices_true * self.model.loss_scale_factor
        slices_target = slices_target.unsqueeze(1).repeat_interleave(num_samples, dim=1)
        # We don't want to compare the general masked area
        # loss = self.loss_func((slices_scale_factor * slices_pred)[general_mask.bool()].cpu(), slices_target[general_mask.bool()].cpu())
        loss = self.loss_func(slices_pred[general_mask.unsqueeze(1).repeat_interleave(num_samples, dim=1).bool()].cpu(), 
                              slices_target[general_mask.unsqueeze(1).repeat_interleave(num_samples, dim=1).bool()].cpu())
        
        return {
            'orientations': orientations,
            'slices_pred': slices_pred,
            'slices_target': slices_target,
            'slice_input': slices_input,
        } 

    # def _step(self, batch, batch_idx):
        
    #     task_type = 'step'
        
    #     if isinstance(batch, dict):
    #         slices_true = batch['image'].to(self.dtype)
    #         input_mask  = batch['input_mask'].bool()
    #         general_mask = batch['general_mask'].bool()
    #     else:
    #         slices_true = batch[0].to(self.dtype)
    #         input_mask = torch.ones_like(slices_true).bool().bool()
    #         general_mask = torch.ones_like(slices_true).bool().bool()

    #     # Apply input and general masks and loss scale factor to get input slices.
    #     slices_input  = torch.log(input_mask * slices_true * self.model.loss_scale_factor + 1e-12)

    #     # predict orientations from images
    #     orientation_logits = self.model.orientation_predictor.compute_logits(slices_input)
    #     orientation_probs = nn.functional.softmax(orientation_logits, dim=-1)
    #     # orientation_probs  = nn.functional.gumbel_softmax(orientation_logits, dim=-1, tau=self.gumbel_softmax_temperature(self.current_epoch), hard=False)
    #     num_samples = min(75 * slices_input.size(0), 70) // slices_input.size(0)
    #     orientation_indices = torch.multinomial(orientation_probs, num_samples, replacement=True)
    #     orientations = self.model.orientation_predictor.output_rotmats[orientation_indices]

    #     orientations_shape = orientations.shape[:-2]
    #     # get reciprocal positions based on orientations
    #     # HKL has shape (3, num_qpts)
    #     HKL = gen_nonuniform_normalized_positions(
    #         orientations.view(-1,3,3), self.model.pixel_position_reciprocal, self.model.over_sampling).T

    #     # predict slices from HKL
    #     _slices_pred = self.model.predict_slice(HKL).view(orientations_shape + (1, self.model.image_dimension,self.model.image_dimension)).clamp(np.log(1e-12), np.log(2500 * self.model.loss_scale_factor))
    #     if self.model.fluctuation_predictor is not None:
    #         slices_scale_factor = self.model.fluctuation_predictor(slices_input).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #         slices_target  = torch.log(general_mask * slices_true * self.model.loss_scale_factor + 1e-12)
    #         slices_pred    = torch.log(general_mask.unsqueeze(1).repeat_interleave(num_samples, dim=1) * torch.exp(_slices_pred) * slices_scale_factor + 1e-12)
    #     else:
    #         slices_target  = general_mask * slices_true * self.model.loss_scale_factor
    #     slices_target = slices_target.unsqueeze(1).repeat_interleave(num_samples, dim=1)
    #     # We don't want to compare the general masked area
    #     # loss = self.loss_func((slices_scale_factor * slices_pred)[general_mask.bool()].cpu(), slices_target[general_mask.bool()].cpu())
    #     loss = self.loss_func(slices_pred[general_mask.unsqueeze(1).repeat_interleave(num_samples, dim=1).bool()].cpu(), 
    #                           slices_target[general_mask.unsqueeze(1).repeat_interleave(num_samples, dim=1).bool()].cpu())
        
    #     return {
    #         'orientations': orientations,
    #         'slices_pred': slices_pred,
    #         'slices_target': slices_target,
    #         'slice_input': slices_input,
    #     }
            
    def configure_optimizers(self):
        if not 'scheduler' in self.configure_optimization:
            optimizer = optim.AdamW(self.parameters(), lr=self.configure_optimization['lr'], weight_decay=self.configure_optimization['weight_decay'])
            return optimizer
        else:
            _configure_optimization = deepcopy(self.configure_optimization)
            optimizer = optim.AdamW(self.parameters(), lr=_configure_optimization['lr'], weight_decay=_configure_optimization['weight_decay'])
            scheduler = scheduler_dict[self.configure_optimization['scheduler'].pop('name')](
                optimizer     = optimizer, 
                warmup_epochs = _configure_optimization['scheduler']['warmup_epochs'],
                total_epochs  = _configure_optimization['scheduler']['total_epochs'],
                min_lr        = _configure_optimization['scheduler']['min_lr'])
            return [optimizer,], [scheduler,]
    
    def get_figure_save_dir(self,):
        if not hasattr(self, 'fig_path'):
            self.fig_path = Path(
                os.path.join(self.trainer.logger.log_dir, 'figures')
            )
            self.fig_path.mkdir(parents=True, exist_ok=True)
            
    def predict_reciprocal_volume(self, zoom=1.0):
        grid_reciprocal = np.pi * self.model.grid_position_reciprocal / self.model.grid_position_reciprocal.max()
        if zoom != 1.0:
            grid_reciprocal = scipy.ndimage.zoom(grid_reciprocal.detach().cpu().numpy(), (zoom,zoom,zoom,1), order=1)
            grid_reciprocal = torch.from_numpy(grid_reciprocal).to(self.device)
        volume = np.zeros(grid_reciprocal.shape[:3])
        with torch.no_grad():
            for i in range(grid_reciprocal.shape[0]):
                input_coords = grid_reciprocal[i,None,...].to(self.device)
                volume[i] = self.model.predict_intensity(input_coords).detach().cpu().clamp(np.log(1e-12), np.log(2500 * self.model.loss_scale_factor)).numpy().squeeze()
        
        volume = np.exp(volume) / self.model.loss_scale_factor
        return volume.clip(0.0)
    
    def get_symmetrized_coordinates(self, grid, symm_ops):
        """
        Get symmetrized coordinates for a given grid
        params:
            grid: torch.Tensor of shape (..., 3)
            symm_ops: torch.Tensor of shape (N, 3, 3)
        """
        symm_ops = symm_ops.to(self.device).to(self.dtype)
        grid = grid.to(self.device).to(self.dtype)
        
        grid_shape = grid.shape[:-1]
        grid_flat = grid.view(-1, 3)
        
        rot_grid = torch.einsum('sij, bi -> sbj', symm_ops, grid_flat).view(-1, *grid_shape, 3)
        
        # rot_grid_test = torch.einsum('sij, bj -> sbi', symm_ops.transpose(1,2), grid_flat).view(-1, *grid_shape, 3)
        # print(torch.allclose(rot_grid, rot_grid_test))
        
        return rot_grid

    def get_zoomed_reciprocal_grid(self, zoom=1.0):
        grid_reciprocal = np.pi * self.model.grid_position_reciprocal / self.model.grid_position_reciprocal.max()
        if zoom != 1.0:
            grid_reciprocal = scipy.ndimage.zoom(grid_reciprocal.detach().cpu().numpy(), (zoom,zoom,zoom,1), order=1)
            grid_reciprocal = torch.from_numpy(grid_reciprocal).to(self.device)
        return grid_reciprocal
    
    
    def predict_symmetrized_reciprocal_volume(self, symm_ops, zoom=1.0):
        grid_reciprocal = self.get_zoomed_reciprocal_grid(zoom=zoom)
        
        volume = torch.zeros((symm_ops.shape[0], *grid_reciprocal.shape[:3])).to(self.device).to(self.dtype)
        symm_grid_reciprocal = self.get_symmetrized_coordinates(grid_reciprocal, symm_ops)
        for i in range(symm_ops.shape[0]):
            for j in range(grid_reciprocal.shape[0]):
                input_coords = symm_grid_reciprocal[i,j,None,...].to(self.device)
                volume[i,j] = self.model.predict_intensity(input_coords)
        
        volume = torch.exp(volume) / self.model.loss_scale_factor
        return volume
    
    def _predict_symmetrized_reciprocal_volume(self, symm_grid_reciprocal):
        
        volume = torch.zeros(symm_grid_reciprocal.shape[:-1]).to(self.device).to(self.dtype)
        # for i in range(symm_grid_reciprocal.shape[0]):
        #     for j in range(symm_grid_reciprocal.shape[1]):
        #         input_coords = symm_grid_reciprocal[i,j,None,...].to(self.device)
        #         volume[i,j] = self.model.predict_intensity(input_coords)
        for i in range(symm_grid_reciprocal.shape[0]):
            input_coords = symm_grid_reciprocal[i,None,...].to(self.device)
            _volume = self.model.predict_intensity(input_coords)
            volume[i] = _volume
        
        volume = torch.exp(volume) / self.model.loss_scale_factor
        
        return volume