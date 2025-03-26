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

from .encoder_i2s import I2S

INTENSITY_MIN = 1e-8
DIVISOR_EPS = 1e-8

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
        return {'rotations': rotmat}
    
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

class IntensityNet_GSInv(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.sym_net = SymmetrizedFeature('I')
        self.net_mag = SirenNet(*args, **kwargs)

    def forward(self, x):
        x_symm = self.sym_net(x)
        # x_symm = x
        return self.net_mag(x_symm)

class NeurOrient(nn.Module):
    def __init__(self, 
                 pixel_position_reciprocal, 
                 over_sampling=1,
                 photons_per_pulse=1e13,
                 use_bifpn=False,
                 rec_level=3,
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
            
        self.orientation_predictor = Slice2RotMat(**config_slice2rotmat)
        if use_fluctuation_predictor:
            self.fluctuation_predictor = FluctuationPredictor(config_slice2rotmat['size'], config_slice2rotmat['pretrained'])
        else:
            self.fluctuation_predictor = None

        # setup volume predictor
        self.volume_predictor = IntensityNet_GSInv(
            dim_in=3,
            dim_hidden=config_intensitynet['dim_hidden'],
            dim_out=1,
            num_layers=config_intensitynet['num_layers'],
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
        slices_input = torch.log(batch['input_mask'] * batch['image'] * self.loss_scale_factor + INTENSITY_MIN)

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
            # self.model.volume_predictor = torch.nn.Sequential(
            #     self.model.volume_predictor,
            #     torch.nn.Softplus()
            # )

    def prepare_input_slices(self, batch):
        
        if isinstance(batch, dict):
            slices_true = batch['image'].to(self.dtype)
            input_mask  = batch['input_mask'].bool()
            general_mask = batch['general_mask'].bool()
        else:
            slices_true = batch[0].to(self.dtype)
            input_mask = torch.ones_like(slices_true).bool().bool()
            general_mask = torch.ones_like(slices_true).bool().bool()

        # Apply input and general masks and loss scale factor to get input slices.
        slices_input  = torch.log(input_mask * slices_true * self.model.loss_scale_factor + INTENSITY_MIN)

        return slices_input
    
    def predict_slice(self, orientations):
        if orientations.ndim == 2:
            orientations = orientations.unsqueeze(0)
        HKL = gen_nonuniform_normalized_positions(
            orientations, self.model.pixel_position_reciprocal, self.model.over_sampling)
        slices_pred = self.model.predict_slice(HKL).view((-1, 1,) + (self.model.image_dimension,)*2)
        return slices_pred

    def estimate_batch(self, batch):
        
        if isinstance(batch, dict):
            slices_true = batch['image'].to(self.dtype)
            input_mask  = batch['input_mask'].bool()
            general_mask = batch['general_mask'].bool()
        else:
            slices_true = batch[0].to(self.dtype)
            input_mask = torch.ones_like(slices_true).bool().bool()
            general_mask = torch.ones_like(slices_true).bool().bool()

        # Apply input and general masks and loss scale factor to get input slices.
        slices_input  = torch.log(input_mask * slices_true * self.model.loss_scale_factor + INTENSITY_MIN)

        # predict orientations from images
        orientations_out = self.model.orientation_predictor(slices_input)
        orientations = orientations_out['rotations']

        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations, self.model.pixel_position_reciprocal, self.model.over_sampling)
        if HKL.ndim == 2 and HKL.shape[0] == 3:
            HKL = HKL.T
        # predict slices from HKL
        # if self.log_transform:
        #     _slices_pred_symm, _slices_pred_asymm = self.model.volume_predictor.forward_with_separated_outputs(HKL)
        #     loss_intens_adjust = _slices_pred_asymm.abs().mean()
        #     loss_additional += loss_intens_adjust
        #     _slices_pred = (_slices_pred_symm + _slices_pred_asymm).view((-1, 1,) + (self.model.image_dimension,)*2).clamp(np.log(INTENSITY_MIN), np.log(2500 * self.model.loss_scale_factor))
        # else:
        #     _slices_pred_symm, _slices_pred_asymm = self.model.volume_predictor.forward_with_separated_outputs(HKL)
        #     loss_intens_adjust = _slices_pred_asymm.abs().mean()
        #     loss_additional += loss_intens_adjust
        #     _slices_pred = (_slices_pred_symm + _slices_pred_asymm).view((-1, 1,) + (self.model.image_dimension,)*2)

        if self.log_transform:
            _slices_pred = self.model.volume_predictor(HKL).view((-1, 1,) + (self.model.image_dimension,)*2).clamp(np.log(INTENSITY_MIN), np.log(2500 * self.model.loss_scale_factor))
        else:
            _slices_pred = self.model.volume_predictor(HKL).view((-1, 1,) + (self.model.image_dimension,)*2)

        if self.model.fluctuation_predictor is not None:
            slices_scale_factor = self.model.fluctuation_predictor(slices_input).unsqueeze(-1).unsqueeze(-1)
        else:
            slices_scale_factor = 1.0
        # slices_scale_factor = 1.0
        
        if self.log_transform:
            slices_target  = torch.log(general_mask * slices_true * self.model.loss_scale_factor + INTENSITY_MIN)
            slices_pred    = torch.log(general_mask * torch.exp(_slices_pred) * slices_scale_factor + INTENSITY_MIN)
        else:
            slices_target  = general_mask * slices_true
            slices_pred    = general_mask * _slices_pred * slices_scale_factor
        
        output = {
            'slices_pred': slices_pred,
            'slices_target': slices_target,
            'slices_input': slices_input,
            'orientations': orientations,
            'orientations_out': orientations_out,
            'slice_scale_factor': slices_scale_factor,
        }
        return output
        
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
        slices_input  = torch.log(input_mask * slices_true * self.model.loss_scale_factor + INTENSITY_MIN)

        orientations_out = self.model.orientation_predictor(slices_input)

        orientations = orientations_out['rotations']

        loss_additional = 0.0
        for k, v in orientations_out.items():
            if 'min' in k or 'max' in k:
                self.log(f"{task_type}/{k}", v.item(), sync_dist=True)
            elif 'loss' in k:
                self.log(f"{task_type}/{k}", v.item(), sync_dist=True)
                loss_additional += v

        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations, self.model.pixel_position_reciprocal, self.model.over_sampling)
        if HKL.ndim == 2 and HKL.shape[0] == 3:
            HKL = HKL.T
        # predict slices from HKL
        # if self.log_transform:
        #     _slices_pred_symm, _slices_pred_asymm = self.model.volume_predictor.forward_with_separated_outputs(HKL)
        #     loss_intens_adjust = _slices_pred_asymm.abs().mean()
        #     loss_additional += loss_intens_adjust
        #     _slices_pred = (_slices_pred_symm + _slices_pred_asymm).view((-1, 1,) + (self.model.image_dimension,)*2).clamp(np.log(INTENSITY_MIN), np.log(2500 * self.model.loss_scale_factor))
        # else:
        #     _slices_pred_symm, _slices_pred_asymm = self.model.volume_predictor.forward_with_separated_outputs(HKL)
        #     loss_intens_adjust = _slices_pred_asymm.abs().mean()
        #     loss_additional += loss_intens_adjust
        #     _slices_pred = (_slices_pred_symm + _slices_pred_asymm).view((-1, 1,) + (self.model.image_dimension,)*2)
        # self.log(f"{task_type}/loss_intens_adjust", loss_intens_adjust.item(), sync_dist=True)

        if self.log_transform:
            _slices_pred = self.model.volume_predictor(HKL).view((-1, 1,) + (self.model.image_dimension,)*2).clamp(np.log(INTENSITY_MIN), np.log(2500 * self.model.loss_scale_factor))
        else:
            _slices_pred = self.model.volume_predictor(HKL).view((-1, 1,) + (self.model.image_dimension,)*2)
        
        if self.model.fluctuation_predictor is not None:
            slices_scale_factor = self.model.fluctuation_predictor(slices_input).unsqueeze(-1).unsqueeze(-1)
        else:
            slices_scale_factor = 1.0
        
        if self.log_transform:
            slices_target  = torch.log(general_mask * slices_true * self.model.loss_scale_factor + INTENSITY_MIN)
            slices_pred    = torch.log(general_mask * torch.exp(_slices_pred) * slices_scale_factor + INTENSITY_MIN)
        else:
            slices_target  = general_mask * slices_true
            slices_pred    = general_mask * _slices_pred * slices_scale_factor

        loss_reco = self.loss_func(slices_pred[general_mask.bool()].cpu(), slices_target[general_mask.bool()].cpu())
        
        loss = loss_reco + loss_additional
        self.log(f"{task_type}/loss_reco", loss_reco.item(), sync_dist=True)
        self.log(f"{task_type}/loss", loss.item(), prog_bar=True, sync_dist=True)

        # display_volumes(rho, save_to=f'{self.path}/rho.png')
        if self.global_step % 10 == 0:
            self.get_figure_save_dir()
            num_figs = min(10, slices_true.shape[0])
            if self.log_transform:
                slice_disp = torch.exp(_slices_pred) / self.model.loss_scale_factor
            else:
                slice_disp = torch.log(input_mask * _slices_pred * self.model.loss_scale_factor + INTENSITY_MIN)
            display_images_in_parallel(slice_disp[:num_figs], slices_true[:num_figs], save_to=f'{self.fig_path}/version_{self.logger.version}_{task_type}.png', closefig=True)
            display_images_in_parallel(_slices_pred[:num_figs], slices_input[:num_figs], save_to=f'{self.fig_path}/version_{self.logger.version}_{task_type}_raw.png', closefig=True)
            
            reciprocal_volume = self.predict_reciprocal_volume()
            try:
                display_volumes(reciprocal_volume, closefig=True, cmap='gray',
                                vmax=1e-3 * reciprocal_volume.max(),
                                save_to=f'{self.fig_path}/version_{self.logger.version}_{task_type}_reciprocal_vol.png')
                display_volumes(np.log(reciprocal_volume.clip(INTENSITY_MIN, None)) - np.log(INTENSITY_MIN), closefig=True, cmap='gray',
                                vmax=1e-3 * reciprocal_volume.max(),
                                save_to=f'{self.fig_path}/version_{self.logger.version}_{task_type}_reciprocal_vol_log.png')
                save_mrc(f'{self.fig_path}/version_{self.logger.version}_{task_type}_reciprocal_vol.mrc', reciprocal_volume)
                save_mrc(
                    f'{self.fig_path}/version_{self.logger.version}_{task_type}_reciprocal_vol_log.mrc', 
                    np.log(reciprocal_volume.clip(INTENSITY_MIN, None)) - np.log(INTENSITY_MIN)
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
        slices_input  = torch.log(input_mask * slices_true * self.model.loss_scale_factor + INTENSITY_MIN)

        orientations_out = self.model.orientation_predictor(slices_input)

        orientations = orientations_out['rotations']

        loss_additional = 0.0
        for k, v in orientations_out.items():
            if 'min' in k or 'max' in k:
                self.log(f"{task_type}/{k}", v.item(), sync_dist=True)
            elif 'loss' in k:
                self.log(f"{task_type}/{k}", v.item(), sync_dist=True)
                loss_additional += v

        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations, self.model.pixel_position_reciprocal, self.model.over_sampling)
        if HKL.ndim == 2 and HKL.shape[0] == 3:
            HKL = HKL.T
        # predict slices from HKL
        # if self.log_transform:
        #     _slices_pred_symm, _slices_pred_asymm = self.model.volume_predictor.forward_with_separated_outputs(HKL)
        #     loss_intens_adjust = _slices_pred_asymm.abs().mean()
        #     loss_additional += loss_intens_adjust
        #     _slices_pred = (_slices_pred_symm + _slices_pred_asymm).view((-1, 1,) + (self.model.image_dimension,)*2).clamp(np.log(INTENSITY_MIN), np.log(2500 * self.model.loss_scale_factor))
        # else:
        #     _slices_pred_symm, _slices_pred_asymm = self.model.volume_predictor.forward_with_separated_outputs(HKL)
        #     loss_intens_adjust = _slices_pred_asymm.abs().mean()
        #     loss_additional += loss_intens_adjust
        #     _slices_pred = (_slices_pred_symm + _slices_pred_asymm).view((-1, 1,) + (self.model.image_dimension,)*2)
        # self.log(f"{task_type}/loss_intens_adjust", loss_intens_adjust.item(), sync_dist=True)

        if self.log_transform:
            _slices_pred = self.model.volume_predictor(HKL).view((-1, 1,) + (self.model.image_dimension,)*2).clamp(np.log(INTENSITY_MIN), np.log(2500 * self.model.loss_scale_factor))
        else:
            _slices_pred = self.model.volume_predictor(HKL).view((-1, 1,) + (self.model.image_dimension,)*2)
        
        if self.model.fluctuation_predictor is not None:
            slices_scale_factor = self.model.fluctuation_predictor(slices_input).unsqueeze(-1).unsqueeze(-1)
        else:
            slices_scale_factor = 1.0
        
        if self.log_transform:
            slices_target  = torch.log(general_mask * slices_true * self.model.loss_scale_factor + INTENSITY_MIN)
            slices_pred    = torch.log(general_mask * torch.exp(_slices_pred) * slices_scale_factor + INTENSITY_MIN)
        else:
            slices_target  = general_mask * slices_true
            slices_pred    = general_mask * _slices_pred * slices_scale_factor

        loss_reco = self.loss_func(slices_pred[general_mask.bool()].cpu(), slices_target[general_mask.bool()].cpu())
        
        loss = loss_reco + loss_additional
        self.log(f"{task_type}/loss_reco", loss_reco.item(), sync_dist=True)
        self.log(f"{task_type}/loss", loss.item(), prog_bar=True, sync_dist=True)

        # display_volumes(rho, save_to=f'{self.path}/rho.png')
        if self.global_step % 10 == 0:
            self.get_figure_save_dir()
            num_figs = min(10, slices_true.shape[0])
            if self.log_transform:
                slice_disp = torch.exp(_slices_pred) / self.model.loss_scale_factor
            else:
                slice_disp = torch.log(input_mask * _slices_pred * self.model.loss_scale_factor + INTENSITY_MIN)
            display_images_in_parallel(slice_disp[:num_figs], slices_true[:num_figs], save_to=f'{self.fig_path}/version_{self.logger.version}_{task_type}.png', closefig=True)
            display_images_in_parallel(_slices_pred[:num_figs], slices_input[:num_figs], save_to=f'{self.fig_path}/version_{self.logger.version}_{task_type}_raw.png', closefig=True)
        
        if self.current_epoch % 10 == 1 and batch_idx == 0:
            
            reciprocal_volume = self.predict_reciprocal_volume()
            try:
                display_volumes(reciprocal_volume, closefig=True, cmap='gray',
                                vmax=1e-3 * reciprocal_volume.max(),
                                save_to=f'{self.fig_path}/version_{self.logger.version}_{task_type}_reciprocal_vol.png')
                display_volumes(np.log(reciprocal_volume.clip(INTENSITY_MIN, None)) - np.log(INTENSITY_MIN), closefig=True, cmap='gray',
                                vmax=1e-3 * reciprocal_volume.max(),
                                save_to=f'{self.fig_path}/version_{self.logger.version}_{task_type}_reciprocal_vol_log.png')
                # save_mrc(f'{self.fig_path}/version_{self.logger.version}_{task_type}_reciprocal_vol.mrc', reciprocal_volume)
                # save_mrc(
                #     f'{self.fig_path}/version_{self.logger.version}_{task_type}_reciprocal_vol_log.mrc', 
                #     np.log(reciprocal_volume.clip(INTENSITY_MIN, None)) - np.log(INTENSITY_MIN)
                # )
                # torch.save({'volume': reciprocal_volume, 
                #             'volume_log': np.log(reciprocal_volume.clip(INTENSITY_MIN, None)) - np.log(INTENSITY_MIN)
                #             }, 
                #            f'{self.fig_path}/version_{self.logger.version}_{task_type}_reciprocal_vol_epoch{self.current_epoch}.pt')
            except ValueError:
                pass
                
            
            
    def configure_optimizers(self):
        if not 'scheduler' in self.configure_optimization:
            optimizer = optim.AdamW(self.parameters(), lr=self.configure_optimization['lr'], weight_decay=self.configure_optimization['weight_decay'])
            return optimizer
        else:
            _configure_optimization = deepcopy(self.configure_optimization)
            optimizer = optim.AdamW(self.parameters(), lr=_configure_optimization['lr'], weight_decay=_configure_optimization['weight_decay'])
            scheduler = scheduler_dict[self.configure_optimization['scheduler']['name']](
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
                volume[i] = self.model.predict_intensity(input_coords).detach().cpu().clamp(np.log(INTENSITY_MIN), np.log(2500 * self.model.loss_scale_factor)).numpy().squeeze()
        
        if self.log_transform:
            volume = np.exp(volume) / self.model.loss_scale_factor
        
        return volume.clip(0.0)
    
    
            
    # def predict_detailed_reciprocal_volume(self, zoom=1.0):
    #     grid_reciprocal = np.pi * self.model.grid_position_reciprocal / self.model.grid_position_reciprocal.max()
    #     if zoom != 1.0:
    #         grid_reciprocal = scipy.ndimage.zoom(grid_reciprocal.detach().cpu().numpy(), (zoom,zoom,zoom,1), order=1)
    #         grid_reciprocal = torch.from_numpy(grid_reciprocal).to(self.device)
    #     volume_symm = np.zeros(grid_reciprocal.shape[:3])
    #     volume_nonsymm = np.zeros(grid_reciprocal.shape[:3])
    #     with torch.no_grad():
    #         for i in range(grid_reciprocal.shape[0]):
    #             input_coords = grid_reciprocal[i,None,...].to(self.device)
                
    #             if input_coords.ndim > 2 and input_coords.shape[-1] == 3:
    #                 out_shape = input_coords.shape[:-1]
    #                 input_coords = input_coords.view(-1, 3)
    #                 intensity_symm, intensity_nonsymm = self.model.volume_predictor.forward_with_separated_outputs(input_coords)
    #                 intensity_symm = intensity_symm.view(out_shape)
    #                 intensity_nonsymm = intensity_nonsymm.view(out_shape)
    #             else:
    #                 intensity_symm, intensity_nonsymm = self.model.volume_predictor.forward_with_separated_outputs(input_coords)

    #             volume_symm[i] = intensity_symm.detach().cpu().clamp(np.log(INTENSITY_MIN), np.log(2500 * self.model.loss_scale_factor)).numpy().squeeze()
    #             volume_nonsymm[i] = intensity_nonsymm.detach().cpu().clamp(np.log(INTENSITY_MIN), np.log(2500 * self.model.loss_scale_factor)).numpy().squeeze()
    #     # if self.log_transform:
    #     #     volume_symm = np.exp(volume_symm) / self.model.loss_scale_factor
    #         # volume_nonsymm = np.exp(volume_nonsymm) / self.model.loss_scale_factor
        
    #     return volume_symm, volume_nonsymm