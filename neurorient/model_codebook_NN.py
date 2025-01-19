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
from pytorch3d.transforms import rotation_6d_to_matrix, euler_angles_to_matrix, random_rotations, \
    matrix_to_rotation_6d, matrix_to_axis_angle, axis_angle_to_matrix

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


from pytorch3d.transforms import matrix_to_axis_angle
from .so3_decomposition import threshold_dict

def clamp_with_linear_slope(x, min_val, max_val, slope=0.01):
    """
    function written by ChatGPT o1 on Jan 4, 2025
    Piecewise linear clamp:
     - Identity in [min_val, max_val].
     - Linear with slope outside the boundary.
     - Ensures continuity at boundaries, non-zero gradient outside.
    """
    # Below min_val
    below_mask = (x < min_val)
    # Above max_val
    above_mask = (x > max_val)
    # Within range
    within_mask = (~below_mask) & (~above_mask)
    
    y = torch.empty_like(x)
    
    # Within [min_val, max_val]: identity
    y[within_mask] = x[within_mask]
    
    # Below min_val: y = min_val + slope*(x - min_val)
    x_below = x[below_mask]
    y[below_mask] = min_val + slope * (x_below - min_val)
    
    # Above max_val: y = max_val + slope*(x - max_val)
    x_above = x[above_mask]
    y[above_mask] = max_val + slope * (x_above - max_val)
    
    return y


class RotationClamp(nn.Module):
    def __init__(self, rec_level=None, max_rel_angle=None):
        super().__init__()
        if rec_level is not None:
            max_half_angle = threshold_dict[str(rec_level)] / 2
        elif max_rel_angle is not None:
            max_half_angle = max_rel_angle / 2
        
        # angle in degrees
        self.register_buffer('max_half_angle', torch.rad2deg(torch.tensor(max_half_angle)))
        
    def forward(self, R):
        """ R of shape (B, 3, 3)
        """
        if R.size(-1) == 6:
            R = rotation_6d_to_matrix(R)
        axis_angle = matrix_to_axis_angle(R)
        axis = axis_angle / (axis_angle.norm(dim=-1, keepdim=True) + 1e-8)
        angle = axis_angle.norm(dim=-1, keepdim=True)
        angle = torch.deg2rad(clamp_with_linear_slope(torch.rad2deg(angle), -self.max_half_angle, self.max_half_angle))
        new_axis_angle = angle * axis
        new_R = axis_angle_to_matrix(new_axis_angle)
        return new_R

class PerturbRotation6D(nn.Module):
    def __init__(self, base_matrix=None):
        super().__init__()
        if base_matrix is None:
            self.register_buffer('base_rotation_6d', matrix_to_rotation_6d(torch.eye(3)))
        else:
            self.register_buffer('base_rotation_6d', matrix_to_rotation_6d(base_matrix))
    
    def forward(self, rotation_6d):
        rotation_6d = self.base_rotation_6d + rotation_6d
        return rotation_6d

    
class Slice2RotMat_Refine(nn.Module):
    def __init__(self, size=18, pretrained=False, rec_level=2, hidden_dim=64, image_dimension=128):
        super().__init__()
        self.hidden_dim = hidden_dim    

        self.get_ref_rotations(rec_level)

        weights = 'DEFAULT' if pretrained else None
        self.resnet = eval(f'resnet.resnet{size}')(weights=weights)

        # Average the weights in the input channels...
        conv1_weight = self.resnet.conv1.weight.data.mean(dim = 1, keepdim = True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.conv1.weight.data = conv1_weight
        # Output 6D rotation matrix
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.hidden_dim)

        self._max_val = 1000.0
        
        print(f'Initialized with {self.n_ref} reference rotations and max_val: {self.max_val}')
        
        self.get_dummy_ref_images(image_dimension=image_dimension)

        self.orientation_embed = torch.nn.Sequential(
            torch.nn.Linear(6, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.adjuster = torch.nn.Sequential(
            torch.nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * self.hidden_dim, 6),
            PerturbRotation6D(),
            RotationClamp(rec_level=rec_level),
        )

        for p in self.adjuster.parameters():
            p.data = 1e-4 * torch.randn_like(p.data)

    def get_ref_rotations(self, rec_level=2):
        euler_yxy = so3_healpix_grid(rec_level).T
        grid_rotations = euler_angles_to_matrix(euler_yxy, convention='YXY')
        rand_rotations = random_rotations(grid_rotations.shape[0] // 4)
        self.register_buffer('ref_rotations', torch.cat([grid_rotations, rand_rotations], dim=0))
        self.n_ref = self.ref_rotations.shape[0]
    
    @property
    def max_val(self):
        return self._max_val
    
    @max_val.setter
    def max_val(self, value):
        self._max_val = value
    
    def get_dummy_ref_images(self, intens_func=None, pixel_position_reciprocal=None, image_dimension=None, over_sampling=1):
        if image_dimension is None:
            image_dimension = 128
        ref_images = torch.rand(self.n_ref, image_dimension, image_dimension)
        self.register_buffer('ref_images', ref_images)
        print(f'\nReference images generated, sampling at max_val: {self.max_val}')
        
    def get_ref_images(self, intens_func, pixel_position_reciprocal, image_dimension, over_sampling=1):
        with torch.no_grad():
            # print(HKL.shape)
            rotation_dset = TensorDataset(self.ref_rotations)
            rotation_dloader = DataLoader(rotation_dset, batch_size=50, shuffle=False)
            _ref_images = []
            for batch in tqdm(rotation_dloader, miniters=int(len(rotation_dloader)/10)):
                _rotations = batch[0].to(self.ref_rotations.device)
                HKL = gen_nonuniform_normalized_positions(
                    _rotations, pixel_position_reciprocal, over_sampling).T
                _intens = intens_func(HKL)
                _ref_images.append(_intens.view((-1, ) + (image_dimension,)*2))
            ref_images = torch.cat(_ref_images, dim=0)
        self.register_buffer('ref_images', ref_images)
        print(f'\nReference images generated, sampling at max_val: {self.max_val}')
        
    # def normalize_to_range(self, x, min_val=0.0, max_val=2*np.pi):
    #     return (x - x.min()) / (x.max() - x.min() + 1e-8) * (max_val - min_val) + min_val
        
    def normalize_to_range(self, x, min_val=0.0, max_val=2*np.pi):
        return (x - x.amin(dim=-1, keepdim=True)) / (x.amax(dim=-1, keepdim=True) - x.amin(dim=-1, keepdim=True) + 1e-8) * (max_val - min_val) + min_val
    
    def distance_func_L2(self, image):
        if image.ndim == 4:
            image = image.squeeze(1)
        dist = (image[:,None] - self.ref_images[None]).pow(2).mean(dim=(-2,-1))
        return dist
        
    def distance_func_PC(self, image):
        image_flat = image.view(image.shape[0], -1)
        ref_flat = self.ref_images.view(self.ref_images.shape[0], -1)
        
        image_flat = image_flat - image_flat.mean(dim=-1, keepdim=True)
        ref_flat = ref_flat - ref_flat.mean(dim=-1, keepdim=True)
        
        numerator = torch.einsum('bi, ni -> bn', image_flat, ref_flat)
        divisor = torch.einsum('b, n -> bn', image_flat.norm(dim=-1).pow(2), ref_flat.norm(dim=-1).pow(2)).sqrt()
        
        return 1 - numerator / (divisor + 1e-8)
    
    def forward(self, image):
        if image.ndim == 3:
            image = image.unsqueeze(1)
        dist = self.distance_func_PC(image.squeeze(1))
        if hasattr(self, 'max_val'):
            max_val = self.max_val
        else:
            max_val = 1000.0
        dist_norm = self.normalize_to_range(dist, min_val=0.0, max_val=max_val)
        probs = torch.softmax(-dist_norm, dim=-1)
        rotations_samp = self.ref_rotations[torch.multinomial(probs, 1).squeeze(1)]
        
        image_feat = self.resnet(image)
        orientation_feat = self.orientation_embed(matrix_to_rotation_6d(rotations_samp))
        feat = torch.cat([image_feat, orientation_feat], dim=-1)
        rotations_tune = self.adjuster(feat)

        rotations = torch.bmm(rotations_tune, rotations_samp)

        tune_angle = torch.rad2deg(matrix_to_axis_angle(rotations_tune).norm(dim=-1).mean())

        # output = rotations
        output = {
            'rotations': rotations,
            'rotations_samp': rotations_samp,
            'rotations_tune': rotations_tune,
            'tune_angle': tune_angle,
            'dist_norm': dist_norm,
            'dist': dist,
        }

        return output
    
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
        real_mesh, reciprocal_mesh = get_real_mesh(self.image_dimension, 
                                                   self.pixel_position_reciprocal.max(), return_reciprocal=True)
        self.register_buffer('grid_position_reciprocal', reciprocal_mesh)
        self.register_buffer('grid_position_real', real_mesh)
        del real_mesh, reciprocal_mesh

        self.over_sampling = over_sampling
            
        # self.orientation_predictor = Slice2RotMat(**config_slice2rotmat)
        self.orientation_predictor = Slice2RotMat_Refine(rec_level=rec_level, image_dimension=self.image_dimension)
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
        slices_input = torch.log(batch['input_mask'] * batch['image'] * self.loss_scale_factor + 1e-8)

        # predict orientations from images
        orientations = self.image_to_orientation(slices_input)['rotations']
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
        
        symm_ops = so3_point_group_operations('I')
        self.register_buffer('symm_ops', symm_ops)
        self.training_zoom_level = 1/3
        self.register_buffer('training_reciprocal_grid', self.get_zoomed_reciprocal_grid(zoom=self.training_zoom_level))
        self.register_buffer('training_symm_reciprocal_grid', self.get_symmetrized_coordinates(self.training_reciprocal_grid, symm_ops))
        # self.training_reciprocal_grid = self.get_zoomed_reciprocal_grid(zoom=self.training_zoom_level)
        # self.training_symm_reciprocal_grid = self.get_symmetrized_coordinates(self.training_reciprocal_grid, symm_ops)
        
        # self.lr = config_optimization['lr']
        # self.weight_decay = config_optimization['weight_decay']
        # self.loss_func = eval(f"torch.nn.{config_optimization['loss_func']}()")
        
        # for key, value in config_optimization.items():
        #     self.__setattr__(key, value)
        
        self.configure_optimization = config_optimization
        if config_optimization['loss_func'] != 'PoissonNLLLoss':
            self.loss_func = eval(f"torch.nn.{config_optimization['loss_func']}()")
            self.log_transform = True
        else:
            self.loss_func = torch.nn.PoissonNLLLoss(log_input=False, full=True)
            self.log_transform = False
            
    def on_train_epoch_start(self, *args, **kwargs):
        # self.model.orientation_predictor.max_val = 100 - 99 * np.exp(- 0.1 * self.current_epoch)
        if self.current_epoch < 100:
            self.model.orientation_predictor.max_val = 1000 - 999 * np.exp(- 0.001 * self.current_epoch)
        else:
            self.model.orientation_predictor.max_val = 1000
            
        self.model.orientation_predictor.get_ref_images(
            self.model.volume_predictor, self.model.pixel_position_reciprocal, self.model.image_dimension, self.model.over_sampling)
        
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
        slices_input  = torch.log(input_mask * slices_true * self.model.loss_scale_factor + 1e-8)

        # predict orientations from images
        # orientations_out = self.model.image_to_orientation(slices_input)
        _orientations = self.model.image_to_orientation(slices_input)
        if isinstance(_orientations, dict):
            if 'rotations' in _orientations:
                orientations = _orientations['rotations']
            else:
                if self.current_epoch < 100:
                    orientations = _orientations['rotations_samp']
                else:
                    orientations = _orientations['rotations_pred']

            if 'tune_angle' in _orientations:
                tune_angle = _orientations['tune_angle']
                self.log(f"{task_type}/tune_angle", tune_angle.item(), sync_dist=True)
        else:
            orientations = _orientations

        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations, self.model.pixel_position_reciprocal, self.model.over_sampling)
        # predict slices from HKL
        _slices_pred = self.model.predict_slice(HKL).view((-1, 1,) + (self.model.image_dimension,)*2).clamp(np.log(1e-8), np.log(2000 * self.model.loss_scale_factor))
        if self.model.fluctuation_predictor is not None:
            slices_scale_factor = self.model.fluctuation_predictor(slices_input).unsqueeze(-1).unsqueeze(-1)
            slices_target  = torch.log(general_mask * slices_true * self.model.loss_scale_factor + 1e-8)
            slices_pred    = torch.log(general_mask * torch.exp(_slices_pred) * slices_scale_factor + 1e-8)
        else:
            slices_target  = general_mask * slices_true * self.model.loss_scale_factor
        # We don't want to compare the general masked area
        # loss = self.loss_func((slices_scale_factor * slices_pred)[general_mask.bool()].cpu(), slices_target[general_mask.bool()].cpu())
        loss_reco = self.loss_func(slices_pred[general_mask.bool()].cpu(), slices_target[general_mask.bool()].cpu())
        
        loss = loss_reco
        # loss = loss_reco
        self.log(f"{task_type}/loss", loss.item(), prog_bar=True, sync_dist=True)
        self.log(f"{task_type}/loss_reco", loss_reco.item(), sync_dist=True)

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
                display_volumes(np.log(reciprocal_volume.clip(1e-8, None)) - np.log(1e-8), closefig=True, cmap='gray',
                                vmax=1e-3 * reciprocal_volume.max(),
                                save_to=f'{self.fig_path}/version_{self.logger.version}_{task_type}_reciprocal_vol_log.png')
                save_mrc(f'{self.fig_path}/version_{self.logger.version}_{task_type}_reciprocal_vol.mrc', reciprocal_volume)
                save_mrc(
                    f'{self.fig_path}/version_{self.logger.version}_{task_type}_reciprocal_vol_log.mrc', 
                    np.log(reciprocal_volume.clip(1e-8, None)) - np.log(1e-8)
                )
            except ValueError:
                print('Error in saving reciprocal volume')
                print(reciprocal_volume.shape, reciprocal_volume.max(), reciprocal_volume.min())
                if np.any(np.isnan(reciprocal_volume)):
                    print('NaN in reciprocal volume')
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
        slices_input  = torch.log(input_mask * slices_true * self.model.loss_scale_factor + 1e-8)

        # predict orientations from images
        _orientations = self.model.image_to_orientation(slices_input)
        if isinstance(_orientations, dict):
            if 'rotations' in _orientations:
                orientations = _orientations['rotations']
            else:
                if self.current_epoch < 100:
                    orientations = _orientations['rotations_samp']
                else:
                    orientations = _orientations['rotations_pred']

            if 'tune_angle' in _orientations:
                tune_angle = _orientations['tune_angle']
                self.log(f"{task_type}/tune_angle", tune_angle.item(), sync_dist=True)
        else:
            orientations = _orientations

        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations, self.model.pixel_position_reciprocal, self.model.over_sampling)
        # predict slices from HKL
        _slices_pred = self.model.predict_slice(HKL).view(
            (-1, 1,) + (self.model.image_dimension,)*2
        ).clamp(np.log(1e-8), np.log(2000 * self.model.loss_scale_factor))
        if self.model.fluctuation_predictor is not None:
            slices_scale_factor = self.model.fluctuation_predictor(slices_input).unsqueeze(-1).unsqueeze(-1)
            slices_target  = torch.log(general_mask * slices_true * self.model.loss_scale_factor + 1e-8)
            slices_pred    = torch.log(general_mask * torch.exp(_slices_pred) * slices_scale_factor + 1e-8)
        else:
            slices_target  = general_mask * slices_true * self.model.loss_scale_factor
        # We don't want to compare the general masked area
        # loss = self.loss_func((slices_scale_factor * slices_pred)[general_mask.bool()].cpu(), slices_target[general_mask.bool()].cpu())
        loss_reco = self.loss_func(slices_pred[general_mask.bool()].cpu(), slices_target[general_mask.bool()].cpu())
        
        loss = loss_reco
        # loss = loss_reco
        self.log(f"{task_type}/loss", loss.item(), prog_bar=True, sync_dist=True)
        self.log(f"{task_type}/loss_reco", loss_reco.item(), sync_dist=True)

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
    
    
    def _step(self, batch, batch_idx):
        
        if isinstance(batch, dict):
            slices_true = batch['image'].to(self.dtype)
            input_mask  = batch['input_mask'].bool()
            general_mask = batch['general_mask'].bool()
        else:
            slices_true = batch[0].to(self.dtype)
            input_mask = torch.ones_like(slices_true).bool().bool()
            general_mask = torch.ones_like(slices_true).bool().bool()

        # Apply input and general masks and loss scale factor to get input slices.
        slices_input  = torch.log(input_mask * slices_true * self.model.loss_scale_factor + 1e-8)

        # predict orientations from images
        _orientations = self.model.image_to_orientation(slices_input)
        if isinstance(_orientations, dict):
            if 'rotations' in _orientations:
                orientations = _orientations['rotations']
            else:
                if self.current_epoch < 100:
                    orientations = _orientations['rotations_samp']
                else:
                    orientations = _orientations['rotations_pred']

            if 'tune_angle' in _orientations:
                tune_angle = _orientations['tune_angle']
        else:
            orientations = _orientations

        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations, self.model.pixel_position_reciprocal, self.model.over_sampling)
        # predict slices from HKL
        _slices_pred = self.model.predict_slice(HKL).view(
            (-1, 1,) + (self.model.image_dimension,)*2
        ).clamp(np.log(1e-8), np.log(2000 * self.model.loss_scale_factor))
        if self.model.fluctuation_predictor is not None:
            slices_scale_factor = self.model.fluctuation_predictor(slices_input).unsqueeze(-1).unsqueeze(-1)
            slices_target  = torch.log(general_mask * slices_true * self.model.loss_scale_factor + 1e-8)
            slices_pred    = torch.log(general_mask * torch.exp(_slices_pred) * slices_scale_factor + 1e-8)
        else:
            slices_target  = general_mask * slices_true * self.model.loss_scale_factor
        # We don't want to compare the general masked area
        # loss = self.loss_func((slices_scale_factor * slices_pred)[general_mask.bool()].cpu(), slices_target[general_mask.bool()].cpu())
        loss_reco = self.loss_func(slices_pred[general_mask.bool()].cpu(), slices_target[general_mask.bool()].cpu())
        
        loss = loss_reco

        output = {
            'loss': loss,
            'orientations': orientations,
            'orientations_output': _orientations,
            'slices_pred': slices_pred,
            'slices_target': slices_target,
            'slices_input': slices_input,
        }

        return output
            
            
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
                volume[i] = self.model.predict_intensity(input_coords).detach().cpu().clamp(np.log(1e-8), np.log(2000 * self.model.loss_scale_factor)).numpy().squeeze()
        
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

    
    def volume_symmetry_loss(self, ):
        selected_idx = np.random.choice(np.arange(1, self.symm_ops.shape[0]), 2, replace=False).tolist()
        selected_idx = [0,] + selected_idx
        
        selected_idx = torch.tensor(selected_idx).to(self.device)
        symm_volume = self._predict_symmetrized_reciprocal_volume(self.training_symm_reciprocal_grid[selected_idx])
        loss = (symm_volume - symm_volume[0,None]).abs().mean()
        return loss