import torch
from torch import nn, optim

import os

import lightning as L
import numpy as np
from torchkbnufft import KbNufft

import torchvision.models.resnet as resnet
from pytorch3d.transforms import rotation_6d_to_matrix

from .external.siren_pytorch import SirenNet
from .reconstruction.slicing import get_real_mesh, gen_nonuniform_normalized_positions
from .utils_visualization import display_images_in_parallel

from .utils_model import get_radial_scale_mask


class ResNet2RotMat(nn.Module):
    def __init__(self, size=50, pretrained=False, pool_features=False):
        super().__init__()
        weights = 'DEFAULT' if pretrained else None
        self.resnet = eval(f'resnet.resnet{size}')(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 6)
    
    def forward(self, img):
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        embed = self.resnet(img)
        rotmat = rotation_6d_to_matrix(embed)
        return rotmat
        
class IntensityNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net_mag = SirenNet(*args, **kwargs)
    
    def forward(self, x):
        return self.net_mag(x) + self.net_mag(-x)
    
# class PhaseNet(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.net_phase = SirenNet(*args, **kwargs)
    
#     def forward(self, x):
#         return self.net_phase(x) - self.net_phase(-x)
    
class NeurOrient(L.LightningModule):
    def __init__(self, 
                 pixel_position_reciprocal, 
                 over_sampling=1, 
                 radial_scale_configs=None,
                 photons_per_pulse=1e13,
                 lr=1e-3,
                 weight_decay=1e-4,
                 path=None):
        super().__init__()
        self.save_hyperparameters()

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
        self.orientation_predictor = ResNet2RotMat(
            size=18, pretrained=True, pool_features=False
        )
        
        # setup loss function
        self.loss_func = nn.MSELoss()
        
        # setup volume predictor
        self.volume_predictor = IntensityNet(
            dim_in=3,
            dim_hidden=256,
            dim_out=1,
            num_layers=5,
            final_activation=torch.nn.SiLU(),
        )
            
        self.nufft_forward = KbNufft(im_size=(self.image_dimension,)*3)
        
        self.radial_scale_configs = radial_scale_configs
        if self.radial_scale_configs is None:
            self.register_buffer('radial_scale_factor', torch.ones(1, 1, (self.image_dimension,)*2))
        else:
            _mask = get_radial_scale_mask(
                self.radial_scale_configs['q_values'], 
                self.radial_scale_configs['radial_profile'], 
                self.pixel_position_reciprocal.detach().cpu(), self.radial_scale_configs['alpha'])
            self.register_buffer('radial_scale_factor', torch.from_numpy(_mask).unsqueeze(0))
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.fig_path = os.path.join(self.path, 'figures')
        os.makedirs(self.fig_path, exist_ok=True)
        
        self.photons_per_pulse = photons_per_pulse
        self.loss_scale_factor = 1e14 / self.photons_per_pulse
            
    def image_to_orientation(self, x):
        rotmats = self.orientation_predictor(x)
        return rotmats
        
    def predict_intensity(self, grid_position_reciprocal):
        if grid_position_reciprocal.ndim == 4 and grid_position_reciprocal.shape[-1] == 3:
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
            
    def gen_slices(
            self, ac, orientations
            ):
        """
        Generate model slices using given reference orientations (in quaternion)
        """
        # Get q points (normalized by recirocal extent and oversampling)
        HKL = gen_nonuniform_normalized_positions(
            orientations, self.pixel_position_reciprocal, self.over_sampling)
        ac = ac.real + 0j
        nuvect = self.nufft_forward(ac.unsqueeze(0).unsqueeze(0), HKL)[0,0]
        model_slices = nuvect.real.view((-1,) + (self.image_dimension,)*2)
        return model_slices
    
    def forward(self, slices, return_reconstruction=False):
        slices_input = torch.log(slices * self.loss_scale_factor + 1.)
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
            slices_pred = (torch.exp(slices_pred) - 1) / self.loss_scale_factor
            return orientations, slices_pred

    def training_step(self, batch, batch_idx):
        slices_true = batch[0].to(self.device)
        
        slices_input = torch.log(slices_true * self.loss_scale_factor + 1.)

        # predict orientations from images
        orientations = self.image_to_orientation(slices_input)
        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations, self.pixel_position_reciprocal, self.over_sampling)
        # predict slices from HKL
        slices_pred = self.predict_slice(HKL).view((-1, 1,) + (self.image_dimension,)*2)
        
        loss = self.loss_func(slices_pred.cpu(), slices_input.cpu())
        self.log("train_loss", loss.item())
        
        # display_volumes(rho, save_to=f'{self.path}/rho.png')
        if self.global_step % 10 == 0:
            num_figs = min(10, slices_true.shape[0])
            slice_disp = (torch.exp(slices_pred[:num_figs]) - 1) / self.loss_scale_factor
            display_images_in_parallel(slice_disp, slices_true[:num_figs], save_to=f'{self.fig_path}/version_{self.logger.version}_train.png')
        
        return loss

    def validation_step(self, batch, batch_idx):
        slices_true = batch[0].to(self.device)
        
        slices_input = torch.log(slices_true * self.loss_scale_factor + 1.)

        # predict orientations from images
        orientations = self.image_to_orientation(slices_input)
        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations, self.pixel_position_reciprocal, self.over_sampling)
        # predict slices from HKL
        slices_pred = self.predict_slice(HKL).view((-1, 1,) + (self.image_dimension,)*2)
        
        loss = self.loss_func(slices_pred.cpu(), slices_input.cpu())
        self.log("val_loss", loss.item())
        
        # display_volumes(rho, save_to=f'{self.path}/rho.png')
        if self.global_step % 10 == 0:
            num_figs = min(10, slices_true.shape[0])
            slice_disp = (torch.exp(slices_pred[:num_figs]) - 1) / self.loss_scale_factor
            display_images_in_parallel(slice_disp, slices_true[:num_figs], save_to=f'{self.fig_path}/version_{self.logger.version}_val.png')
        
        
    def configure_optimizers(self):
        # optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
        #                                                  mode           = 'min',
        #                                                  factor         = 2e-1,
        #                                                  patience       = 10,
        #                                                  threshold      = 1e-4,
        #                                                  threshold_mode ='rel',
        #                                                  verbose        = True,
        #                                                  min_lr         = 1e-6
        #                                                 )
        # return {'optimizer': optimizer,
        #         'scheduler': scheduler,
        #         'monitor': 'val_loss'}
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
