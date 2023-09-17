import torch
from torch import nn, optim

import os

import numpy as np
from torchkbnufft import KbNufft

from .image_encoder import ImageEncoder
from .bifpn import DepthwiseSeparableConv2d, BiFPN
from pytorch3d.transforms import rotation_6d_to_matrix

from .external.siren_pytorch import SirenNet
from .reconstruction.slicing import get_real_mesh, gen_nonuniform_normalized_positions
from .utils_visualization import display_images_in_parallel

from .utils_model import get_radial_scale_mask

from .config import CONFIG

class ResNet2RotMat(nn.Module):
    def __init__(self, size=50, pretrained=False):
        super().__init__()
        resnet_type = f"resnet{size}"
        self.backbone = ImageEncoder(resnet_type, pretrained)

        # Create the adapter layer between backbone and bifpn...
        self.backbone_to_bifpn = nn.ModuleList([
            DepthwiseSeparableConv2d(in_channels  = in_channels,
                                     out_channels = CONFIG.BIFPN.NUM_FEATURES,
                                     kernel_size  = 1,
                                     stride       = 1,
                                     padding      = 0)
            for _, in_channels in CONFIG.BACKBONE.OUTPUT_CHANNELS.items()
        ])

        self.bifpn = BiFPN(num_blocks   = CONFIG.BIFPN.NUM_BLOCKS,
                           num_features = CONFIG.BIFPN.NUM_FEATURES,
                           num_levels   = CONFIG.BIFPN.NUM_LEVELS)

        self.regressor_head = nn.Linear(CONFIG.REGRESSOR_HEAD.IN_FEATURES, CONFIG.REGRESSOR_HEAD.OUT_FEATURES)

    def forward(self, x):
        # Calculate and save feature maps in multiple resolutions...
        fmap_in_backbone_layers = self.backbone(x)

        # Apply the BiFPN adapter...
        bifpn_input_list = []
        for idx, fmap in enumerate(fmap_in_backbone_layers):
            bifpn_input = self.backbone_to_bifpn[idx](fmap)
            bifpn_input_list.append(bifpn_input)

        # Apply the BiFPN layer...
        bifpn_output_list = self.bifpn(bifpn_input_list)

        # Use the -2-th feature maps for regression...
        regressor_input = bifpn_output_list[-2]
        B, C, H, W = regressor_input.shape
        regressor_input = regressor_input.view(B, C * H * W)
        logits = self.regressor_head(regressor_input)

        rotmat = rotation_6d_to_matrix(logits)
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

class NeurOrient(nn.Module):
    def __init__(self, 
                 pixel_position_reciprocal, 
                 over_sampling=1, 
                 radial_scale_configs=None,
                 photons_per_pulse=1e13,
                 lr=1e-3,
                 weight_decay=1e-4,
                 path=None):
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
        self.orientation_predictor = ResNet2RotMat(
            size=CONFIG.BACKBONE.RES_TYPE, pretrained=True,
        )

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


    def forward(self, x):
        slices_true = x

        slices_true = nn.functional.relu(slices_true * self.loss_scale_factor + 1.) + 1e-6 # Cutoff before log
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
