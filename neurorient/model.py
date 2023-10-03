import torch
from torch import nn, optim
import torchvision.models.resnet as resnet

import os

import lightning as L
import numpy as np
from torchkbnufft import KbNufft
from pathlib import Path
from copy import deepcopy

from .image_encoder import ImageEncoder
from .bifpn import DepthwiseSeparableConv2d, BiFPN
from pytorch3d.transforms import rotation_6d_to_matrix

from .external.siren_pytorch import SirenNet
from .reconstruction.slicing import get_real_mesh, gen_nonuniform_normalized_positions
from .utils_visualization import display_images_in_parallel
from .lr_scheduler import CosineLRScheduler

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

class Slice2RotMat_BIFPN(nn.Module):
    def __init__(
            self, 
            size=18, 
            pretrained=False,
            num_features=64,
            num_blocks=1,
            output_channels={
                "relu"   : 64,
                "layer1" : 256,
                "layer2" : 512,
                "layer3" : 1024,
                "layer4" : 2048,
            },
            num_levels=5,
            regressor_out_features=6,
            scale=-1,
            ):
        super().__init__()
        resnet_type = f"resnet{size}"
        self.backbone = ImageEncoder(resnet_type, pretrained)
        self.num_levels = num_levels
        self.scale = scale

        # Create the adapter layer between backbone and bifpn...
        self.backbone_to_bifpn = nn.ModuleList([
            DepthwiseSeparableConv2d(in_channels  = in_channels,
                                     out_channels = num_features,
                                     kernel_size  = 1,
                                     stride       = 1,
                                     padding      = 0) 
            if num_blocks > 0 else nn.Identity()
            for _, in_channels in output_channels.items()
        ])[-num_levels:]    # Only consider fmaps from the most coarse level

        self.bifpn = BiFPN(num_blocks   = num_blocks,
                           num_features = num_features,
                           num_levels   = num_levels) \
                     if num_blocks > 0 else           \
                     nn.Identity()
        with torch.no_grad():
            _x = torch.randn(1, 1, 128, 128)
            _out_shape = self.forward_without_regressor(_x).shape

        self.regressor_head = nn.Linear(_out_shape[-1], regressor_out_features)
        del _x, _out_shape

    def forward_without_regressor(self, x):
        # Calculate and save feature maps in multiple resolutions...
        fmap_in_backbone_layers = self.backbone(x)
        fmap_in_backbone_layers = fmap_in_backbone_layers[-self.num_levels:]    # Only consider fmaps from the most coarse level

        # Apply the BiFPN adapter...
        bifpn_input_list = []
        for idx, fmap in enumerate(fmap_in_backbone_layers):
            bifpn_input = self.backbone_to_bifpn[idx](fmap)
            bifpn_input_list.append(bifpn_input)

        # Apply the BiFPN layer...
        bifpn_output_list = self.bifpn(bifpn_input_list)

        # Use the N-th feature maps for regression...
        regressor_input = bifpn_output_list[self.scale]
        B, C, H, W = regressor_input.shape
        regressor_input = regressor_input.view(B, C * H * W)
        return regressor_input
        
    def forward(self, x):
        regressor_input = self.forward_without_regressor(x)
        logits = self.regressor_head(regressor_input)

        rotmat = rotation_6d_to_matrix(logits)
        return rotmat

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
        if use_bifpn:
            self.orientation_predictor = Slice2RotMat_BIFPN(**config_slice2rotmat)
        else:
            self.orientation_predictor = Slice2RotMat(**config_slice2rotmat)

        # setup volume predictor
        self.volume_predictor = IntensityNet(
            dim_in=3,
            dim_hidden=config_intensitynet['dim_hidden'],
            dim_out=1,
            num_layers=config_intensitynet['num_layers'],
            final_activation=torch.nn.SiLU(),
        )

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


    def estimate(self, x, return_reconstruction=False):
        slices_true = x

        slices_true = slices_true * self.loss_scale_factor + 1.
        slices_input = torch.log(slices_true)

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

            return orientations, slices_pred


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
                 config_slice2rotmat={'size': 18, 'pretrained': True},
                 config_intensitynet={'dim_hidden': 256, 'num_layers': 5},
                 config_optimization={'lr': 1e-3, 'weight_decay': 1e-4, 'loss_func': 'MSELoss'}):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = NeurOrient(
            pixel_position_reciprocal, 
            over_sampling=over_sampling,
            photons_per_pulse=photons_per_pulse,
            use_bifpn=use_bifpn,
            config_slice2rotmat=config_slice2rotmat,
            config_intensitynet=config_intensitynet,
        )
        # self.lr = config_optimization['lr']
        # self.weight_decay = config_optimization['weight_decay']
        # self.loss_func = eval(f"torch.nn.{config_optimization['loss_func']}()")
        
        # for key, value in config_optimization.items():
        #     self.__setattr__(key, value)
        
        self.configure_optimization = config_optimization
        self.loss_func = eval(f"torch.nn.{config_optimization['loss_func']}()")
        
    def training_step(self, batch, batch_idx):
        slices_true = batch[0].to(self.device)

        slices_input = torch.log(slices_true * self.model.loss_scale_factor + 1.)

        # predict orientations from images
        orientations = self.model.image_to_orientation(slices_input)
        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations, self.model.pixel_position_reciprocal, self.model.over_sampling)
        # predict slices from HKL
        slices_pred = self.model.predict_slice(HKL).view((-1, 1,) + (self.model.image_dimension,)*2)

        loss = self.loss_func(slices_pred.cpu(), slices_input.cpu())
        self.log("train_loss", loss.item())

        # display_volumes(rho, save_to=f'{self.path}/rho.png')
        if self.global_step % 10 == 0:
            self.get_figure_save_dir()
            num_figs = min(10, slices_true.shape[0])
            slice_disp = (torch.exp(slices_pred[:num_figs]) - 1) / self.model.loss_scale_factor
            display_images_in_parallel(slice_disp, slices_true[:num_figs], save_to=f'{self.fig_path}/version_{self.logger.version}_train.png')

        return loss


    def validation_step(self, batch, batch_idx):
        slices_true = batch[0].to(self.device)

        slices_input = torch.log(slices_true * self.model.loss_scale_factor + 1.)

        # predict orientations from images
        orientations = self.model.image_to_orientation(slices_input)
        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations, self.model.pixel_position_reciprocal, self.model.over_sampling)
        # predict slices from HKL
        slices_pred = self.model.predict_slice(HKL).view((-1, 1,) + (self.model.image_dimension,)*2)

        loss = self.loss_func(slices_pred.cpu(), slices_input.cpu())
        self.log("val_loss", loss.item())

        # display_volumes(rho, save_to=f'{self.path}/rho.png')
        if self.global_step % 10 == 0:
            self.get_figure_save_dir()
            num_figs = min(10, slices_true.shape[0])
            slice_disp = (torch.exp(slices_pred[:num_figs]) - 1) / self.model.loss_scale_factor
            display_images_in_parallel(slice_disp, slices_true[:num_figs], save_to=f'{self.fig_path}/version_{self.logger.version}_val.png')
            
            
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
                os.path.join(self.trainer.logger.save_dir, 'lightning_logs', f'version_{self.trainer.logger.version}', 'figures')
            )
            self.fig_path.mkdir(parents=True, exist_ok=True)