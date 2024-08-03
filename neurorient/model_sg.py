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
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix

from .external.siren_pytorch import SirenNet
from .reconstruction.slicing import get_real_mesh, gen_nonuniform_normalized_positions
from .utils_visualization import display_images_in_parallel, display_volumes
from .lr_scheduler import CosineLRScheduler
from .so3_relative_angle import so3_relative_angle

class KbNufftRealView(KbNufft):
    def __init__(self, im_size, grid_size = None):
        super(KbNufftRealView, self).__init__(im_size, grid_size)

        # Convert all buffers to real view
        for name, buf in self.named_buffers():
            if (buf.dtype != torch.complex128) and (buf.dtype != torch.complex64): continue
            real_view_buf = torch.view_as_real(buf)
            self.register_buffer(name, real_view_buf)

class Slice2MultiRotMat(nn.Module):
    def __init__(self, size=18, pretrained=False, N_rotmat=5):
        super().__init__()
        weights = 'DEFAULT' if pretrained else None
        self.resnet = eval(f'resnet.resnet{size}')(weights=weights)

        # Average the weights in the input channels...
        conv1_weight = self.resnet.conv1.weight.data.mean(dim = 1, keepdim = True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.conv1.weight.data = conv1_weight
        # Output 6D rotation matrix
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, N_rotmat * 6)
    
    def forward(self, img):
        if img.ndim == 3:
            img = img.unsqueeze(1)
        embed = self.resnet(img)
        embed = embed.view(embed.shape[0], -1, 6)
        # print(embed.shape)
        rotmat = rotation_6d_to_matrix(embed)
        return rotmat
    
class Slice2RotMat_PG(nn.Module):
    def __init__(self, size=18, pretrained=False, point_group=None):
        super().__init__()
        weights = 'DEFAULT' if pretrained else None
        self.resnet = eval(f'resnet.resnet{size}')(weights=weights)

        # Average the weights in the input channels...
        conv1_weight = self.resnet.conv1.weight.data.mean(dim = 1, keepdim = True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.conv1.weight.data = conv1_weight
        # Output 6D rotation matrix
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 6)
        
        self.point_group = point_group
        _point_group_ops = so3_point_group_operations(point_group)
        self.register_buffer('point_group_ops', _point_group_ops)
        self.point_group_order = len(self.point_group_ops)
    
    def forward(self, img):
        bs = img.shape[0]
        if img.ndim == 3:
            img = img.unsqueeze(1)
        embed = self.resnet(img)
        rotmat = rotation_6d_to_matrix(embed)
        
        pg_op_indices = torch.from_numpy(np.random.choice(np.arange(self.point_group_order), bs, replace=True)).to(img.device)
        pg_ops = self.point_group_ops[pg_op_indices]
        rotmat = torch.einsum('bij,bjk -> bik', pg_ops, rotmat)
        
        return rotmat
    
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
    
from .external.image2sphere.reduced_predictor import I2S, flat_wigner
from .external.image2sphere.so3_utils import so3_healpix_grid
from e3nn import o3
import torch.nn.functional as F
from .so3_decomposition import so3_unique_subset, so3_point_group_operations

class Slice2RotMat_I2S(nn.Module):
    def __init__(self, input_size, point_group=None, mode='mean', rec_level=2, lmax=3, size=18, pretrained=False):
        super().__init__()
        self.i2s = I2S(input_size, lmax=lmax)
        # output_xyx = so3_healpix_grid(rec_level=rec_level) # 37K points
        self.point_group = point_group
        if point_group is not None:
            output_xyx, op_tensor = so3_unique_subset(point_group, rec_level)
            self.register_buffer('op_tensor', op_tensor)
        else:
            output_xyx = so3_healpix_grid(rec_level=rec_level)
        
        output_wigners = flat_wigner(lmax, *output_xyx).transpose(0, 1)
        output_rotmats = o3.angles_to_matrix(*output_xyx)
        output_quat = matrix_to_quaternion(output_rotmats)
        
        self.register_buffer('output_wigners', output_wigners)
        self.register_buffer('output_quat', output_quat)
        self.register_buffer('output_rotmats', output_rotmats)
        
        self.mode = mode
    
    def __repr__(self):
        return f'{self.__class__.__name__}(mode={self.mode})'
    
    def forward(self, img, mode=None):
        if img.ndim == 3:
            img = img.unsqueeze(1)
        if mode is None:
            mode = self.mode
        logits = self.i2s.compute_probabilities(img, self.output_wigners, return_logits=True)
        if mode == 'mean':
            probs = F.gumbel_softmax(logits, tau=1, hard=False)
            quat = torch.einsum('ij, jk -> ik', probs, self.output_quat)
            quat = quat / quat.norm(dim=-1, keepdim=True)
            rotmat = quaternion_to_matrix(quat)
        elif mode == 'sampling':
            probs = F.softmax(logits, dim=-1)
            sample_indices = torch.multinomial(probs, 1, replacement=False).squeeze(-1)
            onehot_indices = F.one_hot(sample_indices, num_classes=probs.shape[-1]).float()
            one_hot_straight_through = onehot_indices - probs.detach() + probs
            rotmat = torch.einsum('bi,ijk->bjk', one_hot_straight_through, self.output_rotmats)
            
        if self.point_group is not None:
            rotmat = torch.einsum('gij,bjk -> gbik', self.op_tensor, rotmat)
        return rotmat
    
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

class Slice2RotMat_BIFPN(nn.Module):
    def __init__(
            self, 
            size=18, 
            pretrained=False,
            input_size=(128, 128),
            num_features=64,
            num_blocks=1,
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
        output_channels = self.backbone.output_channels
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
            _x = torch.randn(1, 1, *input_size)
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
        return self.net_mag(x)

class NeurOrient(nn.Module):
    def __init__(self, 
                 pixel_position_reciprocal, 
                 over_sampling=1,
                 photons_per_pulse=1e13,
                 use_bifpn=False,
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
        self.orientation_predictor = Slice2RotMat_PG(**config_slice2rotmat)
        # self.orientation_predictor = Slice2MultiRotMat(**config_slice2rotmat)
        # config_slice2rotmat['input_size'] = (self.image_dimension.item(),)*2
        # self.orientation_predictor = Slice2RotMat_I2S(**config_slice2rotmat)
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
            final_activation=torch.nn.ReLU(),
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
        orientations = self.image_to_orientation(slices_input)[0]
        if not return_reconstruction:
            return orientations
        else:
            # get reciprocal positions based on orientations
            # HKL has shape (3, num_qpts)
            HKL = gen_nonuniform_normalized_positions(
                orientations, self.pixel_position_reciprocal, self.over_sampling)
            # predict slices from HKL
            slices_pred = self.predict_slice(HKL).view((-1, 1,) + (self.image_dimension,)*2)

            return orientations, (torch.exp(slices_pred) - 1) / self.loss_scale_factor


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
                 config_intensitynet={'dim_hidden': 256, 'num_layers': 5},
                 config_optimization={'lr': 1e-3, 'weight_decay': 1e-4, 'loss_func': 'MSELoss'},
                #  config_orientation_diversity_loss={'max': 10.0, 'min': 1e-2, 'scale': 0.05},
                 config_orientation_diversity_loss=None,):
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
        # self.lr = config_optimization['lr']
        # self.weight_decay = config_optimization['weight_decay']
        # self.loss_func = eval(f"torch.nn.{config_optimization['loss_func']}()")
        
        # for key, value in config_optimization.items():
        #     self.__setattr__(key, value)
        
        self.config_orientation_diversity_loss = config_orientation_diversity_loss
        self.configure_optimization = config_optimization
        if config_optimization['loss_func'] != 'PoissonNLLLoss':
            self.loss_func = eval(f"torch.nn.{config_optimization['loss_func']}()")
            self.log_transform = True
        else:
            self.loss_func = torch.nn.PoissonNLLLoss(log_input=False, full=True)
            self.log_transform = False
        
    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            slices_true = batch['image'].to(self.dtype).clamp_min(0.0)
            input_mask  = batch['input_mask'].bool()
            general_mask = batch['general_mask'].bool()
        else:
            slices_true = batch[0].to(self.dtype).clamp_min(0.0)
            input_mask = torch.ones_like(slices_true).bool().bool()
            general_mask = torch.ones_like(slices_true).bool().bool()


        # Apply input and general masks and loss scale factor to get input slices.
        slices_input  = input_mask  * general_mask * torch.log(slices_true * self.model.loss_scale_factor + 1.)

        # predict orientations from images
        orientations = self.model.image_to_orientation(slices_input)

        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations.reshape(-1,3,3), self.model.pixel_position_reciprocal, self.model.over_sampling)
        # predict slices from HKL
        slices_pred = self.model.predict_slice(HKL).view((orientations.shape[0], 1,) + (self.model.image_dimension,)*2)


        if self.model.fluctuation_predictor is not None:
            slices_scale_factor = self.model.fluctuation_predictor(slices_input).unsqueeze(-1).unsqueeze(-1)
            if self.log_transform:
                slices_target  = general_mask * torch.log(slices_true * self.model.loss_scale_factor + 1.)
                slices_pred    = torch.log((torch.exp(slices_pred) - 1) * slices_scale_factor + 1)
            else:
                slices_target  = general_mask * slices_true * self.model.loss_scale_factor
                slices_pred    = (torch.exp(slices_pred) - 1) * slices_scale_factor
        else:
            if self.log_transform:
                slices_target  = general_mask * torch.log(slices_true * self.model.loss_scale_factor + 1.)
            else:
                slices_target  = general_mask * slices_true * self.model.loss_scale_factor


        # We don't want to compare the general masked area
        loss = self.loss_func(
            slices_pred[general_mask.bool()].cpu(), 
            slices_target[general_mask.bool()].cpu())
        self.log("train_loss", loss.item(), prog_bar=True, sync_dist=True)
        
        # display_volumes(rho, save_to=f'{self.path}/rho.png')
        if self.global_step % 10 == 0:
            self.get_figure_save_dir()
            num_figs = min(10, slices_true.shape[0])
            if self.log_transform:
                slice_disp = (torch.exp(slices_pred[:num_figs]) - 1) / self.model.loss_scale_factor
            else:
                slice_disp = slices_pred[:num_figs] / self.model.loss_scale_factor
            slice_disp = slice_disp * slices_true[:num_figs].max() / (slice_disp.max() + 1e-6)
            display_images_in_parallel(slice_disp, slices_true[:num_figs], save_to=f'{self.fig_path}/version_{self.logger.version}_train.png')
            if isinstance(input_mask, torch.Tensor):
                slice_disp_input = (torch.exp(slices_input[:num_figs]) - 1) / self.model.loss_scale_factor
                display_images_in_parallel(input_mask[:num_figs], slice_disp_input, 
                                           titles = ('Input Masks', 'Input Slices'),
                                           save_to=f'{self.fig_path}/version_{self.logger.version}_train_in.png')
            if isinstance(general_mask, torch.Tensor):
                if self.log_transform:
                    slice_disp_output = (torch.exp(slices_target[:num_figs]) - 1) / self.model.loss_scale_factor
                else:
                    slice_disp_output = slices_target[:num_figs] / self.model.loss_scale_factor
                display_images_in_parallel(general_mask[:num_figs], slice_disp_output, 
                                           titles = ('General Masks', 'Output Slices'),
                                           save_to=f'{self.fig_path}/version_{self.logger.version}_train_out.png')
            
            reciprocal_volume = self.predict_reciprocal_volume()
            display_volumes(reciprocal_volume, closefig=True, cmap='gray',
                            vmax=1e-3 * reciprocal_volume.max(),
                            save_to=f'{self.fig_path}/version_{self.logger.version}_train_reciprocal_vol.png')

        return loss


    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            slices_true = batch['image'].to(self.dtype).clamp_min(0.0)
            input_mask  = batch['input_mask'].bool()
            general_mask = batch['general_mask'].bool()
        else:
            slices_true = batch[0].to(self.dtype).clamp_min(0.0)
            input_mask = torch.ones_like(slices_true).bool().bool()
            general_mask = torch.ones_like(slices_true).bool().bool()


        # Apply input and general masks and loss scale factor to get input slices.
        slices_input  = input_mask  * general_mask * torch.log(slices_true * self.model.loss_scale_factor + 1.)

        # predict orientations from images
        orientations = self.model.image_to_orientation(slices_input)

        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations.reshape(-1,3,3), self.model.pixel_position_reciprocal, self.model.over_sampling)
        # predict slices from HKL
        slices_pred = self.model.predict_slice(HKL).view((orientations.shape[0], 1,) + (self.model.image_dimension,)*2)


        if self.model.fluctuation_predictor is not None:
            slices_scale_factor = self.model.fluctuation_predictor(slices_input).unsqueeze(-1).unsqueeze(-1)
            if self.log_transform:
                slices_target  = general_mask * torch.log(slices_true * self.model.loss_scale_factor + 1.)
                slices_pred    = torch.log((torch.exp(slices_pred) - 1) * slices_scale_factor + 1)
            else:
                slices_target  = general_mask * slices_true * self.model.loss_scale_factor
                slices_pred    = (torch.exp(slices_pred) - 1) * slices_scale_factor
        else:
            if self.log_transform:
                slices_target  = general_mask * torch.log(slices_true * self.model.loss_scale_factor + 1.)
            else:
                slices_target  = general_mask * slices_true * self.model.loss_scale_factor


        # We don't want to compare the general masked area
        loss = self.loss_func(
            slices_pred[general_mask.bool()].cpu(), 
            slices_target[general_mask.bool()].cpu())
        self.log("val_loss", loss.item(), prog_bar=True, sync_dist=True)
        
        if self.global_step % 10 == 0:
            self.get_figure_save_dir()
            num_figs = min(10, slices_true.shape[0])
            if self.log_transform:
                slice_disp = (torch.exp(slices_pred[:num_figs]) - 1) / self.model.loss_scale_factor
            else:
                slice_disp = slices_pred[:num_figs] / self.model.loss_scale_factor
            slice_disp = slice_disp * slices_true[:num_figs].max() / (slice_disp.max() + 1e-6)
            display_images_in_parallel(slice_disp, slices_true[:num_figs], save_to=f'{self.fig_path}/version_{self.logger.version}_val.png')
            if isinstance(input_mask, torch.Tensor):
                slice_disp_input = (torch.exp(slices_input[:num_figs]) - 1) / self.model.loss_scale_factor
                display_images_in_parallel(input_mask[:num_figs], slice_disp_input, 
                                           titles = ('Input Masks', 'Input Slices'),
                                           save_to=f'{self.fig_path}/version_{self.logger.version}_val_in.png')
            if isinstance(general_mask, torch.Tensor):
                if self.log_transform:
                    slice_disp_output = (torch.exp(slices_target[:num_figs]) - 1) / self.model.loss_scale_factor
                else:
                    slice_disp_output = slices_target[:num_figs] / self.model.loss_scale_factor
                display_images_in_parallel(general_mask[:num_figs], slice_disp_output, 
                                           titles = ('General Masks', 'Output Slices'),
                                           save_to=f'{self.fig_path}/version_{self.logger.version}_val_out.png')
            
            reciprocal_volume = self.predict_reciprocal_volume()
            display_volumes(reciprocal_volume, closefig=True, cmap='gray',
                            vmax=1e-3 * reciprocal_volume.max(),
                            save_to=f'{self.fig_path}/version_{self.logger.version}_val_reciprocal_vol.png')
            
            
    def configure_optimizers(self):
        if not 'scheduler' in self.configure_optimization:
            optimizer = optim.AdamW(self.parameters(), lr=self.configure_optimization['lr'], weight_decay=self.configure_optimization['weight_decay'])
            return optimizer
        else:
            _configure_optimization = deepcopy(self.configure_optimization)
            optimizer = optim.AdamW(self.parameters(), lr=_configure_optimization['lr'], weight_decay=_configure_optimization['weight_decay'])
            # scheduler = scheduler_dict[self.configure_optimization['scheduler'].pop('name')](
            #     optimizer     = optimizer, 
            #     warmup_epochs = _configure_optimization['scheduler']['warmup_epochs'],
            #     total_epochs  = _configure_optimization['scheduler']['total_epochs'],
            #     min_lr        = _configure_optimization['scheduler']['min_lr'])
            scheduler = scheduler_dict[_configure_optimization['scheduler']['name']](
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
            
    def predict_reciprocal_volume(self,):
        grid_reciprocal = np.pi * self.model.grid_position_reciprocal / self.model.grid_position_reciprocal.max()
        volume = np.zeros(grid_reciprocal.shape[:3])
        with torch.no_grad():
            for i in range(grid_reciprocal.shape[0]):
                input_coords = grid_reciprocal[i,None,...].to(self.device)
                volume[i] = self.model.predict_intensity(input_coords).detach().cpu().numpy().squeeze()
        volume = (np.exp(volume) - 1) / self.model.loss_scale_factor
        return volume.clip(0.0)

    def rotation_diversity_loss(self, rot_mat):
        """ 
        params:
            rot_mat: (B, 3, 3)
        outputs:
            loss: (B, B)
        """
        bs = rot_mat.shape[0]
        relative_angle_matrix = so3_relative_angle(rot_mat, rot_mat)
        triu_mask = torch.triu(torch.ones_like(relative_angle_matrix), diagonal=1).bool()
        relative_angles = relative_angle_matrix[triu_mask]
        # print(relative_angles)
        return torch.exp(-10 * relative_angles).mean()
    
    def get_rotation_diversity_loss_weight(self,):
        try:
            w_max = self.config_orientation_diversity_loss['max']
            w_min = self.config_orientation_diversity_loss['min']
            w_scale = self.config_orientation_diversity_loss['scale']
            w = w_min + (w_max-w_min) * np.exp(-w_scale * self.current_epoch)
        except TypeError:
            w = 1.0
        return w