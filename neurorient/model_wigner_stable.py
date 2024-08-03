import torch
from torch import nn, optim
import torchvision.models.resnet as resnet

import os

import lightning as L
import numpy as np
from torchkbnufft import KbNufft
from pathlib import Path
from copy import deepcopy

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, random_quaternions

from .external.siren_pytorch import SirenNet
from .reconstruction.slicing import get_real_mesh, gen_nonuniform_normalized_positions
from .utils_visualization import display_images_in_parallel, display_volumes
from .lr_scheduler import CosineLRScheduler
from .so3_relative_angle import so3_relative_angle

from .external.rsh import rsh_cart_6, rsh_cart_5
from .wigner_basis import WignerD6Basis


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
        if embed.isnan().any():
            print('NaN in embed')
            # embed[embed.isnan()] = torch.randn_like(embed[embed.isnan()])
            embed = torch.nan_to_num(embed, nan=1.0, posinf=10.0, neginf=-10.0)
            print("Replaced NaN with random values")
        embed = torch.tanh(embed)
        rotmat = rotation_6d_to_matrix(embed)
        if rotmat.isnan().any():
            print('NaN in rotmat')
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
        self.output_silu = nn.SiLU()
    
    def forward(self, img):
        if img.ndim == 3:
            img = img.unsqueeze(1)
        output = self.resnet(img)
        output = self.output_silu(output)
        return output

# class IntensityNet(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.net_mag = SirenNet(*args, **kwargs)

#     def forward(self, x):
#         return self.net_mag(x)
    

class IntensityNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # self.net_mag = SirenNet(*args, **kwargs)
        self.f_q = SirenNet(
            dim_in=3,
            dim_hidden=256,
            dim_out=13,
            num_layers=5,
            final_activation=None,
        )
        
        # self.f_q = nn.Sequential(
        #     nn.Linear(1, 256),
        #     nn.SiLU(),
        #     nn.Linear(256, 256),
        #     nn.SiLU(),
        #     nn.Linear(256, 256),
        #     nn.SiLU(),
        #     nn.Linear(256, 256),
        #     nn.SiLU(),
        #     nn.Linear(256, 13),
        # )

    def forward(self, x):
        q = x.norm(dim=-1, keepdim=True)
        return self.f_q(x)
    
    # def forward(self, x):
    #     q = x.norm(dim=-1, keepdim=True)
    #     phi = torch.acos(x[...,2] / (q.squeeze() + 1e-6)).reshape(-1,1)
    #     theta = torch.atan2(x[...,1], x[...,0]).reshape(-1,1)
    #     input = torch.cat([q, phi, theta], dim=-1)
    #     return self.f_q(input)

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
        self.orientation_predictor = Slice2RotMat()
        if use_fluctuation_predictor:
            self.fluctuation_predictor = FluctuationPredictor(config_slice2rotmat['size'], config_slice2rotmat['pretrained'])
        else:
            self.fluctuation_predictor = None
            
        # setup volume predictor
        self.volume_predictor = IntensityNet()
        self.wigner = WignerD6Basis()
        # d   = self.wigner.evaluate_spha_coefficients(random_quaternions(1)).squeeze(0)
        d   = self.wigner.evaluate_spha_coefficients(torch.tensor([[1.,0.,0.,0.]])).squeeze(0)
        
        self.register_buffer('d', d)
        

        self.photons_per_pulse = photons_per_pulse
        self.loss_scale_factor = 1e14 / self.photons_per_pulse

    def image_to_orientation(self, x):
        rotmats = self.orientation_predictor(x)
        return rotmats

    def predict_intensity(self, HKL):
        if HKL.ndim == 4 and HKL.shape[-1] == 3:
            out_shape = HKL.shape[:-1]
            HKL = HKL.view(-1, 3)
            # intensity = self.volume_predictor(grid_position_reciprocal).view(out_shape)
        else:
            out_shape = None
        # print(HKL.shape)
        q_HKL = HKL.norm(dim=-1, keepdim=True)
        f_q = self.volume_predictor(HKL)
        Ylm = rsh_cart_6(HKL / (q_HKL + 1e-6))[...,-13:]
        # d   = self.wigner.evaluate_spha_coefficients(torch.tensor([[1.,0.,0.,0.]]).to(HKL.device)).squeeze(0)
        d = self.d
        
        if f_q.isnan().any():
            print('NaN in f_q')
            f_q[f_q.isnan()] = 0.0
        if d.isnan().any():
            print('NaN in d')
            d[d.isnan()] = 0.0
        if Ylm.isnan().any():
            print('NaN in Ylm')
            Ylm[Ylm.isnan()] = 0.0
            
        intensity = torch.einsum('qj, qj, j -> q', f_q, Ylm, d).pow(2)
        
        if out_shape is not None:
            intensity = intensity.view(out_shape)
            
        return intensity
    
    # def predict_intensity(self, grid_position_reciprocal):
    #     if grid_position_reciprocal.ndim == 4 and grid_position_reciprocal.shape[-1] == 3:
    #         out_shape = grid_position_reciprocal.shape[:-1]
    #         grid_position_reciprocal = grid_position_reciprocal.view(-1, 3)
    #         intensity = self.volume_predictor(grid_position_reciprocal).view(out_shape)
    #     else:
    #         intensity = self.volume_predictor(grid_position_reciprocal)
    #     return intensity

    def predict_slice(self, HKL):
        if HKL.ndim == 2 and HKL.shape[0] == 3:
            HKL = HKL.T
        q_HKL = HKL.norm(dim=-1, keepdim=True)
        # print(q_HKL.min(), q_HKL.max())
        f_q = self.volume_predictor(HKL)
        Ylm = rsh_cart_6(HKL / (q_HKL + 1e-6))[...,-13:]
        # d   = self.wigner.evaluate_spha_coefficients(torch.tensor([[1.,0.,0.,0.]]).to(HKL.device)).squeeze(0)
        d = self.d  
        
        if f_q.isnan().any():
            print('NaN in f_q')
            f_q[f_q.isnan()] = 0.0
        if d.isnan().any():
            print('NaN in d')
            d[d.isnan()] = 0.0
        if Ylm.isnan().any():
            print('NaN in Ylm')
            Ylm[Ylm.isnan()] = 0.0
            
        slices = torch.einsum('qj, qj, j -> q', f_q, Ylm, d).pow(2)
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
        self.automatic_optimization = False
        
        self.model = NeurOrient(
            pixel_position_reciprocal, 
            over_sampling=over_sampling,
            photons_per_pulse=photons_per_pulse,
            use_bifpn=use_bifpn,
            use_fluctuation_predictor=use_fluctuation_predictor,
            config_slice2rotmat=config_slice2rotmat,
            config_intensitynet=config_intensitynet,
        )
        
        self.config_orientation_diversity_loss = config_orientation_diversity_loss
        self.configure_optimization = config_optimization
        if config_optimization['loss_func'] != 'PoissonNLLLoss':
            self.loss_func = eval(f"torch.nn.{config_optimization['loss_func']}()")
            self.log_transform = True
        else:
            self.loss_func = torch.nn.PoissonNLLLoss(log_input=False, full=True)
            self.log_transform = False
            
        
    def training_step(self, batch, batch_idx):
        
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        
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

        # return loss
        
        
        opt.zero_grad()
        self.manual_backward(loss)
        
        for p in self.parameters():
            # print(p.grad.abs().max())
            if p.grad is not None:
                if p.grad.isnan().any():
                    print('NaN in gradients')
                    p.grad[p.grad.isnan()] = 0.0
                if p.grad.isinf().any():
                    print('Inf in gradients')
                    p.grad[p.grad.isinf()] = 0.0
        
        # clip gradients
        # self.clip_gradients(opt, gradient_clip_val=0.4, gradient_clip_algorithm="value")
        opt.step()

        # # single scheduler
        # sch = self.lr_schedulers()
        # sch.step()

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