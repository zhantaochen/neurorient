import torch
from torch import nn, optim

import lightning as L
import numpy as np
from torchkbnufft import KbNufft

from .predictors import I2S, ResNet2Rotmat
from .external.siren_pytorch import SirenNet
from .reconstruction.slicing import get_real_mesh, get_reciprocal_mesh, gen_nonuniform_normalized_positions
from .utils_visualization import display_volumes, display_images_in_parallel

from .utils_model import get_radial_scale_mask

class IntensityNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # self.net_exp = SirenNet(*args, **kwargs, final_activation=torch.nn.Identity())
        self.net_mag = SirenNet(*args, **kwargs, final_activation=torch.nn.SiLU())
        # self.net_mag = SirenNet(*args, **kwargs)
    
    def forward(self, x):
        return self.net_mag(x) + self.net_mag(-x)
    
    # def forward(self, x):
    #     exp = (self.net_exp(x) + self.net_exp(-x)).exp()
    #     mag = self.net_mag(x) + self.net_mag(-x)
    #     return mag * torch.nan_to_num(exp.exp())


class NeurOrient(L.LightningModule):
    def __init__(self, 
                 pixel_position_reciprocal, 
                 volume_type='real_neural', 
                 over_sampling=1, 
                 loss_type='poisson', 
                 radial_scale_configs=None,
                 lr=1e-3, 
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
        # self.orientation_predictor = I2S(
        #     encoder='e2wrn',
        #     encoder_input_shape=(1, 128, 128),
        #     eval_wigners_file='/pscratch/sd/z/zhantao/neurorient_repo/data/eval_wigners_lmax6_rec5.pt'
        #     )
        self.orientation_predictor = ResNet2Rotmat(
            size=18, pretrained=True, pool_features=False
        )
        
        self.volume_type = volume_type
        if self.volume_type.find('intensity') >= -1:
            self.volume_predictor = IntensityNet(
                dim_in=3,
                dim_hidden=256,
                dim_out=1,
                num_layers=5,
            )
        else:
            self.volume_predictor = SirenNet(
                dim_in=3,
                dim_hidden=512,
                dim_out=1,
                num_layers=5,
                final_activation=torch.nn.ReLU(),
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
        
        self.loss_type = loss_type
        if loss_type == 'poisson':
            # self.volume_predictor.net_mag.last_layer.activation = torch.nn.Identity()
            self.loss_func = nn.PoissonNLLLoss(log_input=True)
        elif loss_type == 'mse':
            self.loss_func = nn.MSELoss()
        
        self.lr = lr
        self.path = path
        
            
    def image_to_orientation(self, x):
        if isinstance(self.orientation_predictor, I2S):
            rotmats = self.orientation_predictor.compute_average_rotmats(
                x, self.orientation_predictor.output_wigners)
        elif isinstance(self.orientation_predictor, ResNet2Rotmat):
            rotmats = self.orientation_predictor(x)
        return rotmats
    
    def predict_rho(self, grid_position_real):
        if grid_position_real.ndim == 4:
            grid_position_real = grid_position_real.view(-1, 3)
        rho = self.volume_predictor(grid_position_real).view((self.image_dimension,)*3)
        return rho
    
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
    
    def compute_autocorrelation(self, rho=None, pred_from_scratch=True, return_volume=False):
        """
        if rho is provided, always use it, pred_from_scratch is ignored
        if rho is not provided, compute rho from scratch depending on pred_from_scratch
        """
        if rho is None:
            if pred_from_scratch:
                rho = self.predict_rho(self.grid_position_real)
        else:
            rho.to(self.device)
        intensity = torch.fft.fftshift(torch.fft.fftn(rho).abs().pow(2))
        ac = torch.fft.ifftshift(torch.fft.ifftn(intensity).abs())
        if return_volume:
            return ac, rho
        else:
            return ac
        
    def compute_slices(self, orientations, rho=None):
        self.compute_autocorrelation(rho=rho)
        model_slices = self.gen_slices(
            self.nufft_forward, self.ac, orientations, self.pixel_position_reciprocal)
        return model_slices
    
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
        
        # slices_input = torch.log(slices_true + torch.finfo(slices_true.dtype).eps)
        # slices_input = torch.log(slices_true * 10 + 1)
        slices_input = torch.log(slices_true * 10 + 1.)

        # predict orientations from images
        orientations = self.image_to_orientation(slices_input)
        # get reciprocal positions based on orientations
        # HKL has shape (3, num_qpts)
        HKL = gen_nonuniform_normalized_positions(
            orientations, self.pixel_position_reciprocal, self.over_sampling)
        # predict slices from HKL
        slices_pred = self.predict_slice(HKL).view((-1, 1,) + (self.image_dimension,)*2)
        
        # # predict autocorrelation with orientations and predicted rho
        # ac, rho = self.compute_autocorrelation(return_volume=True)
        # # predict slices with orientations and predicted autocorrelation
        # slices_pred = self.gen_slices(ac, orientations).unsqueeze(1)
        
        # scale slices_true and slices_pred
        # slices_mags = slices_true.abs().amax(dim=(-2, -1), keepdim=True)
        # slices_pred = slices_mags * slices_pred / (slices_pred.abs().amax(dim=(-2, -1), keepdim=True) + torch.finfo(self.dtype).eps)
        
        # loss = nn.functional.mse_loss(slices_pred.cpu(), slices_input.cpu())
        if self.loss_type == 'poisson':
            loss = self.loss_func(slices_pred.cpu(), slices_true.cpu())
        elif self.loss_type == 'mse':
            loss = self.loss_func(slices_pred.cpu(), slices_input.cpu())
        self.log("train_loss", loss)
        
        # display_volumes(rho, save_to=f'{self.path}/rho.png')
        if self.global_step % 10 == 0:
            display_images_in_parallel((torch.exp(slices_pred) - 1) / 10, slices_true, save_to=f'{self.path}/slices_version_{self.logger.version}.png')
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        slices_true = batch[0].to(self.device)
        # predict orientations from images
        orientations = self.image_to_orientation(slices_true)
        # predict autocorrelation with orientations and predicted rho
        ac, rho = self.compute_autocorrelation(return_volume=True)
        # predict slices with orientations and predicted autocorrelation
        slices_pred = self.gen_slices(ac, orientations)
        
        # scale slices_true and slices_pred
        slices_mags = slices_true.abs().amax(dim=(-2, -1), keepdim=True)
        slices_pred = slices_mags * slices_pred / slices_pred.abs().amax(dim=(-2, -1), keepdim=True)
        
        loss = nn.functional.mse_loss(slices_pred.cpu(), slices_true.cpu())
        self.log("valid_loss", loss)
        
        display_volumes(rho, save_to=f'{self.path}/rho.png')
        display_images_in_parallel(slices_pred, slices_true, save_to=f'{self.path}/slices.png')
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        # scheculer = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1e-4 + 9e-4 * np.exp(- epoch / 25))
        # return [optimizer], [scheculer]