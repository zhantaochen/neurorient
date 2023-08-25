import torch
from torch import nn
import lightning as L
from torchkbnufft import KbNufft

from .predictors import I2S
from .external.siren_pytorch import SirenNet
from .reconstruction.slicing import get_real_mesh, gen_nonuniform_normalized_positions
from .utils_visualization import display_volumes, display_images_in_parallel

class NeurOrient(L.LightningModule):
    def __init__(self, pixel_position_reciprocal, volume_type='real_neural', over_sampling=1, path=None):
        super().__init__()
        
        self.register_buffer('pixel_position_reciprocal', 
                             pixel_position_reciprocal if isinstance(pixel_position_reciprocal, torch.Tensor) 
                                                       else torch.from_numpy(pixel_position_reciprocal)
                            )
        # only works for square images for now
        self.register_buffer('image_dimension', torch.tensor(self.pixel_position_reciprocal.shape[1]))
        self.register_buffer('grid_position_real',
                             get_real_mesh(self.image_dimension, self.pixel_position_reciprocal.max())
                            )
        self.over_sampling = over_sampling
        self.orientation_predictor = I2S(
            encoder='e2wrn',
            encoder_input_shape=(1, 128, 128),
            eval_wigners_file='/pscratch/sd/z/zhantao/neurorient_repo/data/eval_wigners_lmax6_rec5.pt'
            )
        
        self.volume_type = volume_type
        self.volume_predictor = SirenNet(
            dim_in=3,
            dim_hidden=256,
            dim_out=1,
            num_layers=5,
            final_activation=torch.nn.ReLU(),
        )
        self.nufft_forward = KbNufft(im_size=(self.image_dimension,)*3)
        self.path = path
    
    def image_to_orientation(self, x):
        rotmats = self.orientation_predictor.compute_average_rotmats(
            x, self.orientation_predictor.output_wigners)
        return rotmats
    
    def predict_rho(self, grid_position_real):
        if grid_position_real.ndim == 4:
            grid_position_real = grid_position_real.view(-1, 3)
        rho = self.volume_predictor(grid_position_real).view((self.image_dimension,)*3)
        return rho
    
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
        # predict orientations from images
        orientations = self.image_to_orientation(slices_true)
        # predict autocorrelation with orientations and predicted rho
        ac, rho = self.compute_autocorrelation(return_volume=True)
        # predict slices with orientations and predicted autocorrelation
        slices_pred = self.gen_slices(ac, orientations).unsqueeze(1)
        
        # scale slices_true and slices_pred
        slices_mags = slices_true.abs().amax(dim=(-2, -1), keepdim=True)
        slices_pred = slices_mags * slices_pred / slices_pred.abs().amax(dim=(-2, -1), keepdim=True)
        
        loss = nn.functional.mse_loss(slices_pred.cpu(), slices_true.cpu())
        self.log("train_loss", loss)
        
        display_volumes(rho, save_to=f'{self.path}/rho.png')
        display_images_in_parallel(slices_pred, slices_true, save_to=f'{self.path}/slices.png')
        
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
        return torch.optim.Adam(self.parameters(), lr=1e-3)