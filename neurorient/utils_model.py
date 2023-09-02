import scipy
from scipy.interpolate import interp1d

import numpy as np
import torch
from torch_scatter import scatter_mean

def get_model_size(model):
    
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    return size_all_mb

def get_radial_profile(images, pixel_positions=None, reduce_mean=True, decimals=3):
    """
    Compute the radial profile (circular average) of a 2D image using pixel_positions.
    
    images: torch.Tensor, shape (N, H, W)
    pixel_positions: torch.Tensor, shape (H, W, 3)
    
    """
    
    if pixel_positions is None:
        # If center is not provided, assume center of the image
        center = torch.tensor([(x-1)/2.0 for x in images.size()[-2:]])
        
        # Calculate the indices of the grid
        y, x = torch.meshgrid(torch.arange(0, images.size()[-2], dtype=torch.float32, device=images.device),
                              torch.arange(0, images.size()[-1], dtype=torch.float32, device=images.device))
        
        # Calculate q values for each pixel
        q = torch.sqrt((x - center[1])**2 + (y - center[0])**2)
        q = torch.round(q, decimals=decimals)  # Convert the radii to integers
    else:
        pixel_positions = pixel_positions / pixel_positions.max() * np.pi
        q = torch.round(pixel_positions.norm(dim=-1), decimals=decimals)
    
    q_unique, inv_map = torch.unique(q, sorted=True, return_inverse=True)
    Iq = scatter_mean(images.view(images.shape[0], -1), inv_map.view(-1))
    if reduce_mean:
        return q_unique, Iq.mean(dim=0)
    else:
        return q_unique, Iq

def get_radial_scale_mask(q_values, radial_profile, pixel_positions, alpha=0.0):
    alpha = max(alpha, 0.0)
    radial_profile = radial_profile * torch.exp(alpha * q_values)
    radial_scale = (radial_profile.max() / (radial_profile + 1e-6)).clamp_max(1e6)
    f_interp = interp1d(q_values, radial_scale, kind='linear', bounds_error=False, fill_value='extrapolate')
    norm_pixel_positions = (np.pi * (pixel_positions / pixel_positions.max())).norm(dim=-1)
    radial_scale_factor = f_interp(norm_pixel_positions.detach().cpu().numpy())
    return radial_scale_factor

def get_radial_scale_mask_from_images(images, pixel_positions, decimals=3, alpha=0.0):
    q_values, radial_profile = get_radial_profile(images, pixel_positions, reduce_mean=True, decimals=decimals)
    radial_scale_factor = get_radial_scale_mask(q_values, radial_profile, pixel_positions, alpha=alpha)
    return radial_scale_factor

def fit_radial_profile(images, pixel_positions=None):
    q, radial_profile = get_radial_profile(images, pixel_positions, reduce_mean=True)
    
    def fit_func(q, a, b):
        return a * np.exp(-b * q**4)
    
    mu, covar = scipy.optimize.curve_fit(fit_func, q.detach().cpu().numpy(), radial_profile.detach().cpu().numpy())
    
    return mu