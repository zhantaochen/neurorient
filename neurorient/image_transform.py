import numpy as np
import torch
from sklearn.neighbors import KernelDensity

class RandomPatch:
    """ Randomly place num_patch patch with the size of size_y * size_x onto an image.
    """

    def __init__(self, 
                 num_patch, 
                 size_patch_min, 
                 size_patch_max=None,
                 return_mask = True):
        self.num_patch    = num_patch                   # ...Number of patches
        self.size_patch_min = size_patch_min                # ...Size of the patch in y dimension
        if size_patch_max < size_patch_min or size_patch_max is None:
            self.size_patch_max = size_patch_min
        else:
            self.size_patch_max = size_patch_max
        self.return_mask = return_mask                # ...Is it allowed to return a mask
        
        return None


    def __call__(self, img):
        # Get the size of the image...
        size_img_y, size_img_x = img.shape[-2:]

        # Construct a mask of ones with the same size of the image...
        mask = torch.ones_like(img)

        # Generate a number of random position...
        pos_y = torch.randint(low = 0, high = size_img_y, size = (self.num_patch,1))
        pos_x = torch.randint(low = 0, high = size_img_x, size = (self.num_patch,1))

        # Stack two column vectors to form an array of (x, y) indices...
        pos   = torch.hstack((pos_y, pos_x)).to(img.device)

        # Place patch of zeros at all pos as top-left corner...
        for (y, x) in pos:
            size_patch_y = torch.randint(low = self.size_patch_min, high = self.size_patch_max + 1, size = ()).to(img.device)
            size_patch_x = torch.randint(low = self.size_patch_min, high = self.size_patch_max + 1, size = ()).to(img.device)

            # Find the limit of the bottom/right-end of the patch...
            y_end = min(y + size_patch_y, size_img_y)
            x_end = min(x + size_patch_x, size_img_x)

            # Patch the area with zeros...
            mask[..., y : y_end, x : x_end] = 0

        # Apply the mask...
        output = mask * img
        if self.return_mask:
            return output, mask
        else:
            return output

class PhotonFluctuation:
    """ Add photon fluctuation to the image.
    """
    def __init__(self, stats_file='data/image_distribution_by_photon_count.npy', return_mask = True):
        image_distribution_by_photon_count = np.load(stats_file)
        p_counts, probabilities = image_distribution_by_photon_count
        p_counts_mean = (p_counts * probabilities).sum() / probabilities.sum()
        p_counts_relative = (p_counts / p_counts_mean).reshape(-1, 1)
        self.kde = KernelDensity(
                kernel='gaussian', 
                bandwidth=1.0*np.diff(p_counts_relative.squeeze()).mean()
            ).fit(p_counts_relative, sample_weight=probabilities)
        self.return_mask = return_mask
        return None

    def __call__(self, img):
        num_samples = img.shape[0]
        scale_factors = torch.from_numpy(self.kde.sample(num_samples).squeeze(-1)).to(img.device)
        output = torch.einsum('b..., b -> b...', img, scale_factors)
        if self.return_mask:
            return output, scale_factors
        else:
            return output
    
class PoissonNoise:
    """ Add Poisson noise to the image.
    """
    def __init__(self, return_mask = True):
        self.return_mask = return_mask
        return None

    def __call__(self, img):
        output = torch.poisson(img)
        if self.return_mask:
            return output, torch.ones_like(img)
        else:
            return output
        
class GaussianNoise:
    """ Add Gaussian noise (detector readout) to the image.
    """
    def __init__(self, sigma, return_mask = True):
        self.return_mask = return_mask
        self.sigma = sigma
        return None
    
    def __call__(self, img):
        noise = torch.randn_like(img) * self.sigma
        output = (img + noise).clamp(min=0)
        if self.return_mask:
            return output, torch.ones_like(img)
        else:
            return output
        
class BeamStopMask:
    """ Add beam stop mask to the image.
    """
    def __init__(self, width, radius, input_size, mask_orientation='v', return_mask = True):
        self.return_mask = return_mask
        self.width = width
        self.radius = radius
        self.mask_orientation = mask_orientation
        
        # Create meshgrid
        H, W = input_size
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        y, x = y.float(), x.float()

        # Create center mask
        center_y, center_x = H / 2, W / 2
        circle_mask = ((x - center_x)**2 + (y - center_y)**2) > self.radius**2

        # Create beam stop mask
        if self.mask_orientation == 'v':
            beam_mask = torch.logical_or(torch.abs(y - center_y) > self.width / 2, x > center_x)
        elif self.mask_orientation == 'h':
            beam_mask = torch.logical_or(torch.abs(x - center_x) > self.width / 2, y > center_y)
        else:
            raise ValueError("mask_orientation should be either 'h' or 'v'")

        # Combine masks
        self.base_mask = circle_mask * beam_mask
        
        return None
    
    def get_mask(self, img_shape):
        if len(img_shape) == 2:
            return self.base_mask
        else:
            mask = self.base_mask.reshape((1,) * (len(img_shape)-2) + self.base_mask.shape)
            return mask.repeat(*img_shape[:-2], 1, 1)
    
    def __call__(self, img):
        mask = self.get_mask(img.shape)
        output = img * mask
        if self.return_mask:
            return output, mask
        else:
            return output