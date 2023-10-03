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
                 returns_mask = False):
        self.num_patch    = num_patch                   # ...Number of patches
        self.size_patch_min = size_patch_min                # ...Size of the patch in y dimension
        if size_patch_max < size_patch_min or size_patch_max is None:
            self.size_patch_max = size_patch_min
        else:
            self.size_patch_max = size_patch_max
        self.returns_mask = returns_mask                # ...Is it allowed to return a mask
        
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
        img_masked = mask * img

        # Construct the return value...
        # Parentheses are necessary
        output = img_masked if not self.returns_mask else (img_masked, mask)

        return output

class PhotonFluctuation:
    """ Add photon fluctuation to the image.
    """
    def __init__(self, stats_file='data/image_distribution_by_photon_count.npy'):
        image_distribution_by_photon_count = np.load(stats_file)
        p_counts, probabilities = image_distribution_by_photon_count
        p_counts_mean = (p_counts * probabilities).sum() / probabilities.sum()
        p_counts_relative = (p_counts / p_counts_mean).reshape(-1, 1)
        self.kde = KernelDensity(
                kernel='gaussian', 
                bandwidth=1.0*np.diff(p_counts_relative.squeeze()).mean()
            ).fit(p_counts_relative, sample_weight=probabilities)
        return None

    def __call__(self, img):
        num_samples = img.shape[0]
        scale_factors = torch.from_numpy(self.kde.sample(num_samples).squeeze()).to(img.device)
        output = torch.einsum('b...,b -> b...', img, scale_factors)
        return output