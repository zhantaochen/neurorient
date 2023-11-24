import numpy as np
import torch
from torch.utils.data import Dataset
from .image_transform import RandomPatch, PhotonFluctuation, PoissonNoise, BeamStopMask

import warnings

class DictionaryDataset(Dataset):
    """ A minimal dataset class that takes a dictionary of tensors as input.
        Adapted from draft by ChatGPT GPT-4.
        Usage:
            x = torch.randn(100, 3, 32, 32)
            y = torch.randint(0, 2, (100,))
            dataset = DictionaryDataset(features=x, labels=y)
    """
    def __init__(self, **tensors):
        assert all(tensors[next(iter(tensors))].size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)


class TensorDatasetWithTransform(Dataset):
    def __init__(self, dataset, transform_list = None, seed = None):
        self.dataset        = dataset
        self.transform_list = transform_list
        self.seed           = seed
        self.eps = 1e-6

        return None


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        img = self.dataset[idx]    # (C, H, W)

        img_transformed = img.clone()
        input_mask = torch.ones_like(img)
        general_mask = torch.ones_like(img)
        
        if self.seed is not None:
            torch.manual_seed(self.seed + idx)
            np.random.seed(self.seed + idx)
        photon_flux_factor = torch.tensor([1.,])
        for i_transform, transform in enumerate(self.transform_list):
            if transform is None:
                continue
            
            if isinstance(transform, BeamStopMask):
                """ For BeamStopMask transforms, do NOT apply the transform to the image;
                    instead, we only take the mask, since the transform will be applied
                    during the model training.
                    The mask is multiplied to the general_mask.
                """
                if not self.transform_list[i_transform].return_mask:
                    warnings.warn("BeamStopMask transform should have return_mask=True, setting it to True.")
                    self.transform_list[i_transform].return_mask = True
                _, _mask = transform(img_transformed)
                general_mask = general_mask * _mask
            
            elif isinstance(transform, RandomPatch):
                """ For RandomPatch transforms, do NOT apply the transform to the image;
                    instead, we only take the mask, since the transform will be applied
                    during the model training.
                    The mask is multiplied to the input_mask.
                """
                if not self.transform_list[i_transform].return_mask:
                    warnings.warn("RandomPatch transform should have return_mask=True, setting it to True.")
                    self.transform_list[i_transform].return_mask = True
                _, _mask = transform(img_transformed)
                input_mask = input_mask * _mask
                
            elif isinstance(transform, PhotonFluctuation):
                if transform.return_mask:
                    img_transformed, photon_flux_factor = transform(img_transformed)
                    photon_flux_factor = torch.tensor([photon_flux_factor,])
                else:
                    transform.return_mask = True
                    img_transformed, photon_flux_factor = transform(img_transformed)
                    photon_flux_factor = torch.tensor([photon_flux_factor,])
                    transform.return_mask = False
                
            else:
                """ for non-RandomPatch and non-BeamStopMask transforms, apply the transform to the image,
                    and take mask or not depending on the return_mask attribute.
                """
                if transform.return_mask:
                    img_transformed, _ = transform(img_transformed)
                else:
                    img_transformed = transform(img_transformed)
                
        return {"image": img_transformed, "input_mask": input_mask, 'general_mask': general_mask, "photon_flux_factor": photon_flux_factor}    # (C, H, W)
