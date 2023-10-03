import numpy as np
import torch
from torch.utils.data import Dataset
from .image_transform import RandomPatch, PhotonFluctuation, PoissonNoise, BeamStopMask

import warnings

class TensorDatasetWithTransform(Dataset):
    def __init__(self, dataset, transform_list = None):
        self.dataset        = dataset
        self.transform_list = transform_list

        self.eps = 1e-6

        return None


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        img = self.dataset[idx]    # (C, H, W)

        img_transformed = img.clone()
        input_mask = torch.ones_like(img)
        general_mask = torch.ones_like(img)
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
                
            else:
                """ for non-RandomPatch and non-BeamStopMask transforms, apply the transform to the image,
                    and take mask or not depending on the return_mask attribute.
                """
                if transform.return_mask:
                    img_transformed, _ = transform(img_transformed)
                else:
                    img_transformed = transform(img_transformed)
                
        return {"image": img_transformed, "input_mask": input_mask, 'general_mask': general_mask}    # (C, H, W)
