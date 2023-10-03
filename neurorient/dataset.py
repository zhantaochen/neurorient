import numpy as np

from torch.utils.data import Dataset


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
        for transform in self.transform_list:
            if transform is not None:
                img_transformed = transform(img_transformed)

        return (img_transformed, img)    # (C, H, W)
