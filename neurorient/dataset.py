import numpy as np

from torch.utils.data import Dataset


class TensorDatasetWithTransform(Dataset):
    def __init__(self, dataset, transform_list = None, uses_norm = True):
        self.dataset        = dataset
        self.uses_norm      = uses_norm
        self.transform_list = transform_list

        self.eps = 1e-6

        return None


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        img = self.dataset[idx]    # (C, H, W)

        img_transformed = img.copy()
        for transform in self.transform_list:
            if transform is not None:
                img_transformed = transform(img_transformed)

        if self.uses_norm:
            img_transformed_mean = np.nanmean(img_transformed)
            img_transformed_std  = np.nanstd(img_transformed)

            img_transformed = (img_transformed - img_transformed_mean) / img_transformed_std \
                  if img_transformed_std > self.eps else \
                  np.random.rand(*img_transformed.shape)

        return (img_transformed, img)    # (C, H, W)
