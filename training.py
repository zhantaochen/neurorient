# %%

import torch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from neurorient import NeurOrient

from neurorient.utils_model import get_radial_profile

# %%
pdb = '1BXR'
poisson = True
num_images = 10000
increase_factor = 10

spi_data = torch.load(
    f'/pscratch/sd/z/zhantao/neurorient_repo/data/{pdb}_increase{increase_factor}_poisson{poisson}_num{num_images//1000}K.pt')
model_dir = '/pscratch/sd/z/zhantao/neurorient_repo/model'
print(spi_data.keys())

# %%
q_values, radial_profile = get_radial_profile(
    spi_data['intensities'][:1000], 
    spi_data['pixel_position_reciprocal'])

# %%
radial_scale_configs = {
    "q_values": q_values,
    "radial_profile": radial_profile,
    "alpha": 1.0
}

# %%
model = NeurOrient(
    spi_data['pixel_position_reciprocal'], path=model_dir, 
    radial_scale_configs=radial_scale_configs, lr=1e-3,
    photons_per_pulse=1e12 * increase_factor)

# %%
import lightning as L

# %%
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# %%
dataset = TensorDataset(spi_data['intensities'].unsqueeze(1))

train_idx, val_test_idx = train_test_split(np.arange(len(dataset)), test_size=1/10, random_state=42)
val_idx, test_idx = train_test_split(val_test_idx, test_size=1/2, random_state=42)

train_dataloader = DataLoader([dataset[i] for i in train_idx], batch_size=100, shuffle=True)
val_dataloader = DataLoader([dataset[i] for i in val_idx], batch_size=100, shuffle=False)

# %%
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

checkpoint_callback = ModelCheckpoint(
    every_n_train_steps=10, save_last=True, save_top_k=1, monitor="val_loss",
    filename=f'{pdb}-{{epoch}}-{{step}}'
)

# %%
torch.set_float32_matmul_precision('high')
trainer = L.Trainer(
    max_epochs=1000, accelerator='gpu',
    callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
    log_every_n_steps=1, devices=torch.cuda.device_count(),
    enable_checkpointing=True, default_root_dir=model.path)
trainer.fit(model, train_dataloader, val_dataloader)

# %%



