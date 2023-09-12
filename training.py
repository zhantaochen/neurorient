# %%
import torch
import matplotlib.pyplot as plt
from neurorient import NeurOrient

from neurorient.utils_model import get_radial_profile

# %%
pdb = '5BWK'
poisson = True
num_images = 10000
increase_factor = 100

spi_data = torch.load(
    f'/pscratch/sd/z/zhantao/neurorient_repo/data/{pdb}_increase{increase_factor}_poisson{poisson}_num{num_images//1000}K.pt')
model_dir = '/pscratch/sd/z/zhantao/neurorient_repo/model'
print(spi_data.keys())

# %%
q_values, radial_profile = get_radial_profile(
    spi_data['intensities'], 
    spi_data['pixel_position_reciprocal'])

# %%
radial_scale_configs = {
    "q_values": q_values,
    "radial_profile": radial_profile,
    "alpha": 1.0
}

# %%
model = NeurOrient(
    spi_data['pixel_position_reciprocal'], volume_type='intensity', path=model_dir, 
    loss_type='mse', radial_scale_configs=radial_scale_configs, lr=1e-3,
    photons_per_pulse=1e12 * increase_factor)
# model = NeurOrient.load_from_checkpoint('/pscratch/sd/z/zhantao/neurorient_repo/model/lightning_logs/version_14651494/checkpoints/1BXR-epoch=857-step=17150.ckpt')
# model.to('cpu');

# %%
import lightning as L

# %%
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# %%
dataset = TensorDataset(spi_data['intensities'].unsqueeze(1))
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

# %%
# batch = next(iter(dataloader))
# # model.training_step(batch, 0)
# slices_true = batch[0]
# orientations = model.image_to_orientation(slices_true)

# from neurorient.reconstruction.slicing import gen_nonuniform_normalized_positions
# # get reciprocal positions based on orientations
# HKL = gen_nonuniform_normalized_positions(
#     orientations, model.pixel_position_reciprocal, model.over_sampling)
# slices_pred = model.predict_slice(HKL)

# %%
# from neurorient.utils_visualization import display_images_in_parallel, display_volumes

# display_images_in_parallel(torch.randn(10, 3,3), torch.randn(10, 3,3))
# display_volumes(torch.randn(10,3,3,3))

# %%
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

checkpoint_callback = ModelCheckpoint(
    every_n_train_steps=10, save_last=True, save_top_k=1, monitor="train_loss",
    filename=f'{pdb}-{{epoch}}-{{step}}'
)

# %%
from lightning.pytorch.strategies import DDPStrategy

# Explicitly specify the process group backend if you choose to
# ddp = DDPStrategy(process_group_backend="nccl")

# %%
torch.set_float32_matmul_precision('high')
trainer = L.Trainer(
    max_epochs=1000, accelerator='gpu',
    callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
    log_every_n_steps=1, devices=torch.cuda.device_count(),
    enable_checkpointing=True, default_root_dir=model.path)
trainer.fit(model, dataloader)

# %%



