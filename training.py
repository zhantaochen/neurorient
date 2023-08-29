# %%
import torch
import matplotlib.pyplot as plt
from neurorient import NeurOrient

# %%
from mpi4py import MPI

# %%
spi_data = torch.load('/pscratch/sd/z/zhantao/neurorient_repo/data/1BXR_increase10_poissonFalse_num10K.pt')
model_dir = '/pscratch/sd/z/zhantao/neurorient_repo/model'
print(spi_data.keys())

# %%
# model = NeurOrient.load_from_checkpoint('/pscratch/sd/z/zhantao/neurorient_repo/model/lightning_logs/version_2/checkpoints/last.ckpt')

# %%
model = NeurOrient(spi_data['pixel_position_reciprocal'], volume_type='intensity', path=model_dir)
# model.to('cpu');

# %%
# model.compute_autocorrelation()
# orientations = model.image_to_orientation(spi_data['images'][:10].unsqueeze(1))
# from neurorient.reconstruction.slicing import get_rho_function
# rho_true = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(spi_data['volume']))).real.clip(0)
# rho_func = get_rho_function(spi_data['vol_real_mesh'], rho_true)
# rho_img_coords = torch.from_numpy(rho_func(spi_data['img_real_mesh'].numpy()))
# rho_true = torch.from_numpy(rho_func(model.grid_position_real.numpy()))
# ac = model.compute_autocorrelation(rho=rho_true)
# slices = model.gen_slices(ac, orientations)
# plt.imshow(slices.detach().numpy()[0].clip(0), vmax=slices.max() * 0.001)

# %%
import lightning as L

# %%
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# %%
dataset = TensorDataset(spi_data['intensities'].unsqueeze(1))
dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

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
from neurorient.utils_visualization import display_images_in_parallel, display_volumes

# display_images_in_parallel(torch.randn(10, 3,3), torch.randn(10, 3,3))
# display_volumes(torch.randn(10,3,3,3))

# %%
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

checkpoint_callback = ModelCheckpoint(
    every_n_train_steps=10, save_last=True, save_top_k=1, monitor="train_loss"
)

# %%
trainer = L.Trainer(
    max_epochs=1000, accelerator='gpu',
    callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
    log_every_n_steps=1, devices=torch.cuda.device_count(),
    enable_checkpointing=True, default_root_dir=model.path)
trainer.fit(model, dataloader)


