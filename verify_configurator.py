# %%
"""
This notebook is used to verify the configurator.py file.
"""

# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import yaml
import logging
import socket
import tqdm
import signal
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
## from torch.optim.lr_scheduler import ReduceLROnPlateau

# Libraries used for Distributed Data Parallel (DDP)
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from neurorient.model     import NeurOrientLightning
from neurorient.utils_model     import get_radial_profile
from neurorient.data            import TensorDatasetWithTransform
from neurorient.image_transform import RandomPatch
from neurorient.utils           import init_logger, MetaLog, split_dataset, save_checkpoint, load_checkpoint, set_seed, init_weights
from neurorient.configurator    import Configurator
from neurorient.lr_scheduler    import CosineLRScheduler
from neurorient.config          import _CONFIG
from neurorient.utils_config    import prepare_Slice2RotMat_BIFPN_inputs

torch.autograd.set_detect_anomaly(False)    # [WARNING] Making it True may throw errors when using bfloat16
                                            # Reference: https://discuss.pytorch.org/t/convolutionbackward0-returned-nan-values-in-its-0th-output/175571/4

logger = logging.getLogger(__name__)

# %%
# [[[ ARG ]]]
# parser = argparse.ArgumentParser(description="Load training configuration from a YAML file to a dictionary.")
# parser.add_argument("yaml_file", help="Path to the YAML file")

# args = parser.parse_args()

args = argparse.Namespace(yaml_file='/global/homes/z/zhantao/Projects/NeuralOrientationMatching/base_config.1BXR.test.yaml')

# %%
# [[[ HYPER-PARAMERTERS ]]]
# Load CONFIG from YAML
fl_yaml = args.yaml_file
with open(fl_yaml, 'r') as fh:
    config_dict = yaml.safe_load(fh)
CONFIG = Configurator.from_dict(config_dict)

# ...Checkpoint
timestamp_prev      = CONFIG.CHKPT.TIMESTAMP_PREV
epoch_prev          = CONFIG.CHKPT.EPOCH_PREV
dir_chkpt           = Path(os.path.join(CONFIG.MISC.BASE_DIRECTORY, CONFIG.CHKPT.DIRECTORY))
dir_chkpt.mkdir(parents=True, exist_ok=True)
fl_chkpt_prefix     = CONFIG.CHKPT.FILENAME_PREFIX

# ...Dataset
dir_dataset       = CONFIG.DATASET.DIRECTORY
pdb               = CONFIG.DATASET.PDB
poisson           = CONFIG.DATASET.POISSON
increase_factor   = CONFIG.DATASET.INCREASE_FACTOR
num_images        = CONFIG.DATASET.NUM_IMG
frac_train        = CONFIG.DATASET.FRAC_TRAIN
size_batch        = CONFIG.DATASET.BATCH_SIZE
num_workers       = CONFIG.DATASET.NUM_WORKERS
uses_random_patch = CONFIG.DATASET.USES_RANDOM_PATCH

# ...Model
num_bifpn_blocks    = CONFIG.MODEL.BIFPN.NUM_BLOCKS
num_bifpn_features  = CONFIG.MODEL.BIFPN.NUM_FEATURES
freezes_backbone    = CONFIG.MODEL.FREEZES_BACKBONE
uses_random_weights = CONFIG.MODEL.USES_RANDOM_WEIGHTS

# ...Optimizer
lr           = float(CONFIG.OPTIM.LR)
weight_decay = float(CONFIG.OPTIM.WEIGHT_DECAY)
grad_clip    = float(CONFIG.OPTIM.GRAD_CLIP)

# ...Loss
loss_scale_factor = CONFIG.LOSS.SCALE_FACTOR

# ...Scheduler
## patience = CONFIG.LR_SCHEDULER.PATIENCE
warmup_epochs = CONFIG.LR_SCHEDULER.WARMUP_EPOCHS
total_epochs  = CONFIG.LR_SCHEDULER.TOTAL_EPOCHS
min_lr        = float(CONFIG.LR_SCHEDULER.MIN_LR)

# ...DDP
ddp_backend            = CONFIG.DDP.BACKEND
uses_unique_world_seed = CONFIG.DDP.USES_UNIQUE_WORLD_SEED

# ...Logging
dir_log       = Path(os.path.join(CONFIG.MISC.BASE_DIRECTORY, CONFIG.LOGGING.DIRECTORY))
dir_log.mkdir(parents=True, exist_ok=True)
fl_log_prefix = CONFIG.LOGGING.FILENAME_PREFIX

# ...Misc
uses_mixed_precision = CONFIG.MISC.USES_MIXED_PRECISION
max_epochs           = CONFIG.MISC.MAX_EPOCHS
num_gpus             = CONFIG.MISC.NUM_GPUS
dir_base             = CONFIG.MISC.BASE_DIRECTORY

# %%
merged_config = CONFIG.merge_with_priority(_CONFIG, self_has_priority=True)

# %%
uses_ddp = True

# [[[ DATASET ]]]
spi_data = torch.load(os.path.join(dir_dataset, f'{pdb}_increase{increase_factor}_poisson{poisson}_num{num_images//1000}K.pt'))

# Set global seed and split data...
# set_seed(base_seed)
data              = spi_data['intensities']
spi_data_train    = data[:int(len(data) * frac_train) ]
spi_data_validate = data[ int(len(data) * frac_train):]

# Set world seed and set up transformation rules
# set_seed(world_seed)
num_patch    = 200
size_patch_y = 5
size_patch_x = 5
var_patch_y  = 0.2
var_patch_x  = 0.2
returns_mask = False
random_patch = RandomPatch(num_patch    = num_patch,
                           size_patch_y = size_patch_y,
                           size_patch_x = size_patch_x,
                           var_patch_y  = var_patch_y,
                           var_patch_x  = var_patch_x,
                           returns_mask = returns_mask)
random_patch = None if uses_random_patch else random_patch    # Janky inline workaround to turn off random patching
transform_list = ( random_patch, )

dataset_train    = TensorDatasetWithTransform(spi_data_train.unsqueeze(1).numpy(), transform_list = transform_list, uses_norm = False)
# sampler_train    = torch.utils.data.DistributedSampler(dataset_train) if uses_ddp else None
sampler_train    = None
dataloader_train = torch.utils.data.DataLoader( dataset_train,
                                                sampler     = sampler_train,
                                                shuffle     = False,
                                                pin_memory  = True,
                                                batch_size  = size_batch,
                                                num_workers = num_workers, )

dataset_validate    = TensorDatasetWithTransform(spi_data_validate.unsqueeze(1).numpy(), transform_list = transform_list, uses_norm = False)
# sampler_validate    = torch.utils.data.DistributedSampler(dataset_validate, shuffle=False) if uses_ddp else None
sampler_validate    = None
dataloader_validate = torch.utils.data.DataLoader( dataset_validate,
                                                   sampler     = sampler_validate,
                                                   shuffle     = False,
                                                   pin_memory  = True,
                                                   batch_size  = size_batch,
                                                   num_workers = num_workers, )

# %%
# [[[ MODEL ]]]

config_slice2rotmat = prepare_Slice2RotMat_BIFPN_inputs(merged_config, use_bifpn=merged_config.MODEL.USE_BIFPN)
model = NeurOrientLightning(
    spi_data['pixel_position_reciprocal'],
    over_sampling=1, photons_per_pulse=1e13,
    use_bifpn=merged_config.MODEL.USE_BIFPN,
    config_slice2rotmat=config_slice2rotmat,
    config_optimization={'lr': lr, 'weight_decay': weight_decay, 'loss_func': 'MSELoss'}
)

# %%
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

# %%
checkpoint_callback = ModelCheckpoint(
    every_n_train_steps=10, save_last=True, save_top_k=1, monitor="val_loss",
    filename=f'{pdb}-{{epoch}}-{{step}}'
)

# %%
torch.set_float32_matmul_precision('high')
trainer = L.Trainer(
    max_epochs=500, accelerator='gpu',
    callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
    log_every_n_steps=1, devices=3,
    enable_checkpointing=True, default_root_dir=dir_chkpt)
trainer.fit(model, dataloader_train, dataloader_validate)

# %%

# if ddp_rank == 0:
#     print(f"{sum(p.numel() for p in model.parameters())/1e6} M pamameters.")

# # Initialized by the main rank and weights will be broadcast by DDP wrapper
# # if ddp_rank == 0:
# #     if uses_random_weights:
# #         # Use random weights...
# #         model.apply(init_weights)
# #     else:
# #         # [TODO]
# #         pass

# # Freeze the backbone???
# if freezes_backbone:
#     for param in model.backbone.parameters():
#         param.requires_grad = False

# model.float()

# if uses_ddp:
#     # Convert BatchNorm to SyncBatchNorm...
#     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

#     # Wrap it up using DDP...
#     model = DDP(model, device_ids = [ddp_local_rank], find_unused_parameters=True)


# [[[ CRITERION ]]]
# criterion = nn.MSELoss()

# [[[ OPTIMIZER ]]]
# param_iter = model.module.parameters() if hasattr(model, "module") else model.parameters()
# optimizer = optim.AdamW(param_iter,
#                         lr = lr,
#                         weight_decay = weight_decay)
# scheduler = CosineLRScheduler(optimizer     = optimizer,
#                               warmup_epochs = warmup_epochs,
#                               total_epochs  = total_epochs,
#                               min_lr        = min_lr)
## scheduler = ReduceLROnPlateau(optimizer, mode           = 'min',
##                                          factor         = 2e-1,
##                                          patience       = patience,
##                                          threshold      = 1e-4,
##                                          threshold_mode ='rel',
##                                          verbose        = True)

# %%
# # [[[ TRAIN LOOP ]]]
# # From a prev training???
# epoch_min = 0
# loss_min  = float('inf')
# if path_chkpt_prev is not None:
#     epoch_min, loss_min = load_checkpoint(model, optimizer, scheduler, path_chkpt_prev)
#     ## epoch_min, loss_min = load_checkpoint(model, None, None, path_chkpt_prev)
#     epoch_min += 1    # Next epoch
#     logger.info(f"PREV - epoch_min = {epoch_min}, loss_min = {loss_min}")

# if ddp_rank == 0:
#     print(f"Current timestamp: {timestamp}")

# try:
#     chkpt_saving_period = 5
#     epoch_unstable_end  = 40
#     for epoch in tqdm.tqdm(range(max_epochs)):
#         epoch += epoch_min

#         if uses_ddp:
#             # Shuffle the training examples...
#             sampler_train.set_epoch(epoch)

#         # Uses mixed precision???
#         if uses_mixed_precision: scaler = torch.cuda.amp.GradScaler()

#         # ___/ TRAIN \___
#         # Turn on training related components in the model...
#         model.train()

#         # Fetch batches...
#         batch_train  = tqdm.tqdm(enumerate(dataloader_train), total = len(dataloader_train))
#         train_loss   = torch.zeros(len(batch_train)).to(device).float()
#         train_sample = torch.zeros(len(batch_train)).to(device).float()
#         for batch_idx, batch_entry in batch_train:
#             # Unpack the batch entry and move them to device...
#             batch_input, batch_target = batch_entry
#             batch_input  = batch_input.to(device, dtype = torch.float)
#             batch_target = batch_target.to(device, dtype = torch.float)

#             # Use log scale...
#             batch_target = batch_target * loss_scale_factor + 1.
#             batch_target = torch.log(batch_target)

#             # Forward, backward and update...
#             if uses_mixed_precision:
#                 with torch.cuda.amp.autocast(dtype = torch.float16):
#                     # Forward pass...
#                     batch_output = model(batch_input)

#                     # Calculate the loss...
#                     loss = criterion(batch_output, batch_target)
#                     loss = loss.mean()

#                 # Backward pass and optimization...
#                 optimizer.zero_grad()
#                 scaler.scale(loss).backward()
#                 if grad_clip != 0.0:
#                     scaler.unscale_(optimizer)
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 # Forward pass...
#                 batch_output = model(batch_input)

#                 # Calculate the loss...
#                 loss = criterion(batch_output, batch_target)
#                 loss = loss.mean()

#                 # Backward pass and optimization...
#                 optimizer.zero_grad()
#                 loss.backward()
#                 if grad_clip != 0.0:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
#                 optimizer.step()

#             # Reporting...
#             train_loss  [batch_idx] = loss
#             train_sample[batch_idx] = len(batch_input)

#         # Calculate the wegihted mean...
#         train_loss_sum   = torch.dot(train_loss, train_sample)
#         train_sample_sum = train_sample.sum()

#         if uses_ddp:
#             # Gather training metrics
#             world_train_loss_sum   = [ torch.tensor(0.0).to(device).float() for _ in range(ddp_world_size) ]
#             world_train_sample_sum = [ torch.tensor(0.0).to(device).float() for _ in range(ddp_world_size) ]
#             dist.all_gather(world_train_loss_sum  , train_loss_sum)
#             dist.all_gather(world_train_sample_sum, train_sample_sum)

#             world_train_loss_mean = torch.tensor(world_train_loss_sum).sum() / torch.tensor(world_train_sample_sum).sum()
#         else:
#             world_train_loss_mean = train_loss_sum / train_sample_sum

#         if ddp_rank == 0:
#             logger.info(f"MSG (device:{device}) - epoch {epoch}, mean train loss = {world_train_loss_mean:.8f}")


#         # ___/ VALIDATE \___
#         model.eval()

#         # Fetch batches...
#         batch_validate  = tqdm.tqdm(enumerate(dataloader_validate), total = len(dataloader_validate))
#         validate_loss   = torch.zeros(len(batch_validate)).to(device).float()
#         validate_sample = torch.zeros(len(batch_validate)).to(device).float()
#         for batch_idx, batch_entry in batch_validate:
#             """
#             Work on any preprocessing when necessary
#             """
#             # Unpack the batch entry and move them to device...
#             batch_input, batch_target = batch_entry
#             batch_input  = batch_input.to(device, dtype = torch.float)
#             batch_target = batch_target.to(device, dtype = torch.float)

#             # Use log scale...
#             batch_target = batch_target * loss_scale_factor + 1.
#             batch_target = torch.log(batch_target)

#             # Forward only...
#             with torch.no_grad():
#                 if uses_mixed_precision:
#                     with torch.cuda.amp.autocast(dtype = torch.float16):
#                         # Forward pass...
#                         batch_output = model(batch_input)

#                         # Calculate the loss...
#                         loss = criterion(batch_output, batch_target)
#                         loss = loss.mean()
#                 else:
#                     # Forward pass...
#                     batch_output = model(batch_input)

#                     # Calculate the loss...
#                     loss = criterion(batch_output, batch_target)
#                     loss = loss.mean()

#             # Reporting...
#             validate_loss  [batch_idx] = loss
#             validate_sample[batch_idx] = len(batch_input)

#         # Calculate the wegihted mean...
#         validate_loss_sum   = torch.dot(validate_loss, validate_sample)
#         validate_sample_sum = validate_sample.sum()

#         if uses_ddp:
#             # Gather training metrics
#             world_validate_loss_sum   = [ torch.tensor(0.0).to(device).float() for _ in range(ddp_world_size) ]
#             world_validate_sample_sum = [ torch.tensor(0.0).to(device).float() for _ in range(ddp_world_size) ]
#             dist.all_gather(world_validate_loss_sum  , validate_loss_sum)
#             dist.all_gather(world_validate_sample_sum, validate_sample_sum)

#             world_validate_loss_mean = torch.tensor(world_validate_loss_sum).sum() / torch.tensor(world_validate_sample_sum).sum() \
#                                        if len(spi_data_validate) else                                                              \
#                                        world_train_loss_mean
#         else:
#             world_validate_loss_mean = validate_loss_sum / validate_sample_sum \
#                                        if len(spi_data_validate) else          \
#                                        world_train_loss_mean

#         if ddp_rank == 0:
#             logger.info(f"MSG (device:{device}) - epoch {epoch}, mean val   loss = {world_validate_loss_mean:.8f}")

#             # Report the learning rate used in the last optimization...
#             lr_used = optimizer.param_groups[0]['lr']
#             logger.info(f"MSG (device:{device}) - epoch {epoch}, lr used = {lr_used:.8f}")

#         # Update learning rate in the scheduler...
#         scheduler.step()


#         # ___/ SAVE CHECKPOINT??? \___
#         if ddp_rank == 0:
#             if world_validate_loss_mean < loss_min:
#                 loss_min = world_validate_loss_mean

#                 if (epoch % chkpt_saving_period == 0) or (epoch > epoch_unstable_end):
#                     fl_chkpt = f"{timestamp}.epoch_{epoch}.chkpt"
#                     if fl_chkpt_prefix is not None: fl_chkpt = f"{fl_chkpt_prefix}.{fl_chkpt}"
#                     path_chkpt = os.path.join(dir_chkpt, fl_chkpt)
#                     save_checkpoint(model, optimizer, scheduler, epoch, loss_min, path_chkpt)
#                     logger.info(f"MSG (device:{device}) - save {path_chkpt}")

#         if uses_ddp: dist.barrier()

# except KeyboardInterrupt:
#     print(f"DDP RANK {ddp_rank}: Training was interrupted!")
# except Exception as e:
#     print(f"DDP RANK {ddp_rank}: Error occurred: {e}")
# finally:
#     # Ensure that the process group is always destroyed
#     if dist.is_initialized():
#         dist.destroy_process_group()


# %%



