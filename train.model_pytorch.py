#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import socket
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from neurorient.model_pytorch   import NeurOrient
from neurorient.utils_model     import get_radial_profile
from neurorient.data            import TensorDatasetWithTransform
from neurorient.image_transform import RandomPatch
from neurorient.utils           import init_logger, MetaLog, split_dataset, save_checkpoint, load_checkpoint, set_seed, init_weights

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

# [[[ USER INPUT ]]]
timestamp_prev = None # "2023_0825_2111_46"
epoch          = None # 57

drc_chkpt = "chkpts"
fl_chkpt_prev   = None if timestamp_prev is None else f"{timestamp_prev}.epoch_{epoch}.chkpt"
path_chkpt_prev = None if fl_chkpt_prev is None else os.path.join(drc_chkpt, fl_chkpt_prev)

lr             = 10**(-3.0)
weight_decay   = 1e-4
grad_clip      = 1.0

compiles_model       = False
uses_mixed_precision = False

num_gpu     = 1
size_batch  = 50  * num_gpu
num_workers = 10  * num_gpu    # mutiple of size_sample // size_batch
seed        = 0

timestamp = init_logger(returns_timestamp = True)


# [[[ DATASET ]]]
# Load the source data...
pdb             = '1BXR'
poisson         = True
num_images      = 10000
increase_factor = 10

spi_data = torch.load(f'data/{pdb}_increase{increase_factor}_poisson{poisson}_num{num_images//1000}K.pt')

# Set global seed...
set_seed(seed)

# Set up transformation rules
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
transform_list = ( random_patch, )

train_frac        = 0.7
data              = spi_data['intensities']
spi_data_train    = data[:int(len(data) * train_frac) ]
spi_data_validate = data[ int(len(data) * train_frac):]
dataset_train     = TensorDatasetWithTransform(spi_data_train.unsqueeze(1).numpy(), transform_list = transform_list)
dataloader_train  = torch.utils.data.DataLoader( dataset_train,
                                                 shuffle     = False,
                                                 pin_memory  = True,
                                                 batch_size  = size_batch,
                                                 num_workers = num_workers, )

dataset_validate  = TensorDatasetWithTransform(spi_data_validate.unsqueeze(1).numpy(), transform_list = transform_list)
dataloader_validate = torch.utils.data.DataLoader( dataset_validate,
                                                   shuffle     = False,
                                                   pin_memory  = True,
                                                   batch_size  = size_batch,
                                                   num_workers = num_workers, )


# [[[ MODEL ]]]
# Config the channels in the network...
model_dir = 'model'
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
model = NeurOrient(spi_data['pixel_position_reciprocal'],
                   volume_type='intensity',
                   radial_scale_configs=radial_scale_configs)

# Set device...
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# [[[ CRITERION ]]]
criterion = nn.MSELoss()

# [[[ OPTIMIZER ]]]
param_iter = model.module.parameters() if hasattr(model, "module") else model.parameters()
optimizer = optim.AdamW(param_iter,
                        lr = lr,
                        weight_decay = weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode           = 'min',
                                         factor         = 2e-1,
                                         patience       = 10,
                                         threshold      = 1e-4,
                                         threshold_mode ='rel',
                                         verbose        = True)


# [[[ TRAIN LOOP ]]]
max_epochs = 1000

# From a prev training???
epoch_min = 0
loss_min  = float('inf')
if path_chkpt_prev is not None:
    epoch_min, loss_min = load_checkpoint(model, optimizer, scheduler, path_chkpt_prev)
    ## epoch_min, loss_min = load_checkpoint(model, None, None, path_chkpt_prev)
    epoch_min += 1    # Next epoch
    logger.info(f"PREV - epoch_min = {epoch_min}, loss_min = {loss_min}")

# Compile the model...
# [CAVEAT] It does not like some interpolation methods.
if compiles_model:
    print("compiling the model... (takes a few minute)")
    ## unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

print(f"Current timestamp: {timestamp}")

chkpt_saving_period = 5
epoch_unstable_end  = 40
for epoch in tqdm.tqdm(range(max_epochs)):
    epoch += epoch_min

    # Uses mixed precision???
    if uses_mixed_precision: scaler = torch.cuda.amp.GradScaler()

    # ___/ TRAIN \___
    # Turn on training related components in the model...
    model.train()

    # Fetch batches...
    train_loss_list = []
    batch_train = tqdm.tqdm(enumerate(dataloader_train), total = len(dataloader_train))
    for batch_idx, batch_entry in batch_train:
        # Unpack the batch entry and move them to device...
        batch_input  = batch_entry[0]    # target is input itself
        batch_input  = batch_input.to(device, dtype = torch.float)
        batch_target = batch_input

        # Forward, backward and update...
        if uses_mixed_precision:
            with torch.cuda.amp.autocast(dtype = torch.float16):
                # Forward pass...
                batch_output = model(batch_input)

                # Calculate the loss...
                loss = criterion(batch_output, batch_target)
                loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus

            # Backward pass, optional gradient clipping and optimization...
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass...
            batch_output = model(batch_input)

            # Calculate the loss...
            loss = criterion(batch_output, batch_target)
            loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus

            # Backward pass, optional gradient clipping and optimization...
            optimizer.zero_grad()
            loss.backward()
            if grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Reporting...
        train_loss_list.append(loss.item())

    train_loss_mean = np.mean(train_loss_list)
    logger.info(f"MSG (device:{device}) - epoch {epoch}, mean train loss = {train_loss_mean:.8f}")


    # ___/ VALIDATE \___
    model.eval()

    # Fetch batches...
    validate_loss_list = []
    batch_validate = tqdm.tqdm(enumerate(dataloader_validate), total = len(dataloader_validate))
    for batch_idx, batch_entry in batch_validate:
        # Unpack the batch entry and move them to device...
        batch_input  = batch_entry[0]    # target is input itself
        batch_input  = batch_input.to(device, dtype = torch.float)
        batch_target = batch_input

        # Forward only...
        with torch.no_grad():
            if uses_mixed_precision:
                with torch.cuda.amp.autocast(dtype = torch.float16):
                    # Forward pass...
                    batch_output = model.forward_eval(batch_input)

                    # Calculate the loss...
                    loss = criterion(batch_output, batch_target)
                    loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus
            else:
                # Forward pass...
                batch_output = model.forward_eval(batch_input)

                # Calculate the loss...
                loss = criterion(batch_output, batch_target)
                loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus

        # Reporting...
        validate_loss_list.append(loss.item())

    validate_loss_mean = np.mean(validate_loss_list)
    logger.info(f"MSG (device:{device}) - epoch {epoch}, mean val   loss = {validate_loss_mean:.8f}")

    # Report the learning rate used in the last optimization...
    lr_used = optimizer.param_groups[0]['lr']
    logger.info(f"MSG (device:{device}) - epoch {epoch}, lr used = {lr_used}")

    # Update learning rate in the scheduler...
    scheduler.step(validate_loss_mean)


    # ___/ SAVE CHECKPOINT??? \___
    if validate_loss_mean < loss_min:
        loss_min = validate_loss_mean

        if (epoch % chkpt_saving_period == 0) or (epoch > epoch_unstable_end):
            fl_chkpt   = f"{timestamp}.epoch_{epoch}.chkpt"
            path_chkpt = os.path.join(drc_chkpt, fl_chkpt)
            save_checkpoint(model, optimizer, scheduler, epoch, loss_min, path_chkpt)
            logger.info(f"MSG (device:{device}) - save {path_chkpt}")
