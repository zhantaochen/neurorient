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
import argparse
import numpy as np
import pprint

import torch
from torch.utils.data import TensorDataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.strategies import DDPStrategy

from neurorient.model           import NeurOrientLightning
from neurorient.dataset         import TensorDatasetWithTransform, DictionaryDataset
from neurorient.logger          import Logger
from neurorient.image_transform import RandomPatch, PhotonFluctuation, PoissonNoise, GaussianNoise, BeamStopMask
from neurorient.configurator    import Configurator
# from neurorient.lr_scheduler    import CosineLRScheduler
from neurorient.config          import _CONFIG
from neurorient.utils_config    import (
    prepare_Slice2RotMat_config, prepare_IntensityNet_config, prepare_optimization_config)

torch.autograd.set_detect_anomaly(False)    # [WARNING] Making it True may throw errors when using bfloat16
                                            # Reference: https://discuss.pytorch.org/t/convolutionbackward0-returned-nan-values-in-its-0th-output/175571/4
                                            
logger = Logger()

# %%
# [[[ ARG ]]]
parser = argparse.ArgumentParser(description="Load training configuration from a YAML file to a dictionary.")
parser.add_argument('-yf', '--yaml_file', help="Path to the YAML file", dest='yaml_file', type=str, required=True)

args = parser.parse_args()

# %%
# [[[ HYPER-PARAMERTERS ]]]
# Load CONFIG from YAML
fl_yaml = args.yaml_file

with open(fl_yaml, 'r') as fh:
    config_dict = yaml.safe_load(fh)
CONFIG = Configurator.from_dict(config_dict)
logger.log(f"loaded configuration from yaml_file: {fl_yaml}.")

merged_config = CONFIG.merge_with_priority(_CONFIG, self_has_priority=True)
logger.log(f"overwrite default model configurations with customed configurations.")


if hasattr(merged_config.TRAINING, 'SEED'):
    L.seed_everything(merged_config.TRAINING.SEED)
    logger.log(f"SEED set to {merged_config.TRAINING.SEED}.")
else:
    logger.log(f"SEED not specified and not set.")

# ...Checkpoint
dir_chkpt           = Path(os.path.join(merged_config.TRAINING.BASE_DIRECTORY, merged_config.TRAINING.CHKPT_DIRECTORY))
dir_chkpt.mkdir(parents=True, exist_ok=True)
logger.log(f"checkpoints will be saved to {dir_chkpt}.")

# ...Dataset
dir_dataset       = merged_config.DATASET.DATASET_DIRECTORY
# necessary info to fetch data file name
pdb               = merged_config.DATASET.PDB
num_images        = merged_config.DATASET.NUM_IMG
data_file_name = f'{pdb}_increase1_poissonFalse_num{num_images//1000}K.pt'
logger.log(f'data read from {data_file_name}')

# necessary info to define datasets
frac_train        = merged_config.DATASET.FRAC_TRAIN
size_batch        = merged_config.DATASET.BATCH_SIZE
num_workers       = merged_config.DATASET.NUM_WORKERS

# ...Training
max_epochs           = merged_config.TRAINING.MAX_EPOCHS
num_gpus             = min(torch.cuda.device_count(), merged_config.TRAINING.NUM_GPUS)
logger.log(f'training the model with {max_epochs} epochs and {num_gpus} GPUs')

# %%
# [[[ DATASET ]]]
spi_data = torch.load(os.path.join(dir_dataset, data_file_name))

# Set global seed and split data...
data              = spi_data['intensities'] * merged_config.DATASET.INCREASE_FACTOR
spi_data_train    = data[:int(len(data) * frac_train) ]
spi_data_validate = data[ int(len(data) * frac_train):]

transform_list = []

if merged_config.DATASET.USES_PHOTON_FLUCTUATION:
    # set up photon fluctuation transformation
    photon_fluctuation = PhotonFluctuation(
        'neurorient/data/image_distribution_by_photon_count.npy',
        return_mask=False)
    transform_list.append(photon_fluctuation)
    logger.log(f'transformation: photon fluctuation applied to training and validation datasets.')


if merged_config.DATASET.USES_POISSON_NOISE:
    poisson_noise = PoissonNoise(return_mask=False)
    transform_list.append(poisson_noise)
    logger.log(f'transformation: poisson noise applied to training and validation datasets.')


if merged_config.DATASET.USES_GAUSSIAN_NOISE:
    gaussian_noise = GaussianNoise(sigma=merged_config.DATASET.GAUSSIAN_NOISE.SIGMA, return_mask=False)
    transform_list.append(gaussian_noise)
    logger.log(f'transformation: gaussian noise applied to training and validation datasets.')


if merged_config.DATASET.USES_BEAM_STOP_MASK:
    beam_stop_mask = BeamStopMask(width              = merged_config.DATASET.BEAM_STOP_MASK.WIDTH, 
                                  radius             = merged_config.DATASET.BEAM_STOP_MASK.RADIUS, 
                                  input_size         = data.shape[-2:],
                                  mask_orientation   = merged_config.DATASET.BEAM_STOP_MASK.ORIENTATION,
                                  return_mask        = True)
    transform_list.append(beam_stop_mask)
    logger.log(f'transformation: beam stop mask applied to training and validation datasets.')
    
    
if merged_config.DATASET.USES_RANDOM_PATCH:
    # set up random patch transformation
    num_patch       = merged_config.DATASET.PATCH.NUM_PATCHES
    size_patch_min  = merged_config.DATASET.PATCH.SIZE_PATCH_MIN
    size_patch_max  = merged_config.DATASET.PATCH.SIZE_PATCH_MAX
    random_patch = RandomPatch(num_patch       = num_patch,
                               size_patch_min  = size_patch_min,
                               size_patch_max  = size_patch_max,
                               return_mask     = True)
    transform_list.append(random_patch)
    logger.log(f'transformation: random patch applied to training and validation datasets.')
    
    
if len(transform_list) > 0:
    transform_list   = tuple(transform_list)
    _dataset_train    = TensorDatasetWithTransform(
        spi_data_train.unsqueeze(1), transform_list = transform_list, seed=merged_config.TRAINING.SEED)
    _dataset_validate = TensorDatasetWithTransform(
        spi_data_validate.unsqueeze(1), transform_list = transform_list, seed=merged_config.TRAINING.SEED)
    
    logger.log(f'{len(transform_list)} transformations applied to training and validation datasets.')
    
    train_data = {key: [] for key in _dataset_train[0].keys()}
    for i, d in enumerate(_dataset_train):
        for _key in d.keys():
            train_data[_key].append(d[_key])
    for _key in train_data.keys():
        train_data[_key] = torch.stack(train_data[_key], dim=0)
    dataset_train = DictionaryDataset(**train_data)
    del train_data
    
    validate_data = {key: [] for key in _dataset_validate[0].keys()}
    for i, d in enumerate(_dataset_validate):
        for _key in d.keys():
            validate_data[_key].append(d[_key])
    for _key in validate_data.keys():
        validate_data[_key] = torch.stack(validate_data[_key], dim=0)
    dataset_validate = DictionaryDataset(**validate_data)
    del validate_data
    
    logger.log(f'created dictionary datasets for training and validation.')
else:
    dataset_train    = TensorDataset(spi_data_train.unsqueeze(1))
    dataset_validate = TensorDataset(spi_data_validate.unsqueeze(1))
    logger.log(f'NO random patch transformation applied to training and validation datasets.')

logger.log(f'created training dataset with {len(dataset_train)} images and validation dataset with {len(dataset_validate)} images.')

# %%

# lightning will handle the samplers for those dataloaders
sampler_train    = None
dataloader_train = torch.utils.data.DataLoader( dataset_train,
                                                sampler     = sampler_train,
                                                shuffle     = True,
                                                pin_memory  = True,
                                                batch_size  = size_batch,
                                                num_workers = num_workers, )

sampler_validate    = None
dataloader_validate = torch.utils.data.DataLoader( dataset_validate,
                                                   sampler     = sampler_validate,
                                                   shuffle     = False,
                                                   pin_memory  = True,
                                                   batch_size  = size_batch,
                                                   num_workers = num_workers, )

# %%
# [[[ MODEL ]]]
over_sampling = merged_config.MODEL.OVERSAMPLING
photons_per_pulse = merged_config.DATASET.INCREASE_FACTOR * 1e12
config_optimization = prepare_optimization_config(merged_config)
config_intensitynet = prepare_IntensityNet_config(merged_config)
config_slice2rotmat = prepare_Slice2RotMat_config(merged_config)
model = NeurOrientLightning(
    spi_data['pixel_position_reciprocal'],
    over_sampling=over_sampling, 
    photons_per_pulse=photons_per_pulse,
    use_bifpn=merged_config.MODEL.USE_BIFPN,
    use_fluctuation_predictor=True if merged_config.DATASET.USES_PHOTON_FLUCTUATION else False,
    config_slice2rotmat=config_slice2rotmat,
    config_intensitynet=config_intensitynet,
    config_optimization=config_optimization
)
logger.log( 
    'arguments being used in building the model:\n',
    f'over_sampling={over_sampling}\n',
    f'photons_per_pulse={photons_per_pulse:.2e}\n',
    'config_slice2rotmat: ', '\n', pprint.pformat(config_slice2rotmat), '\n',
    'config_optimization: ', '\n', pprint.pformat(config_optimization))

logger.log(
    "model created with the following architecture:\n",
    pprint.pformat(model)
)

# %%
checkpoint_callback = ModelCheckpoint(
    every_n_train_steps=10, save_last=True, save_top_k=1, monitor="val_loss",
    filename=f'{pdb}-{{epoch}}-{{step}}'
)

torch.set_float32_matmul_precision('high')

ddp = DDPStrategy(process_group_backend="nccl")
trainer = L.Trainer(
    max_epochs=max_epochs, accelerator='gpu', strategy=ddp,
    callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
    log_every_n_steps=1, devices=num_gpus, sync_batchnorm = True,
    enable_checkpointing=True, default_root_dir=dir_chkpt)

# dump configuration to file for later reference
dump_yaml_fname = Path(os.path.join(trainer.logger.log_dir, 'input.yaml'))
dump_yaml_fname.parent.mkdir(parents=True, exist_ok=True)
merged_config.dump_to_file(dump_yaml_fname)

dump_log_fname = Path(os.path.join(trainer.logger.log_dir, 'log.txt'))
dump_log_fname.parent.mkdir(parents=True, exist_ok=True)
logger.dump_to_file(dump_log_fname)

trainer.fit(model, dataloader_train, dataloader_validate)


