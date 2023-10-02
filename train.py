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

from neurorient.model           import NeurOrientLightning
from neurorient.data            import TensorDatasetWithTransform
from neurorient.logger          import Logger
from neurorient.image_transform import RandomPatch
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

# args = argparse.Namespace(
#     yaml_file='/global/homes/z/zhantao/Projects/NeuralOrientationMatching/base_config_resnet.yaml')

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
poisson           = merged_config.DATASET.POISSON
increase_factor   = merged_config.DATASET.INCREASE_FACTOR
num_images        = merged_config.DATASET.NUM_IMG
data_file_name = f'{pdb}_increase{increase_factor}_poisson{poisson}_num{num_images//1000}K.pt'
logger.log(f'data read from {data_file_name}')

# necessary info to define datasets
frac_train        = merged_config.DATASET.FRAC_TRAIN
size_batch        = merged_config.DATASET.BATCH_SIZE
num_workers       = merged_config.DATASET.NUM_WORKERS
uses_random_patch = merged_config.DATASET.USES_RANDOM_PATCH


# ...Training
max_epochs           = merged_config.TRAINING.MAX_EPOCHS
num_gpus             = min(torch.cuda.device_count(), merged_config.TRAINING.NUM_GPUS)
logger.log(f'training the model with {max_epochs} epochs and {num_gpus} GPUs')

# %%
# [[[ DATASET ]]]
spi_data = torch.load(os.path.join(dir_dataset, data_file_name))

# Set global seed and split data...
data              = spi_data['intensities']
spi_data_train    = data[:int(len(data) * frac_train) ]
spi_data_validate = data[ int(len(data) * frac_train):]

# Set world seed and set up transformation rules
if uses_random_patch:
    num_patch    = 200
    size_patch_y = 5
    size_patch_x = 5
    var_patch_y  = 0.2
    var_patch_x  = 0.2
    returns_mask = False
    random_patch = RandomPatch(num_patch    = num_patch,
                               size_patch_y    = size_patch_y,
                               size_patch_x    = size_patch_x,
                               var_patch_y     = var_patch_y,
                               var_patch_x     = var_patch_x,
                               returns_mask    = returns_mask)
    transform_list   = ( random_patch, )
    dataset_train    = TensorDatasetWithTransform(spi_data_train.unsqueeze(1).numpy(), transform_list = transform_list, uses_norm = False)
    dataset_validate = TensorDatasetWithTransform(spi_data_validate.unsqueeze(1).numpy(), transform_list = transform_list, uses_norm = False)
else:
    dataset_train    = TensorDataset(spi_data_train.unsqueeze(1))
    dataset_validate = TensorDataset(spi_data_validate.unsqueeze(1))

logger.log(f'created training dataset with {len(dataset_train)} images and validation dataset with {len(dataset_validate)} images.')
    

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

trainer = L.Trainer(
    max_epochs=max_epochs, accelerator='gpu',
    callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
    log_every_n_steps=1, devices=num_gpus,
    enable_checkpointing=True, default_root_dir=dir_chkpt)

# dump configuration to file for later reference
dump_yaml_fname = Path(os.path.join(trainer.logger.save_dir, 'lightning_logs', f'version_{trainer.logger.version}', 'input.yaml'))
dump_yaml_fname.parent.mkdir(parents=True, exist_ok=True)
merged_config.dump_to_file(dump_yaml_fname)

dump_log_fname = Path(os.path.join(trainer.logger.save_dir, 'lightning_logs', f'version_{trainer.logger.version}', 'log.txt'))
dump_log_fname.parent.mkdir(parents=True, exist_ok=True)
logger.dump_to_file(dump_log_fname)

trainer.fit(model, dataloader_train, dataloader_validate)


