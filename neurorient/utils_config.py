import os
from .configurator import Configurator

import torch
from torch.utils.data import TensorDataset
from neurorient.dataset         import TensorDatasetWithTransform, DictionaryDataset
from neurorient.image_transform import RandomPatch, PhotonFluctuation, PoissonNoise, GaussianNoise, BeamStopMask

from tqdm import tqdm

def flags_long2short(long_flags):
    """ Handy function to convert long flags to short flags, e.g. "f1_p1_g1_b1_c1" to "fpgbc".
        Created by ChatGPT GPT-4 model.
    """
    # Split the string by underscores
    parts = long_flags.split('_')
    result = []

    # Iterate over the split parts
    for part in parts:
        # Check if the part has a length of 2 and ends with "1"
        if len(part) == 2 and part[1] == '1':
            # Append the first character of the part to the result
            result.append(part[0])

    # Join the result list to get the final string
    out_str = ''.join(result)
    if len(out_str) == 0:
        return long_flags
    else:
        return ''.join(result)

def flags_short2long(short_flags):
    """ Handy function to convert long flags to short flags, e.g. "fpgbc" to "f1_p1_g1_b1_c1".
        Created by ChatGPT GPT-4 model.
    """
    # Define the available flags and their default values
    available_flags = {'f': '0', 'p': '0', 'g': '0', 'b': '0', 'c': '0'}
    
    # Iterate over the characters in the short flags
    for char in short_flags:
        # Check if the character is a valid flag
        if char in available_flags:
            # Set the flag to '1'
            available_flags[char] = '1'
    
    # Convert the flags dictionary to the long format string
    long_flags = '_'.join([f'{key}{value}' for key, value in available_flags.items()])
    return long_flags


def prepare_Slice2RotMat_config(config):
    if config.MODEL.USE_BIFPN:
        out_config = {
            'size': config.MODEL.BACKBONE.RES_TYPE, 
            'pretrained': config.MODEL.BACKBONE.PRETRAIN,
            'num_features': config.MODEL.BIFPN.NUM_FEATURES,
            'num_blocks': config.MODEL.BIFPN.NUM_BLOCKS,
            'num_levels': config.MODEL.BIFPN.NUM_LEVELS,
            'regressor_out_features': config.MODEL.REGRESSOR_HEAD.OUT_FEATURES,
            'scale': config.MODEL.RESNET2ROTMAT.SCALE
        }
    else:
        out_config = {
            'size': config.MODEL.BACKBONE.RES_TYPE, 
            'pretrained': config.MODEL.BACKBONE.PRETRAIN
        }
    return out_config

def prepare_IntensityNet_config(config):
    out_config = {
        'dim_hidden': int(config.MODEL.INTENSITY_NET.DIM_HIDDEN),
        'num_layers': int(config.MODEL.INTENSITY_NET.NUM_LAYERS)
    }
    return out_config

def _prepare_optimization_config(config_dict):
    out_config = {}
    
    for key, value in config_dict.items():
        # If the value is another dictionary, process it recursively
        if isinstance(value, dict):
            out_config[key.lower()] = _prepare_optimization_config(value)
        else:
            if key.lower() in ['lr', 'weight_decay', 'min_lr']:
                out_config[key.lower()] = float(value)
            else:
                out_config[key.lower()] = value
            
    return out_config

def prepare_optimization_config(config):
    return _prepare_optimization_config(config.OPTIM.to_dict())


def prepare_dataset(config, data_input, verbose=True):
    # Current script's directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    data = config.DATASET.INCREASE_FACTOR * data_input.clone()
    del data_input
    transform_list = []

    if config.DATASET.USES_PHOTON_FLUCTUATION:
        # set up photon fluctuation transformation
        photon_fluctuation = PhotonFluctuation(
            os.path.join(script_dir, 'data/image_distribution_by_photon_count.npy'),
            return_mask=False)
        transform_list.append(photon_fluctuation)
        if verbose:
            print(f'transformation: photon fluctuation applied to training and validation datasets.')


    if config.DATASET.USES_POISSON_NOISE:
        poisson_noise = PoissonNoise(return_mask=False)
        transform_list.append(poisson_noise)
        if verbose:
            print(f'transformation: poisson noise applied to training and validation datasets.')


    if config.DATASET.USES_GAUSSIAN_NOISE:
        gaussian_noise = GaussianNoise(sigma=config.DATASET.GAUSSIAN_NOISE.SIGMA, return_mask=False)
        transform_list.append(gaussian_noise)
        if verbose:
            print(f'transformation: gaussian noise applied to training and validation datasets.')


    if config.DATASET.USES_BEAM_STOP_MASK:
        beam_stop_mask = BeamStopMask(width              = config.DATASET.BEAM_STOP_MASK.WIDTH, 
                                      radius             = config.DATASET.BEAM_STOP_MASK.RADIUS, 
                                      input_size         = data.shape[-2:],
                                      mask_orientation   = config.DATASET.BEAM_STOP_MASK.ORIENTATION,
                                      return_mask        = True)
        transform_list.append(beam_stop_mask)
        if verbose:
            print(f'transformation: beam stop mask applied to training and validation datasets.')
        
        
    if config.DATASET.USES_RANDOM_PATCH:
        # set up random patch transformation
        num_patch       = config.DATASET.PATCH.NUM_PATCHES
        size_patch_min  = config.DATASET.PATCH.SIZE_PATCH_MIN
        size_patch_max  = config.DATASET.PATCH.SIZE_PATCH_MAX
        random_patch = RandomPatch(num_patch       = num_patch,
                                   size_patch_min  = size_patch_min,
                                   size_patch_max  = size_patch_max,
                                   return_mask     = True)
        transform_list.append(random_patch)
        if verbose:
            print(f'transformation: random patch applied to training and validation datasets.')
        
        
    if len(transform_list) > 0:
        transform_list   = tuple(transform_list)
        _dataset    = TensorDatasetWithTransform(
            data.unsqueeze(1), transform_list = transform_list, seed=config.TRAINING.SEED)

        if verbose:
            print(f'{len(transform_list)} transformations applied to training and validation datasets.')
        
        data_dict = {key: [] for key in _dataset[0].keys()}

        if verbose:
            pbar = tqdm(enumerate(_dataset), total=len(_dataset))
        else:
            pbar = enumerate(_dataset)
        for i, d in pbar:
            for _key in d.keys():
                data_dict[_key].append(d[_key])
        for _key in data_dict.keys():
            data_dict[_key] = torch.stack(data_dict[_key], dim=0)
        dataset = DictionaryDataset(**data_dict)
        del data_dict
    else:
        dataset = TensorDataset(data.unsqueeze(1))
        if verbose:
            print(f'NO transformation applied to training and validation datasets.')

    if verbose:
        print(f'created dataset with {len(dataset)} images.')

    return dataset