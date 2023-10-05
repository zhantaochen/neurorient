from .configurator import Configurator

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