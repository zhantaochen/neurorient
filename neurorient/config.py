from .configurator import Configurator

"""
This is the default configuration for the model.
It contains all changable parameters for the model.
The configuration will be overwritten by the user's configuration file.
"""
_CONFIG = Configurator()
with _CONFIG.enable_auto_create():
    # the followings are mainly called in the utils_config.py
    # to set up the default configuration for the model
    _CONFIG.MODEL.BACKBONE.RES_TYPE       = 18
    _CONFIG.MODEL.BACKBONE.PRETRAIN       = True

    # whether to use the BiFPN module
    _CONFIG.MODEL.USE_BIFPN               = True

    # if use bifpn, the following parameters will be used
    _CONFIG.MODEL.BIFPN.NUM_BLOCKS        = 3
    _CONFIG.MODEL.BIFPN.NUM_FEATURES      = 64
    _CONFIG.MODEL.BIFPN.NUM_LEVELS        = 3
    _CONFIG.MODEL.BIFPN.RELU_INPLACE      = False
    _CONFIG.MODEL.BIFPN.OUTPUT_CHANNELS_FROM_BACKBONE = {
        "relu"   : 64,
        "layer1" : 256,
        "layer2" : 512,
        "layer3" : 1024,
        "layer4" : 2048,
    }
    # TODO: REMOVE THIS
    _CONFIG.MODEL.REGRESSOR_HEAD.IN_FEATURES  = _CONFIG.MODEL.BIFPN.NUM_FEATURES * 4 * 4
    _CONFIG.MODEL.REGRESSOR_HEAD.OUT_FEATURES = 6
    _CONFIG.MODEL.RESNET2ROTMAT.SCALE = -1
    
    # the following parameters are used to set up the volume predictor
    _CONFIG.MODEL.INTENSITY_NET.DIM_HIDDEN = 256
    _CONFIG.MODEL.INTENSITY_NET.NUM_LAYERS = 5


"""
The bifpn_internal_config is used to store the configuration for the BiFPN module only.
This configuration is not meant to be used by the user.
"""
bifpn_internal_config = Configurator()
with bifpn_internal_config.enable_auto_create():
    bifpn_internal_config.MODEL.BIFPN.DOWN_SCALE_FACTOR = 0.5
    bifpn_internal_config.MODEL.BIFPN.UP_SCALE_FACTOR   = 2
    bifpn_internal_config.MODEL.BIFPN.BN.EPS            = 1e-5
    bifpn_internal_config.MODEL.BIFPN.BN.MOMENTUM       = 1e-1
    bifpn_internal_config.MODEL.BIFPN.FUSION.EPS        = 1e-5