from .configurator import Configurator

"""
This is the default configuration for the model.
The configuration will be overwritten by the user's configuration file.
"""

_CONFIG = Configurator()
with _CONFIG.enable_auto_create():
    _CONFIG.MODEL.BACKBONE.RES_TYPE       = 50

    _CONFIG.MODEL.BIFPN.NUM_BLOCKS        = 1
    _CONFIG.MODEL.BIFPN.NUM_FEATURES      = 64
    _CONFIG.MODEL.BIFPN.NUM_LEVELS        = 5
    _CONFIG.MODEL.BIFPN.BN.EPS            = 1e-5
    _CONFIG.MODEL.BIFPN.BN.MOMENTUM       = 1e-1
    _CONFIG.MODEL.BIFPN.RELU_INPLACE      = False
    _CONFIG.MODEL.BIFPN.DOWN_SCALE_FACTOR = 0.5
    _CONFIG.MODEL.BIFPN.UP_SCALE_FACTOR   = 2
    _CONFIG.MODEL.BIFPN.FUSION.EPS        = 1e-5

    # [TODO] Need to set up resnet18 as well
    _CONFIG.MODEL.BACKBONE.OUTPUT_CHANNELS = {
        "relu"   : 64,
        "layer1" : 256,
        "layer2" : 512,
        "layer3" : 1024,
        "layer4" : 2048,
    }

    _CONFIG.MODEL.REGRESSOR_HEAD.IN_FEATURES  = _CONFIG.MODEL.BIFPN.NUM_FEATURES * 64 * 64
    _CONFIG.MODEL.REGRESSOR_HEAD.OUT_FEATURES = 6

    _CONFIG.MODEL.LOSS_SCALE_FACTOR = 10

    _CONFIG.MODEL.RESNET2ROTMAT.SCALE = -1
