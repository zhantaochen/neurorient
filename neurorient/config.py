from .configurator import Configurator

CONFIG = Configurator()
with CONFIG.enable_auto_create():
    CONFIG.BACKBONE.RES_TYPE       = 50

    CONFIG.BIFPN.NUM_BLOCKS        = 1
    CONFIG.BIFPN.NUM_FEATURES      = 64
    CONFIG.BIFPN.NUM_LEVELS        = 5
    CONFIG.BIFPN.BN.EPS            = 1e-5
    CONFIG.BIFPN.BN.MOMENTUM       = 1e-1
    CONFIG.BIFPN.RELU_INPLACE      = False
    CONFIG.BIFPN.DOWN_SCALE_FACTOR = 0.5
    CONFIG.BIFPN.UP_SCALE_FACTOR   = 2
    CONFIG.BIFPN.FUSION.EPS        = 1e-5

    # [TODO] Need to set up resnet18 as well
    CONFIG.BACKBONE.OUTPUT_CHANNELS = {
        "relu"   : 64,
        "layer1" : 256,
        "layer2" : 512,
        "layer3" : 1024,
        "layer4" : 2048,
    }

    CONFIG.REGRESSOR_HEAD.IN_FEATURES  = CONFIG.BIFPN.NUM_FEATURES * 64 * 64
    CONFIG.REGRESSOR_HEAD.OUT_FEATURES = 6

    CONFIG.LOSS_SCALE_FACTOR = 10

    CONFIG.RESNET2ROTMAT.SCALE = 0
